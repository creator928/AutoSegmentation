# -*- coding: utf-8 -*-
"""외부 Python 환경에서 PT 기반 오토 세그멘테이션을 수행합니다."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from autolabeler.services.worklog_service import (
    auto_label_allowed, commit_auto_label, read_explicit_statuses, worklog_lock,
)
from uuid import uuid4


def parse_args() -> argparse.Namespace:
    """명령행 인자를 해석합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--conf", required=True, type=float)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ultralytics-dir", required=True)
    parser.add_argument("--stop-index-path", required=True)
    parser.add_argument("--worklog-path", required=True)
    parser.add_argument("--max-polygon-points", type=int, default=24)
    args = parser.parse_args()
    if args.max_polygon_points < 3:
        parser.error("--max-polygon-points must be at least 3")
    return args


def limit_polygon_points(polygon, max_points: int):
    """닫힌 윤곽선을 근사하여 폴리곤별 최대 꼭짓점 수를 제한합니다."""
    if max_points < 3:
        raise ValueError("max_points must be at least 3")
    if len(polygon) <= max_points:
        return polygon
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    low, high = 0.0, cv2.arcLength(contour, True)
    best = contour
    # Douglas-Peucker 허용 오차를 탐색해 제한 이내에서 세부 윤곽을 최대한 남깁니다.
    for _ in range(32):
        epsilon = (low + high) / 2.0
        candidate = cv2.approxPolyDP(contour, epsilon, True)
        if len(candidate) > max_points:
            low = epsilon
        else:
            high = epsilon
            if len(candidate) >= 3:
                best = candidate
    points = best.reshape(-1, 2)
    if len(points) > max_points:
        # 대칭 윤곽 등에서 근사점 수가 4개에서 2개로 건너뛰어도 상한과 최소 3점을 보장합니다.
        indices = np.linspace(0, len(points), max_points, endpoint=False, dtype=int)
        points = points[indices]
    return points


def read_stop_index(stop_index_path: Path) -> int:
    """GUI가 기록한 현재 사용자 작업 경계 인덱스를 읽습니다."""
    if not stop_index_path.exists():
        raise RuntimeError("정지 경계 파일이 없어 자동 처리를 중단합니다.")
    try:
        value = int(stop_index_path.read_text(encoding="utf-8").strip())
        if value < -1:
            raise ValueError("invalid stop index")
        return value
    except (OSError, ValueError):
        raise RuntimeError("정지 경계를 읽지 못해 자동 처리를 중단합니다.")


def main() -> int:
    """manifest에 포함된 전체 이미지를 뒤에서부터 다시 오토 세그멘테이션합니다."""
    args = parse_args()
    os.environ["YOLO_CONFIG_DIR"] = args.ultralytics_dir

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    # 오토 라벨링은 진행 로그만 남기고 Ultralytics 기본 출력은 억제합니다.
    LOGGER.setLevel(logging.ERROR)

    model = YOLO(args.model)
    manifest_path = Path(args.manifest)
    stop_index_path = Path(args.stop_index_path)
    worklog_path = Path(args.worklog_path)
    work_dir = worklog_path.parent
    image_paths = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not image_paths:
        raise RuntimeError("오토 라벨 대상 이미지 목록이 비어 있습니다.")

    initial_stop_index = read_stop_index(stop_index_path)
    with worklog_lock(work_dir):
        initial_statuses = read_explicit_statuses(work_dir)
    total = sum(
        initial_statuses.get(path.stem) in {"a", "n"}
        for path in image_paths[max(initial_stop_index + 1, 0):]
    )
    processed_count = 0
    labeled_count = 0
    backup_dir = work_dir / "auto_label_backups" / uuid4().hex

    for image_index in range(len(image_paths) - 1, -1, -1):
        current_stop_index = read_stop_index(stop_index_path)
        if current_stop_index >= 0 and image_index <= current_stop_index:
            print(
                f"AUTO_LABELER_STOP|{image_index}|{current_stop_index}",
                flush=True,
            )
            break

        image_path = image_paths[image_index]
        if not auto_label_allowed(work_dir, image_path):
            print(f"AUTO_LABELER_SKIP|{image_path.name}|protected", flush=True)
            continue
        label_path = image_path.with_suffix(".txt")
        previous = label_path.read_bytes() if label_path.exists() else None
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            task="segment",
            device=args.device,
            conf=args.conf,
            iou=0.45,
            max_det=300,
            verbose=False,
        )

        lines: list[str] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            masks = getattr(results[0], "masks", None)
            if boxes is not None and masks is not None and masks.xyn is not None and boxes.cls is not None:
                class_list = boxes.cls.tolist()
                polygon_list = masks.xyn
                for class_value, polygon in zip(class_list, polygon_list):
                    polygon = limit_polygon_points(polygon, args.max_polygon_points)
                    if len(polygon) < 3:
                        continue
                    point_tokens: list[str] = []
                    for x_value, y_value in polygon:
                        point_tokens.append(f"{float(x_value):.6f} {float(y_value):.6f}")
                    lines.append(
                        f"{int(class_value)} {' '.join(point_tokens)}"
                    )

        # 추론 중 바뀐 정지 경계와 검토 상태를 저장 직전에 다시 확인합니다.
        current_stop_index = read_stop_index(stop_index_path)
        if current_stop_index >= 0 and image_index <= current_stop_index:
            break
        if not commit_auto_label(work_dir, image_paths, image_path,
                                 "\n".join(lines), backup_dir, previous):
            print(f"AUTO_LABELER_SKIP|{image_path.name}|changed", flush=True)
            continue
        if lines:
            labeled_count += 1

        processed_count += 1
        print(f"AUTO_LABELER_PROGRESS|{processed_count}|{max(total, processed_count)}", flush=True)
        print(f"오토 세그 완료: {image_path.name} / polygons={len(lines)}", flush=True)

    print(f"AUTO_LABELER_DONE|{processed_count}|{labeled_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
