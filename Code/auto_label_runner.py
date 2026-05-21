# -*- coding: utf-8 -*-
"""외부 Python 환경에서 PT 기반 오토 세그멘테이션을 수행합니다."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from autolabeler.services.worklog_service import load_worklog_statuses, save_worklog_statuses


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
    return parser.parse_args()


def read_stop_index(stop_index_path: Path) -> int:
    """GUI가 기록한 현재 사용자 작업 경계 인덱스를 읽습니다."""
    if not stop_index_path.exists():
        return -1
    try:
        return int(stop_index_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return -1


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
    total = max(0, len(image_paths) - max(initial_stop_index + 1, 0))
    processed_count = 0
    labeled_count = 0
    work_statuses = load_worklog_statuses(work_dir, image_paths)

    for image_index in range(len(image_paths) - 1, -1, -1):
        current_stop_index = read_stop_index(stop_index_path)
        if current_stop_index >= 0 and image_index <= current_stop_index:
            print(
                f"AUTO_LABELER_STOP|{image_index}|{current_stop_index}",
                flush=True,
            )
            break

        image_path = image_paths[image_index]
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
                    if len(polygon) < 3:
                        continue
                    point_tokens: list[str] = []
                    for x_value, y_value in polygon:
                        point_tokens.append(f"{float(x_value):.6f} {float(y_value):.6f}")
                    lines.append(
                        f"{int(class_value)} {' '.join(point_tokens)}"
                    )

        label_path = image_path.with_suffix(".txt")
        if lines:
            label_path.write_text("\n".join(lines), encoding="utf-8")
            labeled_count += 1
        elif label_path.exists():
            # 재추론 결과가 없으면 기존 자동 라벨도 제거해 현재 결과와 일치시킵니다.
            label_path.unlink()

        # 오토 라벨이 시도된 이미지는 결과 유무와 무관하게 자동 처리 상태로 기록합니다.
        work_statuses[image_path] = "a"
        work_statuses = save_worklog_statuses(work_dir, image_paths, work_statuses)

        processed_count += 1
        print(f"AUTO_LABELER_PROGRESS|{processed_count}|{max(total, processed_count)}", flush=True)
        print(f"오토 세그 완료: {image_path.name} / polygons={len(lines)}", flush=True)

    print(f"AUTO_LABELER_DONE|{processed_count}|{labeled_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
