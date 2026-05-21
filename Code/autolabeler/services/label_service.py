# -*- coding: utf-8 -*-
"""YOLO 세그멘테이션 라벨 파일 로딩과 저장을 담당합니다."""

from __future__ import annotations

from pathlib import Path

from ..constants import CLASS_COLORS
from ..models import LabelPolygon


def label_path_for_image(image_path: Path) -> Path:
    """이미지와 같은 이름의 YOLO 라벨 경로를 계산합니다."""
    return image_path.with_suffix(".txt")


def load_labels(image_path: Path) -> list[LabelPolygon]:
    """이미지에 대응하는 YOLO 세그멘테이션 라벨 파일을 읽습니다."""
    labels: list[LabelPolygon] = []
    txt_path = label_path_for_image(image_path)
    if not txt_path.exists():
        return labels

    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        class_index = int(parts[0])
        color_hex = CLASS_COLORS[class_index % len(CLASS_COLORS)]
        points: list[tuple[float, float]] = []
        for index in range(1, len(parts), 2):
            try:
                x_value = float(parts[index])
                y_value = float(parts[index + 1])
            except (IndexError, ValueError):
                points = []
                break
            points.append((x_value, y_value))
        if len(points) < 3:
            continue
        labels.append(
            LabelPolygon(
                class_index=class_index,
                points=points,
                color_hex=color_hex,
            )
        )
    return labels


def save_labels(image_path: Path, labels: list[LabelPolygon]) -> None:
    """현재 이미지의 모든 세그멘테이션 라벨을 YOLO 포맷으로 저장합니다."""
    txt_path = label_path_for_image(image_path)
    # 라벨이 하나도 없으면 파일 자체를 제거해 미작업 상태와 일치시킵니다.
    if not labels:
        if txt_path.exists():
            txt_path.unlink()
        return

    lines = []
    for label in labels:
        if len(label.points) < 3:
            continue
        point_tokens = " ".join(
            f"{x_value:.6f} {y_value:.6f}"
            for x_value, y_value in label.points
        )
        lines.append(
            f"{label.class_index} {point_tokens}"
        )
    if not lines:
        if txt_path.exists():
            txt_path.unlink()
        return
    txt_path.write_text("\n".join(lines), encoding="utf-8")
