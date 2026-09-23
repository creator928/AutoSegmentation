# -*- coding: utf-8 -*-
"""자매 프로그램의 이미지와 YOLO 라벨을 함께 회전하고 저장합니다."""

from dataclasses import replace
from pathlib import Path
import os
import shutil
import tempfile

from PyQt6.QtGui import QImage, QTransform

from .label_service import save_labels


def rotate_labels(labels: list, clockwise: bool) -> list:
    """화면 좌표계의 90도 회전을 폴리곤 또는 박스의 정규화 좌표에 적용합니다."""
    def rotate_point(x, y):
        return (1.0 - y, x) if clockwise else (y, 1.0 - x)

    rotated = []
    for label in labels:
        if hasattr(label, "points"):
            rotated.append(replace(label, points=[rotate_point(x, y) for x, y in label.points]))
        else:
            x, y = rotate_point(label.x_center, label.y_center)
            rotated.append(replace(label, x_center=x, y_center=y, width=label.height, height=label.width))
    return rotated


def rotate_image_and_labels(image_path: Path, labels: list, clockwise: bool) -> list:
    """두 파일을 먼저 준비하고 교체하며, 교체 실패 시 원본을 복구합니다."""
    image = QImage(str(image_path))
    if image.isNull():
        raise OSError("이미지를 불러오지 못했습니다.")
    rotated_image = image.transformed(QTransform().rotate(90 if clockwise else -90))
    rotated_labels = rotate_labels(labels, clockwise)
    label_path = image_path.with_suffix(".txt")
    staging = Path(tempfile.mkdtemp(prefix=".rotation-", dir=image_path.parent))
    staged_image = staging / image_path.name
    original_image = staging / "original-image"
    original_label = staging / "original-label"
    had_label = label_path.exists()
    keep_backup = False
    try:
        shutil.copy2(image_path, original_image)
        if had_label:
            shutil.copy2(label_path, original_label)
        if not rotated_image.save(str(staged_image)):
            raise OSError("회전된 이미지를 저장하지 못했습니다.")
        # 빈 라벨도 TXT로 남겨 네거티브 샘플 상태를 유지합니다.
        save_labels(staged_image, rotated_labels)
        try:
            os.replace(staged_image, image_path)
            os.replace(staged_image.with_suffix(".txt"), label_path)
        except OSError:
            try:
                os.replace(original_image, image_path)
                if had_label:
                    os.replace(original_label, label_path)
            except OSError as recovery_error:
                keep_backup = True
                raise OSError(f"원본 복구 실패. 백업 위치: {staging}") from recovery_error
            raise
    finally:
        if not keep_backup:
            shutil.rmtree(staging)
    return rotated_labels
