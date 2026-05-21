# -*- coding: utf-8 -*-
"""앱 기본 상수와 기본 설정값을 정의합니다."""

from __future__ import annotations

from .models import ModelOption

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 클래스 인덱스별 기본 표시 색상입니다.
CLASS_COLORS = [
    "#00FF00",
    "#FF0000",
    "#0000FF",
    "#00FFFF",
    "#FF00FF",
    "#FFFF00",
    "#FFFFFF",
    "#000000",
]

DEFAULT_SHORTCUTS = {
    "draw_box": "W",
    "mask_area": "X",
    "edit_box": "E",
    "cancel_mode": "Q",
    "prev_image": "A",
    "next_image": "D",
    "reset_view": "S",
    "delete_last_box": "R",
    "toggle_theme": "T",
    "open_settings": "Ctrl+,",
}

DEFAULT_CLASS_SHORTCUTS = {
    "0": "`",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
}

DEFAULT_CONFIG = {
    "theme_mode": "light",
    "rectangle_input_mode": "click",
    "selected_model": "yolo11n-seg.pt",
    "shortcuts": DEFAULT_SHORTCUTS,
    "class_shortcuts": DEFAULT_CLASS_SHORTCUTS,
    "runtime_options": {
        "use_gpu": "true",
        "dataset_size": "100",
        "epochs": "50",
        "image_size": "640",
        "batch_size": "8",
        "project_name": "AutoSegmentationTrain",
        "auto_label_conf": "0.01",
        "polygon_point_count": "24",
        "brush_size": "18",
    },
}

# 세그멘테이션 프로젝트에서는 YOLO11 Segmentation 계열 가중치를 기본값으로 사용합니다.
MODEL_OPTIONS = [
    ModelOption(
        name="YOLO11 Nano Seg",
        weight_name="yolo11n-seg.pt",
        description="공식 배포 자산으로 바로 다운로드 가능한 경량 세그멘테이션 기본 모델입니다.",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
    ),
    ModelOption(
        name="YOLO11 Small Seg",
        weight_name="yolo11s-seg.pt",
        description="조금 더 무겁지만 정확도가 높을 수 있는 세그멘테이션 대안 모델입니다.",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
    ),
]
