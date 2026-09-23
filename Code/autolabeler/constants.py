# -*- coding: utf-8 -*-
"""앱 기본 상수와 기본 설정값을 정의합니다."""

from __future__ import annotations

from .models import ModelOption

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 빌드/배포본을 화면에서 식별하기 위한 표시 버전입니다.
APP_VERSION = "ver.26.0918.2-seg"

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
    "#FF8800",
    "#00A86B",
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
    "delete_current_pair": "Delete",
    "review_current_image": "Z",
    "find_unreviewed_image": "F",
    "toggle_theme": "T",
    "open_settings": "Ctrl+,",
}

# 설정 창과 좌측 안내 목록에서 단축키 기능명을 일관되게 표시합니다.
SHORTCUT_LABELS = {
    "draw_box": "세그 그리기",
    "mask_area": "마스킹",
    "edit_box": "편집",
    "cancel_mode": "작업 취소",
    "prev_image": "이전 이미지",
    "next_image": "다음 이미지",
    "reset_view": "화면 맞춤/빠른 확대",
    "delete_last_box": "선택/최근 라벨 삭제",
    "delete_current_pair": "이미지+라벨 파일 이동",
    "review_current_image": "검토 완료/빈 이미지",
    "find_unreviewed_image": "미작업 찾기",
    "toggle_theme": "테마 전환",
    "open_settings": "설정 열기",
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
    "10": "0",
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
        "auto_label_max_points": "24",
        "polygon_point_count": "24",
        "brush_size": "18",
    },
}

# 세그멘테이션 프로젝트에서는 YOLO11 Segmentation 계열 가중치를 기본값으로 사용합니다.
MODEL_OPTIONS = [
    ModelOption(
        name="YOLO26 Nano Seg",
        weight_name="yolo26n-seg.pt",
        description="YOLO26 계열의 경량 세그멘테이션 모델입니다.",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
    ),
    ModelOption(
        name="YOLO26 Large Seg",
        weight_name="yolo26l-seg.pt",
        description="YOLO26 계열의 정확도 우선 대형 세그멘테이션 모델입니다.",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt",
    ),
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
    ModelOption(
        name="YOLO11 Large Seg",
        weight_name="yolo11l-seg.pt",
        # Large 모델은 속도보다 세그멘테이션 정확도를 우선할 때 선택합니다.
        description="더 무겁지만 정밀한 세그멘테이션 결과를 기대할 수 있는 Large 모델입니다.",
        download_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
    ),
]
