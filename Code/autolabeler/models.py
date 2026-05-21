# -*- coding: utf-8 -*-
"""앱 전반에서 사용하는 데이터 구조를 정의합니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LabelPolygon:
    """YOLO 세그멘테이션 폴리곤과 화면 표시용 색상 정보를 함께 저장합니다."""

    class_index: int
    points: list[tuple[float, float]]
    color_hex: str


@dataclass
class AppPaths:
    """프로그램이 사용하는 주요 경로를 한곳에서 관리합니다."""

    root_dir: Path
    document_dir: Path
    code_dir: Path
    data_dir: Path
    work_dir: Path
    model_dir: Path
    settings_path: Path
    ultralytics_dir: Path


@dataclass
class ModelOption:
    """다운로드 가능한 모델 후보 정보를 표현합니다."""

    name: str
    weight_name: str
    description: str
    download_url: str


@dataclass
class AppConfig:
    """설정 파일에 저장되는 값을 메모리 객체로 다룹니다."""

    theme_mode: str
    rectangle_input_mode: str
    selected_model: str
    shortcuts: dict[str, str] = field(default_factory=dict)
    class_shortcuts: dict[str, str] = field(default_factory=dict)
    runtime_options: dict[str, str] = field(default_factory=dict)
    paths: AppPaths | None = None
