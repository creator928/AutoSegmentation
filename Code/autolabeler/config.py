# -*- coding: utf-8 -*-
"""앱 경로와 설정 파일 로딩/저장을 담당합니다."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .constants import DEFAULT_CONFIG
from .models import AppConfig, AppPaths


def get_app_root() -> Path:
    """실행 환경에 맞춰 앱 루트 경로를 계산합니다."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def get_runtime_code_dir(root_dir: Path) -> Path:
    """외부 Python으로 실행할 러너 스크립트 위치를 반환합니다."""
    if getattr(sys, "frozen", False):
        # PyInstaller onefile은 add-data 파일을 임시 해제 폴더에 풀어둡니다.
        return Path(getattr(sys, "_MEIPASS", root_dir)).resolve()
    return root_dir / "Code"


def build_paths() -> AppPaths:
    """프로젝트 고정 디렉토리 구조를 경로 객체로 구성합니다."""
    root_dir = get_app_root()
    return AppPaths(
        root_dir=root_dir,
        document_dir=root_dir / "Document",
        code_dir=get_runtime_code_dir(root_dir),
        data_dir=root_dir / "Data",
        work_dir=root_dir / "Work",
        model_dir=root_dir / "Data" / "models",
        settings_path=root_dir / "Data" / "settings.json",
        ultralytics_dir=root_dir / "Data" / "ultralytics",
    )


def cleanup_root_model_artifacts(paths: AppPaths) -> None:
    """앱 루트에 잘못 생성된 모델 파일이 있으면 Data/models로 일원화되도록 정리합니다."""
    for pattern in ("yolo*.pt", "yolo*.onnx"):
        for artifact_path in paths.root_dir.glob(pattern):
            if artifact_path.parent == paths.model_dir:
                continue
            canonical_path = paths.model_dir / artifact_path.name
            if canonical_path.exists() or artifact_path.parent == paths.root_dir:
                artifact_path.unlink(missing_ok=True)


def ensure_app_directories() -> AppPaths:
    """필수 디렉토리와 기본 설정 파일을 없으면 생성합니다."""
    paths = build_paths()
    for directory in (
        paths.document_dir,
        paths.data_dir,
        paths.work_dir,
        paths.model_dir,
        paths.ultralytics_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not paths.settings_path.exists():
        paths.settings_path.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    cleanup_root_model_artifacts(paths)
    return paths


def load_app_config() -> AppConfig:
    """설정 파일을 읽고 누락값을 기본값으로 보정합니다."""
    paths = ensure_app_directories()
    raw = json.loads(paths.settings_path.read_text(encoding="utf-8"))

    merged = {
        "theme_mode": raw.get("theme_mode", DEFAULT_CONFIG["theme_mode"]),
        "rectangle_input_mode": raw.get("rectangle_input_mode", DEFAULT_CONFIG["rectangle_input_mode"]),
        "selected_model": raw.get("selected_model", DEFAULT_CONFIG["selected_model"]),
        # 기존 설정 파일에 없는 신규 단축키도 기본값으로 유지되도록 개별 사전 병합을 수행합니다.
        "shortcuts": {**DEFAULT_CONFIG["shortcuts"], **raw.get("shortcuts", {})},
        "class_shortcuts": {**DEFAULT_CONFIG["class_shortcuts"], **raw.get("class_shortcuts", {})},
        "runtime_options": {**DEFAULT_CONFIG["runtime_options"], **raw.get("runtime_options", {})},
    }

    return AppConfig(
        theme_mode=merged["theme_mode"],
        rectangle_input_mode=merged["rectangle_input_mode"],
        selected_model=merged["selected_model"],
        shortcuts=merged["shortcuts"],
        class_shortcuts=merged["class_shortcuts"],
        runtime_options=merged["runtime_options"],
        paths=paths,
    )


def save_app_config(config: AppConfig) -> None:
    """메모리 상의 설정 객체를 JSON 파일로 저장합니다."""
    if config.paths is None:
        raise ValueError("경로 정보가 없는 설정은 저장할 수 없습니다.")

    payload = {
        "theme_mode": config.theme_mode,
        "rectangle_input_mode": config.rectangle_input_mode,
        "selected_model": config.selected_model,
        "shortcuts": config.shortcuts,
        "class_shortcuts": config.class_shortcuts,
        "runtime_options": config.runtime_options,
    }
    config.paths.settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
