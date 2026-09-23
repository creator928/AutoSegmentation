# -*- coding: utf-8 -*-
"""YOLO 세그멘테이션 모델 검사와 다운로드를 담당합니다."""

from __future__ import annotations

import urllib.request

from PyQt6.QtWidgets import QMessageBox

from ..config import save_app_config
from ..constants import MODEL_OPTIONS
from ..models import AppConfig
from ..ui.dialogs import ModelDownloadDialog


def ensure_model_ready(config: AppConfig) -> bool:
    """선택된 모델이 준비되어 있는지 확인하고 필요 시 다운로드합니다."""
    if config.paths is None:
        return False

    model_path = config.paths.model_dir / config.selected_model
    if model_path.exists():
        return True

    dialog = ModelDownloadDialog(MODEL_OPTIONS, config.selected_model)
    if dialog.exec() == 0:
        return False

    selected_model = dialog.selected_weight_name()
    config.selected_model = selected_model
    save_app_config(config)
    selected_option = next(
        (option for option in MODEL_OPTIONS if option.weight_name == selected_model),
        None,
    )
    if selected_option is None:
        QMessageBox.critical(None, "모델 정보 오류", "선택한 모델 정보를 찾지 못했습니다.")
        return False

    confirmed = QMessageBox.question(
        None,
        "YOLO Seg 모델 다운로드",
        f"{selected_model} 모델이 없습니다.\nData/models에 다운로드하시겠습니까?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if confirmed != QMessageBox.StandardButton.Yes:
        return False

    try:
        target_path = config.paths.model_dir / selected_model
        # PyInstaller 패키징 안정성을 위해 ultralytics 런타임 의존 대신 공식 자산 URL에서 직접 다운로드합니다.
        urllib.request.urlretrieve(selected_option.download_url, str(target_path))
        return target_path.exists()
    except Exception as exc:
        QMessageBox.critical(
            None,
            "모델 다운로드 실패",
            f"모델을 준비하지 못했습니다.\n{exc}",
        )
        return False
