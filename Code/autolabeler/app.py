# -*- coding: utf-8 -*-
"""AutoSegmentation PyQt 애플리케이션을 초기화하고 실행합니다."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from .config import ensure_app_directories, load_app_config
from .services.model_service import ensure_model_ready
from .ui.main_window import MainWindow


def run() -> int:
    """애플리케이션을 실행하고 종료 코드를 반환합니다."""
    app = QApplication(sys.argv)
    app.setApplicationName("AutoSegmentation")

    # 실행 전에 필수 디렉토리와 기본 설정 파일을 준비합니다.
    ensure_app_directories()
    config = load_app_config()

    # 모델이 준비되지 않으면 사용자 선택에 따라 안전하게 종료합니다.
    if not ensure_model_ready(config):
        QMessageBox.information(
            None,
            "AutoSegmentation 종료",
            "필수 YOLO Seg 모델이 준비되지 않아 프로그램을 종료합니다.",
        )
        return 0

    window = MainWindow(config)
    window.show()
    return app.exec()
