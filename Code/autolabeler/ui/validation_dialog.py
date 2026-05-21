# -*- coding: utf-8 -*-
"""Temp/Verify 아래 저장된 검증 이미지를 표시하는 전용 대화상자입니다."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog, QLabel, QSlider, QVBoxLayout, QWidget


class VerifyImageDialog(QDialog):
    """학습 시 미리 생성한 검증 이미지를 슬라이더로 확인하는 창입니다."""

    def __init__(self, image_paths: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("학습 결과 검증")
        self.resize(1100, 820)
        self._image_paths = image_paths

        layout = QVBoxLayout(self)
        self.index_label = QLabel(self)
        layout.addWidget(self.index_label)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(700)
        layout.addWidget(self.image_label, stretch=1)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self._image_paths) - 1))
        self.slider.valueChanged.connect(self.render_index)
        layout.addWidget(self.slider)

        if self._image_paths:
            self.render_index(0)

    def resizeEvent(self, event) -> None:
        """창 크기 변경 시 현재 이미지를 다시 맞춰 표시합니다."""
        super().resizeEvent(event)
        if self._image_paths:
            self.render_index(self.slider.value())

    def render_index(self, index: int) -> None:
        """슬라이더 위치의 검증 이미지를 화면에 표시합니다."""
        if index < 0 or index >= len(self._image_paths):
            return

        image_path = self._image_paths[index]
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("검증 이미지를 불러오지 못했습니다.")
            return

        self.index_label.setText(f"{index + 1} / {len(self._image_paths)}  |  {pixmap.width()}x{pixmap.height()}")
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
