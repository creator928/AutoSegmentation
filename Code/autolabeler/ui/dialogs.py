# -*- coding: utf-8 -*-
"""클래스 입력, 모델 선택, 설정 편집, 학습 결과 검증용 다이얼로그를 정의합니다."""

from __future__ import annotations

import json
import os
import locale
import subprocess
from typing import Iterable

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QKeySequenceEdit,
    QMessageBox,
    QPlainTextEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..constants import CLASS_COLORS
from ..models import AppConfig, ModelOption
from ..services.process_service import build_clean_python_env, clean_windows_dll_search_path, hidden_subprocess_kwargs


class ClassInputDialog(QDialog):
    """여러 줄 텍스트로 클래스를 입력받아 classes.txt를 생성합니다."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("classes.txt 생성")
        self.resize(420, 320)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("한 줄에 하나의 클래스 이름을 입력하세요."))

        self.editor = QPlainTextEdit(self)
        self.editor.setPlaceholderText("예시\nHuman\nDog\nBicycle")
        layout.addWidget(self.editor)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def class_names(self) -> list[str]:
        """입력창의 텍스트를 줄 단위 클래스 목록으로 반환합니다."""
        return [line.strip() for line in self.editor.toPlainText().splitlines() if line.strip()]

    def accept(self) -> None:
        """빈 입력을 막아 잘못된 classes.txt 생성을 방지합니다."""
        if not self.class_names():
            QMessageBox.warning(self, "입력 필요", "최소 1개 이상의 클래스를 입력해야 합니다.")
            return
        super().accept()


class ModelDownloadDialog(QDialog):
    """다운로드할 YOLO 모델을 선택하게 하는 다이얼로그입니다."""

    def __init__(self, options: Iterable[ModelOption], selected_weight: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("YOLO Seg 모델 선택")
        self.resize(420, 180)
        self._options = list(options)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("필수 YOLO Seg 모델이 없습니다. 다운로드할 모델을 선택하세요."))

        self.combo = QComboBox(self)
        for option in self._options:
            self.combo.addItem(f"{option.name} - {option.description}", option.weight_name)
        index = max(0, self.combo.findData(selected_weight))
        self.combo.setCurrentIndex(index)
        layout.addWidget(self.combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_weight_name(self) -> str:
        """현재 선택한 모델 파일명을 반환합니다."""
        return str(self.combo.currentData())


class SettingsDialog(QDialog):
    """테마, 세그멘테이션 입력 모드, 좌표 개수, 단축키를 편집하는 설정 창입니다."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.resize(520, 480)
        self._config = config

        root_layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.theme_combo = QComboBox(self)
        self.theme_combo.addItem("라이트", "light")
        self.theme_combo.addItem("다크", "dark")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(config.theme_mode)))
        form_layout.addRow("테마", self.theme_combo)

        self.input_combo = QComboBox(self)
        self.input_combo.addItem("윤곽선 클릭", "click")
        self.input_combo.addItem("브러시 드래그", "brush")
        self.input_combo.setCurrentIndex(max(0, self.input_combo.findData(config.rectangle_input_mode)))
        form_layout.addRow("세그 입력 방식", self.input_combo)

        self.polygon_point_spin = QSpinBox(self)
        self.polygon_point_spin.setRange(3, 256)
        self.polygon_point_spin.setValue(
            max(3, int(config.runtime_options.get("polygon_point_count", "24") or "24"))
        )
        form_layout.addRow("폴리곤 좌표 갯수", self.polygon_point_spin)

        # 브러시와 지우개가 공유하는 기본 붓 크기를 설정 창에서도 저장할 수 있게 합니다.
        self.brush_size_spin = QSpinBox(self)
        self.brush_size_spin.setRange(2, 96)
        self.brush_size_spin.setValue(
            max(2, int(config.runtime_options.get("brush_size", "18") or "18"))
        )
        form_layout.addRow("붓 크기", self.brush_size_spin)

        root_layout.addLayout(form_layout)
        root_layout.addWidget(QLabel("기능 단축키"))

        self.shortcut_list = QListWidget(self)
        self.shortcut_editors: dict[str, QKeySequenceEdit] = {}
        for action_name, key_value in config.shortcuts.items():
            item = QListWidgetItem(self.shortcut_list)
            row_widget = QWidget(self.shortcut_list)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.addWidget(QLabel(action_name))
            editor = QKeySequenceEdit(QKeySequence(key_value), row_widget)
            self.shortcut_editors[action_name] = editor
            row_layout.addWidget(editor)
            item.setSizeHint(row_widget.sizeHint())
            self.shortcut_list.addItem(item)
            self.shortcut_list.setItemWidget(item, row_widget)
        root_layout.addWidget(self.shortcut_list)

        root_layout.addWidget(QLabel("클래스 선택 단축키"))

        self.class_shortcut_list = QListWidget(self)
        self.class_editors: dict[str, QKeySequenceEdit] = {}
        for class_index, key_value in config.class_shortcuts.items():
            item = QListWidgetItem(self.class_shortcut_list)
            row_widget = QWidget(self.class_shortcut_list)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.addWidget(QLabel(f"클래스 {class_index}"))
            editor = QKeySequenceEdit(QKeySequence(key_value), row_widget)
            self.class_editors[class_index] = editor
            row_layout.addWidget(editor)
            item.setSizeHint(row_widget.sizeHint())
            self.class_shortcut_list.addItem(item)
            self.class_shortcut_list.setItemWidget(item, row_widget)
        root_layout.addWidget(self.class_shortcut_list)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

    def apply_to_config(self) -> None:
        """설정창의 값을 설정 객체에 반영합니다."""
        self._config.theme_mode = str(self.theme_combo.currentData())
        self._config.rectangle_input_mode = str(self.input_combo.currentData())
        self._config.runtime_options["polygon_point_count"] = str(self.polygon_point_spin.value())
        self._config.runtime_options["brush_size"] = str(self.brush_size_spin.value())

        for action_name, editor in self.shortcut_editors.items():
            sequence = editor.keySequence().toString()
            if sequence:
                self._config.shortcuts[action_name] = sequence

        for class_index, editor in self.class_editors.items():
            sequence = editor.keySequence().toString()
            if sequence:
                self._config.class_shortcuts[class_index] = sequence


class ResultValidationDialog(QDialog):
    """Temp 데이터셋 이미지에 대해 ONNX 추론 결과를 슬라이더로 검증하는 창입니다."""

    def __init__(
        self,
        python_command: list[str],
        runner_script_path: str,
        model_path: str,
        image_paths: list[str],
        class_names: list[str],
        conf_threshold: float,
        use_gpu: bool,
        ultralytics_dir: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("학습 결과 검증")
        self.resize(1100, 820)
        self._image_paths = image_paths
        self._class_names = class_names
        self._conf_threshold = conf_threshold
        self._device_text = "0" if use_gpu else "cpu"
        self._python_command = python_command
        self._runner_script_path = runner_script_path
        self._model_path = model_path
        self._ultralytics_dir = ultralytics_dir
        self._result_cache: dict[int, dict[str, object]] = {}

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
        """창 크기가 바뀌면 현재 이미지도 새 크기에 맞춰 다시 그립니다."""
        super().resizeEvent(event)
        if self._image_paths:
            self.render_index(self.slider.value())

    def _class_display_name(self, class_index: int) -> str:
        """클래스 인덱스를 사람이 읽기 쉬운 이름으로 변환합니다."""
        if 0 <= class_index < len(self._class_names):
            return self._class_names[class_index]
        return f"class_{class_index}"

    def _result_for_index(self, index: int):
        """이미지별 추론 결과를 캐시해 슬라이더 이동 시 재추론을 줄입니다."""
        if index not in self._result_cache:
            try:
                process_env = build_clean_python_env()
                with clean_windows_dll_search_path():
                    completed = subprocess.run(
                        [
                            *self._python_command,
                            self._runner_script_path,
                            "--model",
                            self._model_path,
                            "--image",
                            self._image_paths[index],
                            "--imgsz",
                            "640",
                            "--conf",
                            str(self._conf_threshold),
                            "--device",
                            self._device_text,
                            "--ultralytics-dir",
                            self._ultralytics_dir,
                        ],
                        capture_output=True,
                        text=False,
                        env=process_env,
                        check=False,
                        **hidden_subprocess_kwargs(),
                    )
                stdout_text = ""
                for encoding in ("utf-8", locale.getpreferredencoding(False), "cp949"):
                    try:
                        stdout_text = completed.stdout.decode(encoding).strip()
                        break
                    except UnicodeDecodeError:
                        continue
                if not stdout_text:
                    stdout_text = completed.stdout.decode("utf-8", errors="replace").strip()
                if completed.returncode != 0:
                    raise RuntimeError(stdout_text or f"검증 프로세스가 비정상 종료되었습니다. code={completed.returncode}")

                json_line = next(
                    (
                        line.strip()
                        for line in reversed(stdout_text.splitlines())
                        if line.strip().startswith("{") and line.strip().endswith("}")
                    ),
                    "",
                )
                if not json_line:
                    raise RuntimeError(stdout_text or "검증 결과 JSON을 찾지 못했습니다.")

                self._result_cache[index] = json.loads(json_line)
            except Exception as exc:
                self._result_cache[index] = {"detections": [], "error": str(exc)}
        return self._result_cache[index]

    def render_index(self, index: int) -> None:
        """현재 슬라이더 위치의 이미지를 세그 폴리곤과 텍스트를 포함해 렌더링합니다."""
        if index < 0 or index >= len(self._image_paths):
            return

        image_path = self._image_paths[index]
        qimage = QImage(image_path)
        if qimage.isNull():
            self.image_label.setText("이미지를 불러오지 못했습니다.")
            return

        painted = qimage.copy()
        painter = QPainter(painted)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Malgun Gothic", 8))

        results = self._result_for_index(index)
        polygon_count = 0
        for detection in results.get("detections", []):
            class_index = int(detection["class_index"])
            conf_value = float(detection["conf"])
            x1, y1, x2, y2 = detection["xyxy"]
            color = QColor(CLASS_COLORS[class_index % len(CLASS_COLORS)])
            painter.setPen(QPen(color, 2))
            polygon = detection.get("polygon", [])
            if polygon:
                qpolygon = QPolygonF()
                for x_value, y_value in polygon:
                    qpolygon.append(QPointF(float(x_value) * qimage.width(), float(y_value) * qimage.height()))
                if qpolygon.count() >= 3:
                    painter.setBrush(QColor(color.red(), color.green(), color.blue(), 60))
                    painter.drawPolygon(qpolygon)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                else:
                    painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            else:
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            painter.fillRect(
                int(x1),
                max(0, int(y1) - 18),
                max(110, len(self._class_display_name(class_index)) * 8),
                18,
                color,
            )
            painter.setPen(QColor("#000000"))
            painter.drawText(
                int(x1) + 4,
                max(12, int(y1) - 5),
                f"{self._class_display_name(class_index)} {conf_value:.2f}",
            )
            polygon_count += 1

        painter.end()
        error_text = str(results.get("error", "")).strip()
        status_text = (
            f"{index + 1} / {len(self._image_paths)}  |  {qimage.size().width()}x{qimage.size().height()}  |  polygons={polygon_count}"
        )
        if error_text:
            status_text = f"{status_text}  |  오류: {error_text}"
        self.index_label.setText(status_text)
        pixmap = QPixmap.fromImage(painted).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
