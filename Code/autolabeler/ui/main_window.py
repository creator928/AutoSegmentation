# -*- coding: utf-8 -*-
"""AutoSegmentation의 메인 윈도우와 주요 사용자 흐름을 구현합니다."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QPointF, QThread, Qt
from PyQt6.QtGui import QColor, QImage, QKeySequence, QPainter, QPainterPath, QPolygonF, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..config import save_app_config
from ..constants import CLASS_COLORS
from ..models import AppConfig, LabelPolygon
from ..services.auto_label_worker import AutoLabelRequest, AutoLabelWorker
from ..services.hardware_service import detect_hardware
from ..services.label_service import load_labels, save_labels
from ..services.training_service import (
    is_training_ready,
    result_onnx_path,
    result_pt_path,
    verify_image_paths,
)
from ..services.training_worker import TrainingRequest, TrainingWorker
from ..services.worklog_service import (
    WorkStatus,
    load_worklog_statuses,
    set_work_status,
    work_status_for_image,
    worklog_path,
)
from ..services.workspace_service import ensure_classes_file, list_images, read_classes
from .canvas import ImageCanvas
from .dialogs import ClassInputDialog, ResultValidationDialog, SettingsDialog
from .styles import build_stylesheet


class MainWindow(QMainWindow):
    """좌측 제어 패널과 우측 작업 영역을 포함한 메인 화면입니다."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.current_work_dir: Path | None = None
        self.current_image_index = -1
        self.current_image_paths: list[Path] = []
        self.class_names: list[str] = []
        self.current_labels: list[LabelPolygon] = []
        self.work_statuses: dict[Path, WorkStatus] = {}
        self.selected_class_index = 0
        self.shortcuts: list[QShortcut] = []
        self.training_option_inputs: dict[str, QLineEdit] = {}
        self.hardware_status = detect_hardware()
        self.training_thread: QThread | None = None
        self.training_worker: TrainingWorker | None = None
        self.auto_label_thread: QThread | None = None
        self.auto_label_worker: AutoLabelWorker | None = None
        self.validation_dialog: ResultValidationDialog | None = None

        self.setWindowTitle("AutoSegmentation")
        self.resize(1600, 900)
        self._build_ui()
        self._apply_theme()
        self._install_shortcuts()
        self.showMaximized()
    
    def _build_ui(self) -> None:
        """메인 레이아웃과 각 컨트롤을 생성합니다."""
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(2, 2, 2, 2)
        root_layout.setSpacing(2)
        splitter = QSplitter(central)
        root_layout.addWidget(splitter)

        left_panel = QFrame(splitter)
        # 좌측 패널이 과도하게 넓어지지 않도록 초기 폭과 최대 폭을 제한합니다.
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(2)

        title = QLabel("AutoSegmentation", left_panel)
        title.setStyleSheet("font-size: 16pt; font-weight: 700;")
        left_layout.addWidget(title)

        self.mode_label = QLabel("모드: 대기", left_panel)
        left_layout.addWidget(self.mode_label)

        self.class_label = QLabel("선택 클래스: 0", left_panel)
        left_layout.addWidget(self.class_label)

        self.folder_button = QPushButton("작업 폴더 선택", left_panel)
        self.folder_button.clicked.connect(self.select_work_folder)
        left_layout.addWidget(self.folder_button)

        self.settings_button = QPushButton("설정", left_panel)
        self.settings_button.clicked.connect(self.open_settings)
        left_layout.addWidget(self.settings_button)

        self.theme_button = QPushButton("라이트/다크 전환", left_panel)
        self.theme_button.clicked.connect(self.toggle_theme)
        left_layout.addWidget(self.theme_button)

        left_layout.addWidget(QLabel("단축키 안내", left_panel))
        self.shortcut_guide = QListWidget(left_panel)
        self.shortcut_guide.setMaximumHeight(160)
        left_layout.addWidget(self.shortcut_guide)
        self._refresh_shortcut_guide()

        model_panel = QFrame(left_panel)
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(2, 2, 2, 2)
        model_layout.setSpacing(2)
        model_layout.addWidget(QLabel("현재 YOLO Seg 모델", model_panel))
        # 현재 선택된 모델명을 바로 확인할 수 있도록 좌측 패널에 별도 표시합니다.
        self.model_name_label = QLabel(self.config.selected_model, model_panel)
        self.model_name_label.setWordWrap(True)
        model_layout.addWidget(self.model_name_label)
        left_layout.addWidget(model_panel)

        autolabel_panel = QFrame(left_panel)
        autolabel_layout = QVBoxLayout(autolabel_panel)
        autolabel_layout.setContentsMargins(2, 2, 2, 2)
        autolabel_layout.setSpacing(2)
        autolabel_layout.addWidget(QLabel("오토 세그 / 학습 옵션", autolabel_panel))

        self.training_edit_button = QPushButton("설정 수정", autolabel_panel)
        self.training_edit_button.setCheckable(True)
        self.training_edit_button.toggled.connect(self.toggle_training_option_edit_mode)
        autolabel_layout.addWidget(self.training_edit_button)

        # 아직 실제 기능은 연결하지 않고, 후속 자동 라벨/학습 기능을 위한 입력 UI만 먼저 구성합니다.
        training_form = QFormLayout()
        training_form.setContentsMargins(2, 2, 2, 2)
        training_form.setHorizontalSpacing(2)
        training_form.setVerticalSpacing(2)
        self.hardware_name_label = QLabel(self._hardware_display_text(), autolabel_panel)
        # 긴 장치명은 줄바꿈으로 표시해 좌측 패널 폭이 불필요하게 커지지 않도록 합니다.
        self.hardware_name_label.setWordWrap(True)
        training_form.addRow("사용 하드웨어", self.hardware_name_label)
        self.use_gpu_checkbox = QCheckBox("GPU 사용", autolabel_panel)
        # GPU가 있으면 기본값을 체크 상태로 두고, 없으면 체크 해제 및 비활성화합니다.
        saved_use_gpu = self.config.runtime_options.get("use_gpu", "true").lower() == "true"
        self.use_gpu_checkbox.setChecked(self.hardware_status.gpu_available and saved_use_gpu)
        self.use_gpu_checkbox.setEnabled(self.hardware_status.gpu_available)
        self.use_gpu_checkbox.toggled.connect(self.on_gpu_checkbox_toggled)
        training_form.addRow("장치 선택", self.use_gpu_checkbox)
        self.training_option_inputs["dataset_size"] = self._create_training_option_input(
            self.config.runtime_options.get("dataset_size", "100")
        )
        self.training_option_inputs["epochs"] = self._create_training_option_input(
            self.config.runtime_options.get("epochs", "50")
        )
        self.training_option_inputs["image_size"] = self._create_training_option_input(
            self.config.runtime_options.get("image_size", "640")
        )
        self.training_option_inputs["batch_size"] = self._create_training_option_input(
            self.config.runtime_options.get("batch_size", "8")
        )
        self.training_option_inputs["project_name"] = self._create_training_option_input(
            self.config.runtime_options.get("project_name", "AutoSegmentationTrain")
        )
        training_form.addRow("학습 데이터량", self.training_option_inputs["dataset_size"])
        training_form.addRow("Epochs", self.training_option_inputs["epochs"])
        training_form.addRow("이미지 크기", self.training_option_inputs["image_size"])
        training_form.addRow("배치 크기", self.training_option_inputs["batch_size"])
        training_form.addRow("프로젝트명", self.training_option_inputs["project_name"])
        autolabel_layout.addLayout(training_form)
        self.update_hardware_label()

        input_mode_panel = QFrame(autolabel_panel)
        input_mode_layout = QVBoxLayout(input_mode_panel)
        input_mode_layout.setContentsMargins(0, 0, 0, 0)
        input_mode_layout.setSpacing(2)
        input_mode_layout.addWidget(QLabel("입력 모드", input_mode_panel))
        # 현재 선택된 입력 방식을 라디오 버튼으로 명확히 표시하고 바로 전환할 수 있게 합니다.
        self.input_mode_group = QButtonGroup(input_mode_panel)
        self.click_mode_radio = QRadioButton("윤곽선 클릭", input_mode_panel)
        self.brush_mode_radio = QRadioButton("브러시 드래그", input_mode_panel)
        self.input_mode_group.addButton(self.click_mode_radio)
        self.input_mode_group.addButton(self.brush_mode_radio)
        self.click_mode_radio.setChecked(self.config.rectangle_input_mode != "brush")
        self.brush_mode_radio.setChecked(self.config.rectangle_input_mode == "brush")
        self.click_mode_radio.toggled.connect(self.on_input_mode_radio_toggled)
        self.brush_mode_radio.toggled.connect(self.on_input_mode_radio_toggled)
        input_mode_layout.addWidget(self.click_mode_radio)
        input_mode_layout.addWidget(self.brush_mode_radio)
        autolabel_layout.addWidget(input_mode_panel)

        brush_row = QHBoxLayout()
        brush_row.setContentsMargins(0, 0, 0, 0)
        brush_row.setSpacing(2)
        brush_row.addWidget(QLabel("붓 크기", autolabel_panel))
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal, autolabel_panel)
        self.brush_size_slider.setRange(2, 96)
        initial_brush_size = max(2, int(self.config.runtime_options.get("brush_size", "18") or "18"))
        self.brush_size_slider.setValue(initial_brush_size)
        brush_row.addWidget(self.brush_size_slider, stretch=1)
        self.brush_size_spinbox = QSpinBox(autolabel_panel)
        self.brush_size_spinbox.setRange(2, 96)
        self.brush_size_spinbox.setValue(initial_brush_size)
        brush_row.addWidget(self.brush_size_spinbox)
        autolabel_layout.addLayout(brush_row)

        polygon_row = QHBoxLayout()
        polygon_row.setContentsMargins(0, 0, 0, 0)
        polygon_row.setSpacing(2)
        polygon_row.addWidget(QLabel("폴리곤 점 수", autolabel_panel))
        self.polygon_point_slider = QSlider(Qt.Orientation.Horizontal, autolabel_panel)
        self.polygon_point_slider.setRange(3, 256)
        initial_polygon_point_count = max(
            3,
            int(self.config.runtime_options.get("polygon_point_count", "24") or "24"),
        )
        self.polygon_point_slider.setValue(initial_polygon_point_count)
        polygon_row.addWidget(self.polygon_point_slider, stretch=1)
        self.polygon_point_spinbox = QSpinBox(autolabel_panel)
        self.polygon_point_spinbox.setRange(3, 256)
        self.polygon_point_spinbox.setValue(initial_polygon_point_count)
        polygon_row.addWidget(self.polygon_point_spinbox)
        autolabel_layout.addLayout(polygon_row)

        self.background_training_button = QPushButton("백그라운드 학습 진행", autolabel_panel)
        self.background_training_button.clicked.connect(self.start_background_training)
        self.background_training_button.setEnabled(False)
        autolabel_layout.addWidget(self.background_training_button)
        self.validation_button = QPushButton("학습 결과 검증", autolabel_panel)
        self.validation_button.clicked.connect(self.open_result_validation_dialog)
        self.validation_button.setEnabled(False)
        autolabel_layout.addWidget(self.validation_button)
        self.auto_label_button = QPushButton("Auto Segmentation", autolabel_panel)
        self.auto_label_button.clicked.connect(self.start_auto_labeling)
        self.auto_label_button.setEnabled(False)
        autolabel_layout.addWidget(self.auto_label_button)
        self.training_option_inputs["auto_label_conf"] = self._create_training_option_input(
            self.config.runtime_options.get("auto_label_conf", "0.01")
        )
        autolabel_conf_form = QFormLayout()
        autolabel_conf_form.setContentsMargins(2, 2, 2, 2)
        autolabel_conf_form.setHorizontalSpacing(2)
        autolabel_conf_form.setVerticalSpacing(2)
        autolabel_conf_form.addRow("Auto Seg Conf", self.training_option_inputs["auto_label_conf"])
        autolabel_layout.addLayout(autolabel_conf_form)
        left_layout.addWidget(autolabel_panel)
        left_layout.addStretch(1)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 1350])

        work_panel = QFrame(splitter)
        work_layout = QHBoxLayout(work_panel)
        work_layout.setContentsMargins(2, 2, 2, 2)
        work_layout.setSpacing(2)

        work_splitter = QSplitter(Qt.Orientation.Horizontal, work_panel)
        work_layout.addWidget(work_splitter)

        center_panel = QFrame(work_splitter)
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(2, 2, 2, 2)
        center_layout.setSpacing(2)

        self.canvas = ImageCanvas(center_panel)
        self.canvas.polygon_created.connect(self.add_polygon_from_canvas)
        self.canvas.mask_requested.connect(self.request_mask)
        self.canvas.label_edited.connect(self.apply_edited_label)
        self.canvas.label_deleted.connect(self.apply_deleted_label)
        # 그리기/마스킹은 자동 종료하되, 편집은 연속 작업이 가능하도록 별도 처리기로 연결합니다.
        self.canvas.interaction_finished.connect(self.handle_canvas_interaction_finished)
        self.canvas.set_input_mode(self.config.rectangle_input_mode)
        self.canvas.set_polygon_point_count(initial_polygon_point_count)
        self.canvas.set_brush_size(initial_brush_size)
        self.brush_size_slider.valueChanged.connect(self.sync_brush_size_from_slider)
        self.brush_size_spinbox.valueChanged.connect(self.sync_brush_size_from_spinbox)
        self.polygon_point_slider.valueChanged.connect(self.sync_polygon_point_count_from_slider)
        self.polygon_point_spinbox.valueChanged.connect(self.sync_polygon_point_count_from_spinbox)
        center_layout.addWidget(self.canvas, stretch=1)

        self.training_progress_panel = QFrame(center_panel)
        progress_layout = QHBoxLayout(self.training_progress_panel)
        progress_layout.setContentsMargins(2, 2, 2, 2)
        progress_layout.setSpacing(2)
        self.training_progress_label = QLabel("0/0 (0%)", self.training_progress_panel)
        progress_layout.addWidget(self.training_progress_label)
        self.training_progress_bar = QProgressBar(self.training_progress_panel)
        self.training_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.training_progress_bar, stretch=1)
        self.training_progress_panel.hide()
        center_layout.addWidget(self.training_progress_panel)

        right_sidebar = QFrame(work_splitter)
        sidebar_layout = QVBoxLayout(right_sidebar)
        sidebar_layout.setContentsMargins(2, 2, 2, 2)
        sidebar_layout.setSpacing(2)

        status_title = QLabel("작업 상태", right_sidebar)
        sidebar_layout.addWidget(status_title)

        status_card = QFrame(right_sidebar)
        status_card_layout = QHBoxLayout(status_card)
        status_card_layout.setContentsMargins(2, 2, 2, 2)
        status_card_layout.setSpacing(2)
        # 현재 이미지의 worklog 상태를 신호등처럼 표시하는 원형 인디케이터입니다.
        self.status_light = QLabel("●", status_card)
        self.status_light.setStyleSheet("font-size: 28pt; color: #ff4d4f; border: none;")
        status_card_layout.addWidget(self.status_light)
        self.status_text = QLabel("미검토", status_card)
        status_card_layout.addWidget(self.status_text)
        sidebar_layout.addWidget(status_card)

        sidebar_layout.addWidget(QLabel("클래스 정보", right_sidebar))
        self.class_list = QListWidget(right_sidebar)
        sidebar_layout.addWidget(self.class_list, stretch=3)

        sidebar_layout.addWidget(QLabel("이미지 목록", right_sidebar))

        image_nav_panel = QFrame(right_sidebar)
        image_nav_layout = QVBoxLayout(image_nav_panel)
        image_nav_layout.setContentsMargins(2, 2, 2, 2)
        image_nav_layout.setSpacing(2)

        self.image_position_label = QLabel("0 / 0", image_nav_panel)
        image_nav_layout.addWidget(self.image_position_label)

        self.image_index_slider = QSlider(Qt.Orientation.Horizontal, image_nav_panel)
        self.image_index_slider.setMinimum(1)
        self.image_index_slider.setMaximum(1)
        image_nav_layout.addWidget(self.image_index_slider)

        image_go_row = QHBoxLayout()
        image_go_row.setContentsMargins(0, 0, 0, 0)
        image_go_row.setSpacing(2)
        self.image_index_spinbox = QSpinBox(image_nav_panel)
        self.image_index_spinbox.setMinimum(1)
        self.image_index_spinbox.setMaximum(1)
        image_go_row.addWidget(self.image_index_spinbox)
        self.image_go_button = QPushButton("Go", image_nav_panel)
        self.image_go_button.clicked.connect(self.go_to_selected_image_index)
        image_go_row.addWidget(self.image_go_button)
        image_nav_layout.addLayout(image_go_row)

        self.image_index_slider.valueChanged.connect(self.sync_image_spinbox_from_slider)
        self.image_index_spinbox.valueChanged.connect(self.sync_image_slider_from_spinbox)
        sidebar_layout.addWidget(image_nav_panel)

        self.image_list = QListWidget(right_sidebar)
        self.image_list.currentRowChanged.connect(self.load_image_by_index)
        # 사용자가 이미지 이름을 더블 클릭했을 때 해당 이미지를 작업 대상으로 불러옵니다.
        self.image_list.itemDoubleClicked.connect(self.load_image_from_item)
        sidebar_layout.addWidget(self.image_list, stretch=7)

        work_splitter.addWidget(center_panel)
        work_splitter.addWidget(right_sidebar)
        work_splitter.setSizes([1180, 260])

        splitter.addWidget(left_panel)
        splitter.addWidget(work_panel)
        splitter.setSizes([180, 1420])

    def _apply_theme(self) -> None:
        """현재 설정의 테마를 앱 전체 스타일에 반영합니다."""
        self.setStyleSheet(build_stylesheet(self.config.theme_mode))

    def _refresh_shortcut_guide(self) -> None:
        """좌측 안내 패널의 단축키 설명을 다시 그립니다."""
        self.shortcut_guide.clear()
        entries = [
            ("세그 그리기", self.config.shortcuts["draw_box"]),
            ("마스킹", self.config.shortcuts["mask_area"]),
            ("편집", self.config.shortcuts["edit_box"]),
            ("작업 취소", self.config.shortcuts["cancel_mode"]),
            ("이전 이미지", self.config.shortcuts["prev_image"]),
            ("다음 이미지", self.config.shortcuts["next_image"]),
            ("화면 맞춤", self.config.shortcuts["reset_view"]),
            ("마지막 폴리곤 삭제", self.config.shortcuts["delete_last_box"]),
            ("테마 전환", self.config.shortcuts["toggle_theme"]),
            ("설정 열기", self.config.shortcuts["open_settings"]),
        ]
        for title, key_text in entries:
            self.shortcut_guide.addItem(f"{title} : {key_text}")
        # 요청에 맞춰 좌측 단축키 표시에는 `, 1, 2, 3, 4만 간단히 노출합니다.
        for class_index in ("0", "1", "2", "3", "4"):
            key_text = self.config.class_shortcuts.get(class_index)
            if key_text:
                label = "`" if class_index == "0" else class_index
                self.shortcut_guide.addItem(f"클래스 {label} : {key_text}")
        if self.is_plain_key_unassigned("C"):
            self.shortcut_guide.addItem("그리기 모드 변경 : C")
        if self.is_plain_key_unassigned("Z"):
            self.shortcut_guide.addItem("빈 이미지 검토 : Z")

    def _create_training_option_input(self, text: str) -> QLineEdit:
        """학습 옵션 입력칸을 만들고 기본값은 읽기 전용 표시 상태로 둡니다."""
        line_edit = QLineEdit(text, self)
        line_edit.setReadOnly(True)
        line_edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        line_edit.textChanged.connect(self.update_training_availability)
        line_edit.textChanged.connect(lambda _text: self.save_runtime_option_values())
        return line_edit

    def _hardware_display_text(self) -> str:
        """현재 선택된 학습 장치 표시 문자열을 반환합니다."""
        if self.hardware_status.gpu_available:
            return self.hardware_status.gpu_name
        return self.hardware_status.cpu_name

    def update_hardware_label(self) -> None:
        """체크박스 상태에 따라 현재 사용할 학습 장치를 표시합니다."""
        if self.use_gpu_checkbox.isChecked() and self.hardware_status.gpu_available:
            self.hardware_name_label.setText(self.hardware_status.gpu_name)
        else:
            self.hardware_name_label.setText(self.hardware_status.cpu_name)

    def save_runtime_option_values(self) -> None:
        """좌측 학습/오토 라벨 옵션과 GPU 체크 상태를 설정 파일에 저장합니다."""
        self.config.runtime_options["use_gpu"] = "true" if self.use_gpu_checkbox.isChecked() else "false"
        self.config.runtime_options["brush_size"] = str(self.brush_size_spinbox.value())
        self.config.runtime_options["polygon_point_count"] = str(self.polygon_point_spinbox.value())
        for option_name, line_edit in self.training_option_inputs.items():
            self.config.runtime_options[option_name] = line_edit.text().strip()
        save_app_config(self.config)

    def on_gpu_checkbox_toggled(self) -> None:
        """GPU 체크박스 변경 시 표시와 설정 파일을 함께 갱신합니다."""
        self.update_hardware_label()
        self.save_runtime_option_values()

    def should_use_gpu_for_runtime(self) -> bool:
        """현재 체크 상태와 학습 Python의 CUDA 지원 여부를 함께 반영한 실제 GPU 사용 여부를 반환합니다."""
        return (
            self.use_gpu_checkbox.isChecked()
            and self.hardware_status.gpu_available
            and self.hardware_status.cuda_runtime_available
        )

    def current_dataset_size(self) -> int:
        """학습 데이터량 입력값을 정수로 해석합니다."""
        try:
            return max(0, int(self.training_option_inputs["dataset_size"].text() or "0"))
        except ValueError:
            return 0

    def current_epochs(self) -> int:
        """Epochs 입력값을 반환합니다."""
        try:
            return max(1, int(self.training_option_inputs["epochs"].text() or "1"))
        except ValueError:
            return 1

    def current_image_size(self) -> int:
        """이미지 크기 입력값을 반환합니다."""
        try:
            return max(32, int(self.training_option_inputs["image_size"].text() or "640"))
        except ValueError:
            return 640

    def current_batch_size(self) -> int:
        """배치 크기 입력값을 반환합니다."""
        try:
            return max(1, int(self.training_option_inputs["batch_size"].text() or "1"))
        except ValueError:
            return 1

    def current_project_name(self) -> str:
        """프로젝트명 입력값을 반환합니다."""
        text = self.training_option_inputs["project_name"].text().strip()
        return text or "AutoSegmentationTrain"

    def current_auto_label_conf(self) -> float:
        """오토 세그 추론 confidence 입력값을 반환합니다."""
        try:
            value = float(self.training_option_inputs["auto_label_conf"].text() or "0.01")
        except ValueError:
            value = 0.01
        return min(1.0, max(0.0, value))

    def update_training_availability(self) -> None:
        """현재 작업 상태와 입력값 기준으로 학습 버튼 활성화 여부를 결정합니다."""
        ready = (
            self.current_work_dir is not None
            and bool(self.class_names)
            and is_training_ready(self.current_work_dir, self.current_image_paths, self.current_dataset_size())
            and self.config.paths is not None
            and (self.config.paths.model_dir / self.config.selected_model).exists()
            and self.training_thread is None
            and self.auto_label_thread is None
        )
        self.background_training_button.setEnabled(bool(ready))
        validation_ready = (
            self.current_work_dir is not None
            and result_pt_path(self.current_work_dir).exists()
            and bool(verify_image_paths(self.current_work_dir))
            and self.training_thread is None
            and self.auto_label_thread is None
        )
        self.validation_button.setEnabled(bool(validation_ready))
        auto_label_ready = (
            self.current_work_dir is not None
            and self.current_image_index >= 0
            and self.config.paths is not None
            and result_pt_path(self.current_work_dir).exists()
            and self.training_thread is None
            and self.auto_label_thread is None
        )
        self.auto_label_button.setEnabled(bool(auto_label_ready))

    def toggle_training_option_edit_mode(self, checked: bool) -> None:
        """설정 수정 버튼 상태에 따라 학습 옵션 입력칸의 편집 가능 여부를 바꿉니다."""
        self.training_edit_button.setText("설정 잠금" if checked else "설정 수정")
        for line_edit in self.training_option_inputs.values():
            line_edit.setReadOnly(not checked)
            line_edit.setFocusPolicy(Qt.FocusPolicy.StrongFocus if checked else Qt.FocusPolicy.NoFocus)

    def _install_shortcuts(self) -> None:
        """설정값 기준으로 전역 단축키를 다시 바인딩합니다."""
        for shortcut in self.shortcuts:
            shortcut.setParent(None)
        self.shortcuts.clear()

        self._bind_shortcut(self.config.shortcuts["draw_box"], lambda: self.set_canvas_mode("draw"))
        self._bind_shortcut(self.config.shortcuts["mask_area"], lambda: self.set_canvas_mode("mask"))
        self._bind_shortcut(self.config.shortcuts["edit_box"], self.toggle_edit_mode)
        self._bind_shortcut(self.config.shortcuts["cancel_mode"], self.cancel_active_mode)
        self._bind_shortcut(self.config.shortcuts["prev_image"], self.go_prev_image)
        self._bind_shortcut(self.config.shortcuts["next_image"], self.go_next_image)
        self._bind_shortcut(self.config.shortcuts["reset_view"], self.reset_canvas_view)
        self._bind_shortcut(self.config.shortcuts["delete_last_box"], self.delete_last_box)
        self._bind_shortcut(self.config.shortcuts["toggle_theme"], self.toggle_theme)
        self._bind_shortcut(self.config.shortcuts["open_settings"], self.open_settings)

        for class_index, key_text in self.config.class_shortcuts.items():
            self._bind_shortcut(key_text, lambda idx=int(class_index): self.select_class(idx))
        if self.is_plain_key_unassigned("C"):
            self._bind_shortcut("C", self.toggle_input_mode_shortcut)
        if self.is_plain_key_unassigned("Z"):
            self._bind_shortcut("Z", self.confirm_no_object_review)

    def _bind_shortcut(self, key_text: str, callback) -> None:
        """단축키 문자열을 QShortcut으로 바인딩합니다."""
        shortcut = QShortcut(QKeySequence(key_text), self)
        shortcut.activated.connect(callback)
        self.shortcuts.append(shortcut)

    def is_plain_key_unassigned(self, key_text: str) -> bool:
        """지정한 단일 키가 현재 단축키 설정에 비어 있는지 확인합니다."""
        target = key_text.strip().upper()
        for configured_key in self.config.shortcuts.values():
            if configured_key.strip().upper() == target:
                return False
        for configured_key in self.config.class_shortcuts.values():
            if configured_key.strip().upper() == target:
                return False
        return True

    def select_work_folder(self) -> None:
        """사용자가 작업 폴더를 선택하면 클래스와 이미지 목록을 준비합니다."""
        if self.config.paths is None:
            return

        selected = QFileDialog.getExistingDirectory(self, "작업 폴더 선택", str(self.config.paths.work_dir))
        if not selected:
            return

        work_dir = Path(selected)
        classes = read_classes(work_dir)
        if not classes:
            dialog = ClassInputDialog(self)
            if dialog.exec() == 0:
                return
            classes = ensure_classes_file(work_dir, dialog.class_names())

        images = list_images(work_dir)
        if not images:
            QMessageBox.warning(self, "이미지 없음", "선택한 폴더에 이미지 파일이 없습니다.")
            return

        self.current_work_dir = work_dir
        self.class_names = classes
        self.current_image_paths = images
        self.work_statuses = load_worklog_statuses(work_dir, images)
        self.current_image_index = 0
        self._refresh_class_list()
        self._refresh_image_list()
        self.load_image_by_index(0)
        self.update_training_availability()

    def _refresh_class_list(self) -> None:
        """클래스 목록 UI를 최신 상태로 갱신합니다."""
        self.class_list.clear()
        for index, name in enumerate(self.class_names):
            item = QListWidgetItem(f"{index}:{name}")
            item.setForeground(QColor(CLASS_COLORS[index % len(CLASS_COLORS)]))
            self.class_list.addItem(item)
        self.select_class(min(self.selected_class_index, max(0, len(self.class_names) - 1)))

    def _refresh_image_list(self) -> None:
        """이미지 목록 UI를 작업 폴더 기준으로 다시 채웁니다."""
        self.image_list.clear()
        for image_path in self.current_image_paths:
            item = QListWidgetItem(image_path.name)
            self._apply_image_item_status_color(item, image_path)
            self.image_list.addItem(item)
        if 0 <= self.current_image_index < self.image_list.count():
            self.image_list.setCurrentRow(self.current_image_index)
        self._refresh_image_navigation_controls()

    def _refresh_image_navigation_controls(self) -> None:
        """이미지 개수와 현재 위치를 슬라이더/스핀박스/텍스트에 반영합니다."""
        total_count = len(self.current_image_paths)
        current_value = max(1, self.current_image_index + 1) if total_count > 0 else 0

        self.image_index_slider.blockSignals(True)
        self.image_index_spinbox.blockSignals(True)

        self.image_index_slider.setMinimum(1 if total_count > 0 else 0)
        self.image_index_slider.setMaximum(max(1, total_count))
        self.image_index_spinbox.setMinimum(1 if total_count > 0 else 0)
        self.image_index_spinbox.setMaximum(max(1, total_count))

        if total_count > 0:
            self.image_index_slider.setValue(current_value)
            self.image_index_spinbox.setValue(current_value)
            self.image_position_label.setText(f"{current_value} / {total_count}")
        else:
            self.image_position_label.setText("0 / 0")

        self.image_index_slider.blockSignals(False)
        self.image_index_spinbox.blockSignals(False)
        self.persist_auto_label_stop_index()

    def current_auto_label_stop_index(self) -> int:
        """오토 세그가 멈춰야 하는 현재 사용자 작업 인덱스를 반환합니다."""
        if not self.current_image_paths:
            return -1
        return max(0, self.image_index_spinbox.value() - 1)

    def persist_auto_label_stop_index(self) -> None:
        """현재 사용자 작업 경계를 파일로 기록해 백그라운드 오토 세그가 참조하게 합니다."""
        if self.current_work_dir is None:
            return
        stop_index_path = self.current_work_dir / "Temp" / "Result" / "auto_label_stop_index.txt"
        stop_index_path.parent.mkdir(parents=True, exist_ok=True)
        stop_index_path.write_text(str(self.current_auto_label_stop_index()), encoding="utf-8")

    def reload_worklog_statuses(self) -> None:
        """현재 작업 폴더의 worklog.txt를 다시 읽어 상태 캐시를 갱신합니다."""
        if self.current_work_dir is None:
            self.work_statuses = {}
            return
        self.work_statuses = load_worklog_statuses(self.current_work_dir, self.current_image_paths)

    def set_image_work_status(self, image_path: Path, status: WorkStatus) -> None:
        """개별 이미지의 작업 상태를 갱신하고 worklog.txt에 즉시 반영합니다."""
        if self.current_work_dir is None:
            return
        self.work_statuses = set_work_status(
            self.current_work_dir,
            self.current_image_paths,
            self.work_statuses,
            image_path,
            status,
        )

    def mark_current_image_reviewed(self) -> None:
        """현재 이미지가 사용자의 수동 작업 대상이었음을 worklog에 기록합니다."""
        if self.current_image_index < 0 or self.current_image_index >= len(self.current_image_paths):
            return
        image_path = self.current_image_paths[self.current_image_index]
        self.set_image_work_status(image_path, "v")

    def confirm_no_object_review(self) -> None:
        """객체가 없는 이미지를 검토 완료로 표시할지 사용자에게 확인합니다."""
        if self.current_image_index < 0 or self.current_image_index >= len(self.current_image_paths):
            return

        image_path = self.current_image_paths[self.current_image_index]
        answer = QMessageBox.question(
            self,
            "사용자 검토 확인창",
            "해당 이미지를 검토한 것으로 변경하겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.set_image_work_status(image_path, "v")
            self._refresh_work_status(image_path)
            self._refresh_current_image_item_color()
            self.update_training_availability()
            self.statusBar().showMessage(f"검토 완료 표시: {image_path.name}")
            return

        if answer == QMessageBox.StandardButton.No:
            self.set_image_work_status(image_path, "n")
            self._refresh_work_status(image_path)
            self._refresh_current_image_item_color()
            self.update_training_availability()
            self.statusBar().showMessage(f"미검토 표시: {image_path.name}")
            return

        self.statusBar().showMessage(f"상태 유지: {image_path.name}")

    def _apply_image_item_status_color(self, item: QListWidgetItem, image_path: Path) -> None:
        """worklog 상태에 따라 이미지 리스트 배경색을 지정합니다."""
        status = work_status_for_image(self.work_statuses, image_path)
        if self.config.theme_mode == "dark":
            reviewed_color = QColor("#2f4a34")
            auto_color = QColor("#5b4224")
            todo_color = QColor("#4f2f35")
            text_color = QColor("#f3f3f3")
        else:
            reviewed_color = QColor("#dff5d8")
            auto_color = QColor("#f6dfc9")
            todo_color = QColor("#f9dce6")
            text_color = QColor("#202020")
        if status == "v":
            item.setBackground(reviewed_color)
        elif status == "a":
            item.setBackground(auto_color)
        else:
            item.setBackground(todo_color)
        # 항목 글자색을 명시적으로 고정해 다크/라이트 모드와 상태 배경색 충돌을 방지합니다.
        item.setForeground(text_color)

    def load_image_by_index(self, index: int) -> None:
        """이미지 인덱스를 기준으로 파일과 기존 라벨을 불러옵니다."""
        if index < 0 or index >= len(self.current_image_paths):
            return

        self.current_image_index = index
        image_path = self.current_image_paths[index]
        self.current_labels = load_labels(image_path)
        self.canvas.load_image(image_path, self.current_labels)
        if self.class_names:
            active_index = min(self.selected_class_index, len(self.class_names) - 1)
            self.canvas.set_active_class_info(
                CLASS_COLORS[active_index % len(CLASS_COLORS)],
                self.class_names[active_index],
            )
        if self.image_list.currentRow() != index:
            self.image_list.setCurrentRow(index)
        self._refresh_image_navigation_controls()
        self._refresh_work_status(image_path)
        self.enter_hand_mode()
        self.update_training_availability()
        self.statusBar().showMessage(f"현재 이미지: {image_path.name}")

    def load_image_from_item(self, item: QListWidgetItem) -> None:
        """이미지 목록 더블 클릭 시 해당 항목의 이미지를 불러옵니다."""
        row = self.image_list.row(item)
        self.load_image_by_index(row)

    def _refresh_work_status(self, image_path: Path) -> None:
        """현재 이미지의 worklog 상태를 신호등 색상으로 갱신합니다."""
        status = work_status_for_image(self.work_statuses, image_path)
        if status == "v":
            self.status_light.setStyleSheet("font-size: 28pt; color: #52c41a; border: none;")
            self.status_text.setText("사용자 검토 완료")
        elif status == "a":
            self.status_light.setStyleSheet("font-size: 28pt; color: #fa8c16; border: none;")
            self.status_text.setText("오토 세그 완료")
        else:
            self.status_light.setStyleSheet("font-size: 28pt; color: #ff4d4f; border: none;")
            self.status_text.setText("미검토")

    def add_polygon_from_canvas(self, points: list[tuple[float, float]]) -> None:
        """캔버스에서 확정된 폴리곤을 현재 클래스의 라벨로 저장합니다."""
        if not self.current_image_paths or not self.class_names:
            return
        if len(points) < 3:
            return

        color_hex = CLASS_COLORS[self.selected_class_index % len(CLASS_COLORS)]
        self.current_labels.append(
            LabelPolygon(
                class_index=self.selected_class_index,
                points=points,
                color_hex=color_hex,
            )
        )
        self.canvas.set_labels(self.current_labels)
        self._save_current_labels()
        self.mark_current_image_reviewed()
        self._refresh_work_status(self.current_image_paths[self.current_image_index])
        self._refresh_current_image_item_color()
        self.enter_hand_mode()
        self.update_training_availability()

    def request_mask(self, points: list[tuple[float, float]]) -> None:
        """선택한 폴리곤 영역을 검정색으로 덮어쓸지 확인한 뒤 원본 이미지에 저장합니다."""
        if self.current_image_index < 0 or not points or len(points) < 3:
            return

        answer = QMessageBox.question(
            self,
            "마스킹 확인",
            "선택한 영역을 검정색으로 덮어쓰고 원본 이미지에 저장합니다.\n계속하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.mode_label.setText("모드: 마스킹 취소")
            self.enter_hand_mode()
            return

        image_path = self.current_image_paths[self.current_image_index]
        image = QImage(str(image_path))
        if image.isNull():
            QMessageBox.warning(self, "이미지 오류", "이미지를 다시 불러오지 못했습니다.")
            self.enter_hand_mode()
            return

        polygon = QPolygonF()
        for x_value, y_value in points:
            polygon.append(
                QPointF(
                    min(max(x_value, 0.0), 1.0) * max(1, image.width() - 1),
                    min(max(y_value, 0.0), 1.0) * max(1, image.height() - 1),
                )
            )

        painter = QPainter(image)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#000000"))
        path = QPainterPath()
        path.addPolygon(polygon)
        painter.drawPath(path)
        painter.end()

        if not image.save(str(image_path)):
            QMessageBox.warning(self, "저장 실패", "마스킹한 이미지를 저장하지 못했습니다.")
            self.enter_hand_mode()
            return

        self.canvas.load_image(image_path, self.current_labels)
        self.mark_current_image_reviewed()
        self._refresh_work_status(image_path)
        self._refresh_current_image_item_color()
        self.enter_hand_mode()
        self.update_training_availability()

    def _save_current_labels(self) -> None:
        """현재 이미지의 라벨 목록을 파일로 저장합니다."""
        if self.current_image_index < 0:
            return
        save_labels(self.current_image_paths[self.current_image_index], self.current_labels)

    def go_prev_image(self) -> None:
        """이전 이미지가 있을 때만 이동합니다."""
        if self.current_image_index <= 0:
            return
        self.reset_canvas_view()
        self.image_list.setCurrentRow(self.current_image_index - 1)

    def go_next_image(self) -> None:
        """다음 이미지가 있을 때만 이동합니다."""
        if self.current_image_index >= len(self.current_image_paths) - 1:
            return
        self.reset_canvas_view()
        self.image_list.setCurrentRow(self.current_image_index + 1)

    def delete_last_box(self) -> None:
        """현재 이미지의 마지막 폴리곤 라벨 하나를 제거합니다."""
        if not self.current_labels:
            return
        self.current_labels.pop()
        self.canvas.set_labels(self.current_labels)
        self._save_current_labels()
        self.mark_current_image_reviewed()
        self._refresh_work_status(self.current_image_paths[self.current_image_index])
        self._refresh_current_image_item_color()
        self.mode_label.setText("모드: 마지막 폴리곤 삭제")
        self.update_training_availability()

    def apply_edited_label(self, label_index: int, points: list[tuple[float, float]]) -> None:
        """편집 모드에서 변경된 폴리곤 좌표를 저장합니다."""
        if label_index < 0 or label_index >= len(self.current_labels):
            return
        if len(points) < 3:
            return
        label = self.current_labels[label_index]
        label.points = points
        self.canvas.set_labels(self.current_labels)
        self._save_current_labels()
        self.mark_current_image_reviewed()
        self._refresh_work_status(self.current_image_paths[self.current_image_index])
        self._refresh_current_image_item_color()
        self.update_training_availability()

    def apply_deleted_label(self, label_index: int) -> None:
        """편집 모드 우클릭으로 삭제된 폴리곤을 저장 상태에 반영합니다."""
        if label_index < 0 or label_index >= len(self.current_labels):
            return
        self.current_labels.pop(label_index)
        self.canvas.set_labels(self.current_labels)
        self._save_current_labels()
        self.mark_current_image_reviewed()
        self._refresh_work_status(self.current_image_paths[self.current_image_index])
        self._refresh_current_image_item_color()
        self.update_training_availability()

    def select_class(self, class_index: int) -> None:
        """현재 활성 클래스를 바꾸고 UI에도 반영합니다."""
        if not self.class_names:
            self.selected_class_index = class_index
            self.class_label.setText(f"선택 클래스: {class_index}")
            return
        if class_index < 0 or class_index >= len(self.class_names):
            return
        self.selected_class_index = class_index
        self.class_label.setText(f"선택 클래스: {class_index} ({self.class_names[class_index]})")
        self.class_list.setCurrentRow(class_index)
        self.canvas.set_active_class_info(
            CLASS_COLORS[class_index % len(CLASS_COLORS)],
            self.class_names[class_index],
        )

    def set_canvas_mode(self, mode: str) -> None:
        """캔버스의 현재 작업 모드를 전환합니다."""
        self.canvas.set_mode(mode)
        if mode == "draw":
            if self.config.rectangle_input_mode == "brush":
                self.mode_label.setText("모드: 브러시 그리기")
            else:
                self.mode_label.setText("모드: 폴리곤 클릭")
        elif mode == "mask":
            if self.config.rectangle_input_mode == "brush":
                self.mode_label.setText("모드: 브러시 마스킹")
            else:
                self.mode_label.setText("모드: 폴리곤 마스킹")
        elif mode == "edit":
            self.mode_label.setText("모드: 편집")
        else:
            self.mode_label.setText("모드: 손")

    def enter_hand_mode(self) -> None:
        """그리기/지우개 종료 후 기본 손 모드로 복귀합니다."""
        self.canvas.set_mode("hand")
        if self.canvas.zoom_factor > 1.0:
            self.mode_label.setText("모드: 손")
        else:
            self.mode_label.setText("모드: 대기")

    def handle_canvas_interaction_finished(self) -> None:
        """캔버스 작업 완료 후 현재 모드에 맞는 후속 동작을 적용합니다."""
        # 편집 모드는 점 이동/추가/삭제를 연속으로 이어서 할 수 있어야 하므로 자동 해제하지 않습니다.
        if self.canvas.current_mode == "edit":
            self.mode_label.setText("모드: 편집")
            return
        self.enter_hand_mode()

    def cancel_active_mode(self) -> None:
        """현재 작업 모드를 취소하고 손 모드로 전환합니다."""
        self.enter_hand_mode()

    def toggle_edit_mode(self) -> None:
        """편집 모드를 켜거나 끕니다."""
        if self.canvas.current_mode == "edit":
            self.enter_hand_mode()
            return
        self.set_canvas_mode("edit")

    def sync_image_spinbox_from_slider(self, value: int) -> None:
        """슬라이더 값을 스핀박스에 동기화하고 오토 라벨 경계도 함께 갱신합니다."""
        if self.image_index_spinbox.value() != value:
            self.image_index_spinbox.setValue(value)
        self.persist_auto_label_stop_index()

    def sync_image_slider_from_spinbox(self, value: int) -> None:
        """스핀박스 값을 슬라이더에 동기화합니다."""
        if self.image_index_slider.value() != value:
            self.image_index_slider.setValue(value)
        self.persist_auto_label_stop_index()

    def go_to_selected_image_index(self) -> None:
        """입력한 이미지 번호로 즉시 이동합니다."""
        if not self.current_image_paths:
            return
        target_index = self.image_index_spinbox.value() - 1
        if 0 <= target_index < len(self.current_image_paths):
            self.load_image_by_index(target_index)

    def _refresh_current_image_item_color(self) -> None:
        """현재 이미지 항목의 작업 상태 배경색을 다시 적용합니다."""
        if self.current_image_index < 0 or self.current_image_index >= len(self.current_image_paths):
            return
        item = self.image_list.item(self.current_image_index)
        if item is None:
            return
        self._apply_image_item_status_color(item, self.current_image_paths[self.current_image_index])

    def start_background_training(self) -> None:
        """현재 옵션으로 Temp 데이터셋 생성과 백그라운드 학습을 시작합니다."""
        if self.config.paths is None or self.current_work_dir is None:
            return
        if not is_training_ready(self.current_work_dir, self.current_image_paths, self.current_dataset_size()):
            QMessageBox.warning(self, "학습 불가", "지정한 학습 데이터량 범위 내 이미지가 모두 라벨링되어야 합니다.")
            self.update_training_availability()
            return

        model_path = self.config.paths.model_dir / self.config.selected_model
        if not model_path.exists():
            QMessageBox.warning(self, "모델 없음", "선택된 학습 시작 모델 파일을 찾지 못했습니다.")
            return
        if not self.hardware_status.python_command:
            QMessageBox.critical(self, "Python 없음", "외부 학습을 실행할 Python 환경을 찾지 못했습니다.")
            return
        if self.use_gpu_checkbox.isChecked() and not self.hardware_status.cuda_runtime_available:
            QMessageBox.warning(
                self,
                "GPU 학습 불가",
                "GPU 장치는 감지되었지만 학습 Python의 torch가 CUDA를 지원하지 않습니다.\n이번 학습은 CPU로 자동 전환합니다.",
            )

        request = TrainingRequest(
            python_command=self.hardware_status.python_command,
            work_dir=self.current_work_dir,
            image_paths=list(self.current_image_paths),
            class_names=list(self.class_names),
            dataset_size=self.current_dataset_size(),
            epochs=self.current_epochs(),
            image_size=self.current_image_size(),
            batch_size=self.current_batch_size(),
            project_name=self.current_project_name(),
            model_path=model_path,
            use_gpu=self.should_use_gpu_for_runtime(),
            ultralytics_dir=self.config.paths.ultralytics_dir,
            runner_script_path=self.config.paths.code_dir / "training_runner.py",
        )

        self.training_thread = QThread(self)
        self.training_worker = TrainingWorker(request)
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress_changed.connect(self.on_training_progress_changed)
        self.training_worker.status_changed.connect(self.on_training_status_changed)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.failed.connect(self.on_training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(self.cleanup_training_thread)

        self.training_progress_bar.setValue(0)
        self.training_progress_label.setText("0/0 (0%)")
        self.training_progress_panel.show()
        self.background_training_button.setEnabled(False)
        self.auto_label_button.setEnabled(False)
        self.training_thread.start()

    def start_auto_labeling(self) -> None:
        """학습 결과 PT로 현재 이미지부터 마지막 이미지까지 역순 오토 세그를 시작합니다."""
        if self.config.paths is None or self.current_work_dir is None or self.current_image_index < 0:
            return
        if not self.hardware_status.python_command:
            QMessageBox.critical(self, "Python 없음", "외부 오토 세그를 실행할 Python 환경을 찾지 못했습니다.")
            return
        if self.use_gpu_checkbox.isChecked() and not self.hardware_status.cuda_runtime_available:
            QMessageBox.warning(
                self,
                "GPU 추론 불가",
                "GPU 장치는 감지되었지만 학습 Python의 torch가 CUDA를 지원하지 않습니다.\n이번 오토 세그는 CPU로 자동 전환합니다.",
            )

        model_path = result_pt_path(self.current_work_dir)
        if not model_path.exists():
            QMessageBox.information(
                self,
                "학습 데이터 없음",
                "작업 폴더의 Temp/Result/result.pt가 없습니다.\n먼저 학습을 완료해 주세요.",
            )
            self.update_training_availability()
            return

        request = AutoLabelRequest(
            python_command=self.hardware_status.python_command,
            work_dir=self.current_work_dir,
            image_paths=list(self.current_image_paths),
            model_path=model_path,
            image_size=self.current_image_size(),
            conf_threshold=self.current_auto_label_conf(),
            use_gpu=self.should_use_gpu_for_runtime(),
            ultralytics_dir=self.config.paths.ultralytics_dir,
            runner_script_path=self.config.paths.code_dir / "auto_label_runner.py",
            stop_index_path=self.current_work_dir / "Temp" / "Result" / "auto_label_stop_index.txt",
            worklog_path=worklog_path(self.current_work_dir),
        )
        self.persist_auto_label_stop_index()

        self.auto_label_thread = QThread(self)
        self.auto_label_worker = AutoLabelWorker(request)
        self.auto_label_worker.moveToThread(self.auto_label_thread)
        self.auto_label_thread.started.connect(self.auto_label_worker.run)
        self.auto_label_worker.progress_changed.connect(self.on_auto_label_progress_changed)
        self.auto_label_worker.status_changed.connect(self.on_training_status_changed)
        self.auto_label_worker.finished.connect(self.on_auto_label_finished)
        self.auto_label_worker.failed.connect(self.on_auto_label_failed)
        self.auto_label_worker.finished.connect(self.auto_label_thread.quit)
        self.auto_label_worker.failed.connect(self.auto_label_thread.quit)
        self.auto_label_thread.finished.connect(self.cleanup_auto_label_thread)

        self.training_progress_bar.setValue(0)
        self.training_progress_label.setText("0/0 (0%)")
        self.training_progress_panel.show()
        self.background_training_button.setEnabled(False)
        self.auto_label_button.setEnabled(False)
        self.auto_label_thread.start()

    def open_result_validation_dialog(self) -> None:
        """result.onnx로 현재 검증 이미지에 대한 세그 추론 결과 검증 창을 엽니다."""
        if self.current_work_dir is None or self.config.paths is None:
            return
        model_path = result_onnx_path(self.current_work_dir)
        if not model_path.exists():
            QMessageBox.information(self, "검증 불가", "result.onnx가 없습니다.\n먼저 학습을 완료해 주세요.")
            return

        image_paths = verify_image_paths(self.current_work_dir)
        if not image_paths:
            QMessageBox.information(self, "검증 불가", "Temp/Verify 아래에 검증할 복제 이미지가 없습니다.")
            return

        if not self.hardware_status.python_command:
            QMessageBox.critical(self, "Python 없음", "검증 추론을 실행할 Python 환경을 찾지 못했습니다.")
            return

        self.validation_dialog = ResultValidationDialog(
            python_command=self.hardware_status.python_command,
            runner_script_path=str(self.config.paths.code_dir / "validation_runner.py"),
            model_path=str(model_path),
            image_paths=[str(path) for path in image_paths],
            class_names=list(self.class_names),
            conf_threshold=self.current_auto_label_conf(),
            use_gpu=self.should_use_gpu_for_runtime(),
            ultralytics_dir=str(self.config.paths.ultralytics_dir),
            parent=self,
        )
        self.validation_dialog.exec()

    def on_training_progress_changed(self, current_epoch: int, total_epochs: int) -> None:
        """epoch 진행률을 하단 진행바와 텍스트에 반영합니다."""
        percent = int((current_epoch / max(1, total_epochs)) * 100)
        self.training_progress_bar.setValue(percent)
        self.training_progress_label.setText(f"{current_epoch}/{total_epochs} ({percent}%)")

    def on_auto_label_progress_changed(self, current_index: int, total_count: int) -> None:
        """오토 라벨 진행률을 하단 진행바와 텍스트에 반영합니다."""
        percent = int((current_index / max(1, total_count)) * 100)
        self.training_progress_bar.setValue(percent)
        self.training_progress_label.setText(f"{current_index}/{total_count} ({percent}%)")

    def on_training_status_changed(self, message: str) -> None:
        """백그라운드 학습 상태 메시지를 상태바에 표시합니다."""
        self.statusBar().showMessage(message)

    def on_training_finished(self, onnx_path: str) -> None:
        """학습 완료 후 결과를 알리고 진행 UI를 숨깁니다."""
        self.training_progress_bar.setValue(100)
        QMessageBox.information(
            self,
            "학습 완료",
            f"학습과 ONNX export가 완료되었습니다.\n{onnx_path}",
        )
        self.training_progress_panel.hide()
        self.update_training_availability()

    def on_auto_label_finished(self, processed_count: int, labeled_count: int) -> None:
        """오토 세그 완료 후 이미지 목록과 현재 화면을 새 라벨 상태로 갱신합니다."""
        self.training_progress_bar.setValue(100)
        self.reload_worklog_statuses()
        self._refresh_image_list()
        if 0 <= self.current_image_index < len(self.current_image_paths):
            self.load_image_by_index(self.current_image_index)
        QMessageBox.information(
            self,
            "오토 세그 완료",
            f"처리 이미지: {processed_count}개\n라벨 생성 이미지: {labeled_count}개",
        )
        self.training_progress_panel.hide()
        self.update_training_availability()

    def on_training_failed(self, message: str) -> None:
        """학습 실패 시 오류를 알리고 진행 UI를 숨깁니다."""
        QMessageBox.critical(self, "학습 실패", message)
        self.training_progress_panel.hide()
        self.update_training_availability()

    def on_auto_label_failed(self, message: str) -> None:
        """오토 세그 실패 시 오류를 알리고 진행 UI를 숨깁니다."""
        QMessageBox.critical(self, "오토 세그 실패", message)
        self.training_progress_panel.hide()
        self.update_training_availability()

    def cleanup_training_thread(self) -> None:
        """종료된 학습 스레드와 워커 참조를 정리합니다."""
        if self.training_worker is not None:
            self.training_worker.deleteLater()
        if self.training_thread is not None:
            self.training_thread.deleteLater()
        self.training_worker = None
        self.training_thread = None
        self.update_training_availability()

    def cleanup_auto_label_thread(self) -> None:
        """종료된 오토 세그 스레드와 워커 참조를 정리합니다."""
        if self.auto_label_worker is not None:
            self.auto_label_worker.deleteLater()
        if self.auto_label_thread is not None:
            self.auto_label_thread.deleteLater()
        self.auto_label_worker = None
        self.auto_label_thread = None
        self.update_training_availability()

    def reset_canvas_view(self) -> None:
        """이미지를 화면 맞춤으로 되돌리고 손 모드 필요 여부를 갱신합니다."""
        self.canvas.reset_view()
        self.enter_hand_mode()

    def open_settings(self) -> None:
        """설정 다이얼로그를 열고 저장 후 즉시 반영합니다."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() == 0:
            return

        dialog.apply_to_config()
        save_app_config(self.config)
        updated_brush_size = max(2, int(self.config.runtime_options.get("brush_size", "18") or "18"))
        updated_polygon_point_count = max(
            3,
            int(self.config.runtime_options.get("polygon_point_count", "24") or "24"),
        )
        self.brush_size_slider.setValue(updated_brush_size)
        self.brush_size_spinbox.setValue(updated_brush_size)
        self.polygon_point_slider.setValue(updated_polygon_point_count)
        self.polygon_point_spinbox.setValue(updated_polygon_point_count)
        self.click_mode_radio.setChecked(self.config.rectangle_input_mode != "brush")
        self.brush_mode_radio.setChecked(self.config.rectangle_input_mode == "brush")
        self.canvas.set_input_mode(self.config.rectangle_input_mode)
        self.canvas.set_polygon_point_count(updated_polygon_point_count)
        self.canvas.set_brush_size(updated_brush_size)
        self._apply_theme()
        self._refresh_shortcut_guide()
        self._refresh_image_list()
        self._install_shortcuts()

    def on_input_mode_radio_toggled(self) -> None:
        """라디오 버튼 선택에 맞춰 입력 모드와 설정 파일을 즉시 동기화합니다."""
        selected_mode = "brush" if self.brush_mode_radio.isChecked() else "click"
        if self.config.rectangle_input_mode == selected_mode:
            return
        self.config.rectangle_input_mode = selected_mode
        self.canvas.set_input_mode(selected_mode)
        save_app_config(self.config)
        if self.canvas.current_mode == "draw":
            self.set_canvas_mode("draw")
        elif self.canvas.current_mode == "mask":
            self.set_canvas_mode("mask")

    def toggle_input_mode_shortcut(self) -> None:
        """C 단축키로 클릭 폴리곤 입력과 브러시 입력을 빠르게 전환합니다."""
        if self.config.rectangle_input_mode == "brush":
            self.click_mode_radio.setChecked(True)
            self.statusBar().showMessage("입력 모드 변경: 윤곽선 클릭")
        else:
            self.brush_mode_radio.setChecked(True)
            self.statusBar().showMessage("입력 모드 변경: 브러시 드래그")

    def toggle_theme(self) -> None:
        """라이트/다크 테마를 전환하고 설정 파일에 저장합니다."""
        self.config.theme_mode = "dark" if self.config.theme_mode == "light" else "light"
        save_app_config(self.config)
        self._apply_theme()
        self._refresh_image_list()
        if 0 <= self.current_image_index < len(self.current_image_paths):
            self._refresh_work_status(self.current_image_paths[self.current_image_index])

    def sync_brush_size_from_slider(self, value: int) -> None:
        """슬라이더 붓 크기를 스핀박스와 캔버스에 동기화합니다."""
        if self.brush_size_spinbox.value() != value:
            self.brush_size_spinbox.setValue(value)
        self.canvas.set_brush_size(value)
        self.save_runtime_option_values()

    def sync_brush_size_from_spinbox(self, value: int) -> None:
        """스핀박스 붓 크기를 슬라이더와 캔버스에 동기화합니다."""
        if self.brush_size_slider.value() != value:
            self.brush_size_slider.setValue(value)
        self.canvas.set_brush_size(value)
        self.save_runtime_option_values()

    def sync_polygon_point_count_from_slider(self, value: int) -> None:
        """메인 화면 슬라이더의 폴리곤 점 수를 스핀박스와 캔버스에 동기화합니다."""
        if self.polygon_point_spinbox.value() != value:
            self.polygon_point_spinbox.setValue(value)
        self.canvas.set_polygon_point_count(value)
        self.save_runtime_option_values()

    def sync_polygon_point_count_from_spinbox(self, value: int) -> None:
        """메인 화면 스핀박스의 폴리곤 점 수를 슬라이더와 캔버스에 동기화합니다."""
        if self.polygon_point_slider.value() != value:
            self.polygon_point_slider.setValue(value)
        self.canvas.set_polygon_point_count(value)
        self.save_runtime_option_values()
