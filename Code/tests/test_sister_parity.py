# -*- coding: utf-8 -*-
"""자매 프로그램 공통 편집/검토/학습 흐름의 회귀를 검증합니다."""

import copy
import contextlib
import io
import os
import tempfile
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from autolabeler.config import save_app_config
from autolabeler.constants import DEFAULT_CONFIG, MODEL_OPTIONS
from autolabeler.models import AppConfig, AppPaths, LabelPolygon
from autolabeler.services.label_service import save_labels
from autolabeler.services.auto_label_worker import AutoLabelRequest, AutoLabelWorker
from autolabeler.services.training_service import is_training_ready
from autolabeler.ui.dialogs import SettingsDialog
from autolabeler.ui.main_window import MainWindow
import auto_label_runner


class SisterParityTests(unittest.TestCase):
    """실제 Qt 위젯과 임시 데이터로 저장 및 선택 흐름을 확인합니다."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        paths = AppPaths(root, root, root, root, root, root, root / "settings.json", root)
        self.config = AppConfig(**copy.deepcopy(DEFAULT_CONFIG), paths=paths)
        hardware = SimpleNamespace(python_command=[], gpu_available=False, gpu_name="", cpu_name="test", cuda_runtime_available=False)
        with patch("autolabeler.ui.main_window.detect_hardware", return_value=hardware), patch.object(MainWindow, "showMaximized"):
            self.window = MainWindow(self.config)
        self.window.current_work_dir = root
        self.window.class_names = ["class0"]
        self.images = [root / "first.png", root / "second.png"]
        image = QImage(100, 100, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        for path in self.images:
            image.save(str(path))
        self.window.current_image_paths = self.images
        self.window._refresh_image_list()
        self.window.load_image_by_index(0)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.app.removeEventFilter(self.window)
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def add_label(self):
        """검증용 폴리곤을 화면과 저장 상태에 설정합니다."""
        label = LabelPolygon(0, [(0.2, 0.2), (0.5, 0.2), (0.5, 0.5)], "#00ff00")
        self.window.current_labels = [label]
        self.window.canvas.set_labels(self.window.current_labels)
        self.window._save_current_labels()
        return label

    def test_list_selection_enters_edit_and_arrow_moves(self):
        label = self.add_label()
        self.window.image_label_list.setCurrentRow(0)
        self.app.processEvents()
        canvas = self.window.canvas
        self.assertEqual(canvas.current_mode, "edit")
        self.assertTrue(canvas.hasFocus())
        width = canvas.image.width()
        QTest.keyClick(canvas, Qt.Key.Key_Right)
        self.assertAlmostEqual(label.points[0][0], 0.2 + 1 / width)
        QTest.keyClick(canvas, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.assertAlmostEqual(label.points[0][0], 0.2 + 11 / width)
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=canvas.image_to_widget_point(QPoint(90, 90)).toPoint())
        self.assertIsNone(canvas.selected_label_index)
        self.assertFalse(self.window.image_label_list.selectedItems())

    def test_magnifier_tracks_click_coordinate_at_quick_zoom(self):
        """300% 확대에서 클릭 좌표 중심의 1.5배 확대경이 표시되고 초기화 시 숨겨지는지 확인합니다."""
        canvas = self.window.canvas
        pointer = canvas.target_rect().center()
        canvas.mouseMoveEvent(SimpleNamespace(position=lambda: QPointF(pointer)))
        canvas.zoom_to_widget_point(pointer, 3.0)
        self.app.processEvents()

        preview = self.window.magnifier_preview
        self.assertFalse(preview.isHidden())
        self.assertEqual(preview.image_point, canvas.widget_to_image_point(pointer))
        self.assertAlmostEqual(preview.preview_scale, canvas.original_display_scale() * 1.5)
        self.assertAlmostEqual(preview.source_rect().center().x(), preview.image_point.x())
        self.assertAlmostEqual(preview.source_rect().center().y(), preview.image_point.y())
        crosshair_rects = preview.crosshair_rects()
        self.assertEqual(len(crosshair_rects), 5)
        self.assertAlmostEqual(crosshair_rects[0].center().x(), preview.width() / 2.0)
        self.assertAlmostEqual(crosshair_rects[0].center().y(), preview.height() / 2.0)

        canvas.reset_view()
        self.app.processEvents()
        self.assertTrue(preview.isHidden())

    def test_confirmation_dialogs_default_to_yes(self):
        """삭제·마스킹 확인창에서 스페이스로 승인할 수 있도록 Yes가 기본 버튼인지 확인합니다."""
        label = self.add_label()
        no_button = QMessageBox.StandardButton.No
        yes_button = QMessageBox.StandardButton.Yes

        with patch.object(QMessageBox, "question", return_value=no_button) as question:
            self.window.confirm_delete_label(label, 0)
            self.assertEqual(question.call_args.args[-1], yes_button)

            question.reset_mock()
            self.window.confirm_delete_current_image_pair(
                self.images[0],
                self.images[0].with_suffix(".txt"),
                0,
            )
            self.assertEqual(question.call_args.args[-1], yes_button)

            question.reset_mock()
            self.window.request_mask(label.points)
            self.assertEqual(question.call_args.args[-1], yes_button)

    def test_review_creates_empty_label_and_advances(self):
        for answer, status in ((QMessageBox.StandardButton.Yes, "v"), (QMessageBox.StandardButton.No, "n")):
            self.window.load_image_by_index(0)
            with patch.object(QMessageBox, "question", return_value=answer):
                self.window.confirm_no_object_review()
            self.assertEqual(self.images[0].with_suffix(".txt").read_text(), "")
            self.assertEqual(self.window.work_statuses[self.images[0]], status)
            self.assertEqual(self.window.current_image_index, 1)

    def test_review_preserves_labels_and_cancel_stays(self):
        self.add_label()
        original = self.images[0].with_suffix(".txt").read_bytes()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.confirm_no_object_review()
        self.assertEqual(self.images[0].with_suffix(".txt").read_bytes(), original)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel):
            self.window.confirm_no_object_review()
        self.assertEqual(self.window.current_image_index, 1)
        self.assertFalse(self.images[1].with_suffix(".txt").exists())

    def test_delete_last_label_preserves_empty_file(self):
        self.add_label()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
            self.window.apply_deleted_label(0)
        self.assertEqual(len(self.window.current_labels), 1)
        with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
            self.window.apply_deleted_label(0)
        self.assertEqual(self.window.current_labels, [])
        self.assertEqual(self.images[0].with_suffix(".txt").read_text(), "")
        self.assertTrue(self.window.canvas.erased_feedback_active)

    def test_boundary_movement_preserves_polygon_shape(self):
        label = self.add_label()
        canvas = self.window.canvas
        canvas.set_selected_label_index(0)
        canvas._move_selected_label_by_pixels(100000, -100000)
        self.assertAlmostEqual(max(x for x, _ in label.points), 1.0)
        self.assertAlmostEqual(min(y for _, y in label.points), 0.0)
        self.assertAlmostEqual(label.points[1][0] - label.points[0][0], 0.3)
        self.assertAlmostEqual(label.points[2][1] - label.points[1][1], 0.3)

    def test_vertex_add_remove_and_clone_remain_available(self):
        label = self.add_label()
        canvas = self.window.canvas
        canvas.set_mode("edit")
        canvas._insert_vertex_on_nearest_segment(QPoint(35, 20))
        self.assertEqual(len(label.points), 4)
        canvas._remove_nearest_vertex(QPoint(35, 20))
        self.assertEqual(len(label.points), 3)
        canvas._remove_nearest_vertex(QPoint(20, 20))
        self.assertEqual(len(label.points), 3)
        self.assertTrue(canvas._begin_edit(QPoint(40, 30), Qt.KeyboardModifier.ControlModifier))
        canvas._finish_edit()
        self.assertEqual(len(self.window.current_labels), 2)
        self.assertIsNot(label.points, self.window.current_labels[1].points)

    def test_no_detection_keeps_empty_txt(self):
        """무검출 재추론이 기존 라벨을 빈 TXT로 바꾸고 자동 처리 상태를 기록하는지 확인합니다."""
        self.add_label()
        root = self.window.current_work_dir
        # 자동 처리 허용 상태와 명시적인 정지 경계를 준비합니다.
        for image in self.images:
            self.window.set_image_work_status(image, "a")
        (root / "stop.txt").write_text("-1", encoding="utf-8")
        manifest = root / "manifest.txt"
        manifest.write_text("\n".join(map(str, self.images)), encoding="utf-8")
        args = SimpleNamespace(model="unused.pt", manifest=str(manifest), imgsz=640, conf=0.01,
                               max_polygon_points=24,
                               device="cpu", ultralytics_dir=str(root), stop_index_path=str(root / "stop.txt"),
                               worklog_path=str(root / "worklog.txt"))
        fake = MagicMock()
        fake.YOLO.return_value.predict.return_value = []
        with patch.object(auto_label_runner, "parse_args", return_value=args), patch.dict(
            "sys.modules", {"ultralytics": fake, "ultralytics.utils": MagicMock()}
        ), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(auto_label_runner.main(), 0)
        for image in self.images:
            self.assertEqual(image.with_suffix(".txt").read_text(), "")
        self.window.reload_worklog_statuses()
        self.assertTrue(all(status == "a" for status in self.window.work_statuses.values()))

    def test_auto_point_limit_persists_separately(self):
        """자동 제한값 변경이 수동 브러시 설정을 바꾸지 않고 저장되는지 확인합니다."""
        import json
        manual = self.config.runtime_options["polygon_point_count"]
        self.window.auto_label_max_points_spinbox.setValue(12)
        saved = json.loads(self.config.paths.settings_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["runtime_options"]["auto_label_max_points"], "12")
        self.assertEqual(saved["runtime_options"]["polygon_point_count"], manual)

    def test_polygon_limit_bounds_and_preserves_small_contours(self):
        """여러 상한과 대칭 윤곽에서도 최소 3점 및 최대 점 개수를 보장합니다."""
        angle = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        circle = np.column_stack((0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)))
        for limit in (3, 4, 12, 24, 256):
            result = auto_label_runner.limit_polygon_points(circle, limit)
            self.assertGreaterEqual(len(result), 3)
            self.assertLessEqual(len(result), limit)
            self.assertTrue(np.all((result >= 0) & (result <= 1)))
        triangle = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
        self.assertIs(auto_label_runner.limit_polygon_points(triangle, 24), triangle)
        with self.assertRaises(ValueError):
            auto_label_runner.limit_polygon_points(circle, 2)

    def test_runner_saves_limited_polygons(self):
        """실제 러너 저장 경로에서 세그먼트별 점 수 제한과 클래스 유지 여부를 검증합니다."""
        root = self.window.current_work_dir
        (root / "stop.txt").write_text("-1", encoding="utf-8")
        self.window.set_image_work_status(self.images[0], "n")
        manifest = root / "manifest.txt"
        manifest.write_text(str(self.images[0]), encoding="utf-8")
        angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        polygon = np.column_stack((0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)))
        detection = SimpleNamespace(boxes=SimpleNamespace(cls=np.array([0, 1])),
                                    masks=SimpleNamespace(xyn=[polygon, polygon]))
        fake = MagicMock()
        fake.YOLO.return_value.predict.return_value = [detection]
        args = SimpleNamespace(model="unused.pt", manifest=str(manifest), imgsz=640, conf=0.01,
                               device="cpu", ultralytics_dir=str(root), max_polygon_points=8,
                               stop_index_path=str(root / "stop.txt"), worklog_path=str(root / "worklog.txt"))
        with patch.object(auto_label_runner, "parse_args", return_value=args), patch.dict(
            "sys.modules", {"ultralytics": fake, "ultralytics.utils": MagicMock()}
        ), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(auto_label_runner.main(), 0)
        lines = self.images[0].with_suffix(".txt").read_text().splitlines()
        self.assertEqual(len(lines), 2)
        for index, line in enumerate(lines):
            tokens = line.split()
            self.assertEqual(int(tokens[0]), index)
            self.assertLessEqual((len(tokens) - 1) // 2, 8)
            self.assertGreaterEqual((len(tokens) - 1) // 2, 3)

    def test_worker_passes_point_limit_to_runner(self):
        """GUI 요청의 제한값이 외부 추론 프로세스 인자로 전달되는지 확인합니다."""
        root = self.window.current_work_dir
        request = AutoLabelRequest(
            python_command=["python"], work_dir=root, image_paths=self.images,
            model_path=root / "result.pt", image_size=640, conf_threshold=0.01,
            use_gpu=False, ultralytics_dir=root, runner_script_path=root / "auto_label_runner.py",
            stop_index_path=root / "stop.txt", worklog_path=root / "worklog.txt", max_polygon_points=17,
        )
        process = MagicMock()
        process.stdout = [b"AUTO_LABELER_DONE|2|2"]
        process.wait.return_value = 0
        worker = AutoLabelWorker(request)
        errors = []
        worker.failed.connect(errors.append)
        with patch("autolabeler.services.auto_label_worker.subprocess.Popen", return_value=process) as popen:
            worker.run()
        self.assertEqual(errors, [])
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--max-polygon-points") + 1], "17")

    def test_auto_labels_are_training_ready_without_review(self):
        for image in self.images:
            save_labels(image, [])
        self.assertTrue(is_training_ready(self.window.current_work_dir, self.images, 2))
        self.assertFalse(is_training_ready(self.window.current_work_dir, self.images, 3))

    def test_model_selection_persists(self):
        dialog = SettingsDialog(self.config)
        dialog.model_combo.setCurrentIndex(dialog.model_combo.findData("yolo26l-seg.pt"))
        dialog.apply_to_config()
        save_app_config(self.config)
        self.assertEqual(self.config.selected_model, "yolo26l-seg.pt")
        self.assertIn("yolo26l-seg.pt", self.config.paths.settings_path.read_text(encoding="utf-8"))
        self.assertTrue(all(option.weight_name.endswith("-seg.pt") for option in MODEL_OPTIONS))
        dialog.deleteLater()

    def test_validation_uses_existing_images_without_python(self):
        root = self.window.current_work_dir
        result = root / "Temp" / "Result"
        result.mkdir(parents=True, exist_ok=True)
        (result / "result.pt").touch()
        verify = root / "Temp" / "Verify"
        verify.mkdir(parents=True, exist_ok=True)
        QImage(str(self.images[0])).save(str(verify / "first.png"))
        with patch("autolabeler.ui.main_window.VerifyImageDialog") as dialog:
            self.window.open_result_validation_dialog()
        dialog.assert_called_once_with(image_paths=[str(verify / "first.png")], parent=self.window)
        dialog.return_value.exec.assert_called_once()


if __name__ == "__main__":
    unittest.main()
