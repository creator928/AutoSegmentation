# -*- coding: utf-8 -*-
"""두 자매 프로그램에서 공통 실행하는 이미지/라벨 회전 검사입니다."""

import os
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from autolabeler import models
from autolabeler.services import rotation_service as service
from autolabeler.services.label_service import save_labels, load_labels
from autolabeler.ui.canvas import ImageCanvas
from autolabeler.ui.main_window import MainWindow
from autolabeler.constants import DEFAULT_CONFIG


class RotationTests(unittest.TestCase):
    """PNG 픽셀, YOLO 좌표, 빈 라벨, 저장 실패와 캔버스 키를 검증합니다."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "image.png"
        self.image = QImage(6, 4, QImage.Format.Format_RGB32)
        self.image.fill(QColor("black"))
        self.image.setPixelColor(0, 0, QColor("red"))
        self.image.save(str(self.path))
        if hasattr(models, "LabelPolygon"):
            self.labels = [models.LabelPolygon(2, [(0.1, 0.2), (0.3, 0.2), (0.3, 0.6)], "#ff0000")]
        else:
            self.labels = [models.LabelBox(2, 0.2, 0.4, 0.2, 0.4, "#ff0000")]
        save_labels(self.path, self.labels)

    def tearDown(self):
        self.temp.cleanup()

    def test_clockwise_pixels_and_labels(self):
        result = service.rotate_image_and_labels(self.path, self.labels, True)
        image = QImage(str(self.path))
        self.assertEqual((image.width(), image.height()), (4, 6))
        self.assertEqual(image.pixelColor(3, 0), QColor("red"))
        if hasattr(result[0], "points"):
            self.assertEqual(result[0].points[0], (0.8, 0.1))
        else:
            self.assertEqual((result[0].x_center, result[0].y_center), (0.6, 0.2))
            self.assertAlmostEqual(result[0].width, 0.4)
            self.assertAlmostEqual(result[0].height, 0.2)
        self.assertEqual(load_labels(self.path)[0].class_index, 2)

    def test_counterclockwise_pixels_and_labels(self):
        result = service.rotate_image_and_labels(self.path, self.labels, False)
        self.assertEqual(QImage(str(self.path)).pixelColor(0, 5), QColor("red"))
        if hasattr(result[0], "points"):
            self.assertEqual(result[0].points[0], (0.2, 0.9))
        else:
            self.assertEqual((result[0].x_center, result[0].y_center), (0.4, 0.8))

    def test_four_rotations_restore_pixels_and_labels(self):
        original = self.path.with_suffix(".txt").read_bytes()
        labels = self.labels
        for _ in range(4):
            labels = service.rotate_image_and_labels(self.path, labels, True)
        self.assertEqual(QImage(str(self.path)), self.image)
        self.assertEqual(self.path.with_suffix(".txt").read_bytes(), original)

    def test_empty_labels_preserved(self):
        service.rotate_image_and_labels(self.path, [], True)
        self.assertEqual(self.path.with_suffix(".txt").read_text(), "")

    def test_failed_label_replace_rolls_back_image(self):
        original_image = self.path.read_bytes()
        original_label = self.path.with_suffix(".txt").read_bytes()
        replace = service.os.replace
        calls = 0
        def fail_second(src, dst):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("test label failure")
            replace(src, dst)
        with patch.object(service.os, "replace", side_effect=fail_second):
            with self.assertRaises(OSError):
                service.rotate_image_and_labels(self.path, self.labels, True)
        self.assertEqual(self.path.read_bytes(), original_image)
        self.assertEqual(self.path.with_suffix(".txt").read_bytes(), original_label)

    def test_page_keys_work_in_hand_and_edit_modes(self):
        canvas = ImageCanvas()
        canvas.load_image(self.path, self.labels)
        received = []
        canvas.rotation_requested.connect(received.append)
        for mode in ("hand", "edit"):
            canvas.set_mode(mode)
            QTest.keyClick(canvas, Qt.Key.Key_PageUp)
            QTest.keyClick(canvas, Qt.Key.Key_PageDown)
        self.assertEqual(received, [False, True, False, True])
        canvas.deleteLater()

    def test_canvas_key_saves_and_refreshes_window(self):
        """캔버스 입력부터 파일 저장과 화면 갱신까지 실제 신호 연결을 검증합니다."""
        root = self.path.parent
        paths = models.AppPaths(root, root, root, root, root, root, root / "settings.json", root)
        config = models.AppConfig(**copy.deepcopy(DEFAULT_CONFIG), paths=paths)
        hardware = SimpleNamespace(python_command=[], gpu_available=False, gpu_name="", cpu_name="test", cuda_runtime_available=False)
        with patch("autolabeler.ui.main_window.detect_hardware", return_value=hardware), patch.object(MainWindow, "showMaximized"):
            window = MainWindow(config)
        try:
            window.current_work_dir = root
            window.current_image_paths = [self.path]
            window.class_names = ["zero", "one", "two"]
            window._refresh_image_list()
            window.load_image_by_index(0)
            QTest.keyClick(window.canvas, Qt.Key.Key_PageDown)
            self.assertEqual((window.canvas.image.width(), window.canvas.image.height()), (4, 6))
            self.assertEqual(QImage(str(self.path)).pixelColor(3, 0), QColor("red"))
            window.auto_label_thread = object()
            QTest.keyClick(window.canvas, Qt.Key.Key_PageDown)
            self.assertEqual((window.canvas.image.width(), window.canvas.image.height()), (4, 6))
            window.auto_label_thread = None
        finally:
            self.app.removeEventFilter(window)
            window.close()
            window.deleteLater()


if __name__ == "__main__":
    unittest.main()
