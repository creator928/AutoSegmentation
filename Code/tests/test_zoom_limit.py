# -*- coding: utf-8 -*-
"""두 프로그램의 1000% 확대 제한과 축소 동작을 검증합니다."""
import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPoint, QPointF
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from autolabeler.ui.canvas import ImageCanvas


class ZoomLimitTests(unittest.TestCase):
    """직접 설정과 휠 입력에 동일한 상한이 적용되는지 확인합니다."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.canvas = ImageCanvas()
        self.canvas.resize(800, 600)
        self.canvas.image = QImage(800, 600, QImage.Format.Format_RGB32)

    def tearDown(self):
        self.canvas.deleteLater()

    def test_direct_zoom_limit(self):
        self.canvas.zoom_to_widget_point(QPoint(400, 300), 10.0)
        self.assertEqual(self.canvas.zoom_factor, 10.0)
        self.canvas.zoom_to_widget_point(QPoint(400, 300), 20.0)
        self.assertEqual(self.canvas.zoom_factor, 10.0)

    def test_wheel_limit_and_zoom_out(self):
        self.canvas.zoom_to_widget_point(QPoint(400, 300), 9.9)
        event = MagicMock()
        event.position.return_value = QPointF(400, 300)
        event.angleDelta.return_value = QPoint(0, 120)
        self.canvas.wheelEvent(event)
        self.assertEqual(self.canvas.zoom_factor, 10.0)
        self.canvas.wheelEvent(event)
        self.assertEqual(self.canvas.zoom_factor, 10.0)
        event.angleDelta.return_value = QPoint(0, -120)
        self.canvas.wheelEvent(event)
        self.assertEqual(self.canvas.zoom_factor, 9.0)
