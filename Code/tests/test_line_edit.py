# -*- coding: utf-8 -*-
"""두 자매 프로그램의 선분 선택, 드래그, 저장을 검증합니다."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication
from autolabeler import models
from autolabeler.ui.canvas import ImageCanvas
from autolabeler.services.label_service import save_labels, load_labels


class LineEditTests(unittest.TestCase):
    """점 우선순위와 라벨 형식별 선 이동 제약을 확인합니다."""
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.canvas = ImageCanvas()
        self.canvas.resize(1001, 1001)
        self.canvas.image = QImage(1001, 1001, QImage.Format.Format_RGB32)
        self.seg = hasattr(models, "LabelPolygon")
        if self.seg:
            self.label = models.LabelPolygon(0, [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)], "#00ff00")
        else:
            self.label = models.LabelBox(0, 0.5, 0.5, 0.6, 0.6, "#00ff00")
        self.canvas.labels = [self.label]
        self.canvas.set_mode("edit")

    def tearDown(self):
        self.canvas.deleteLater()

    def begin(self, point):
        """각 라벨 형식의 기존 편집 시작 API를 사용합니다."""
        if self.seg:
            return self.canvas._begin_edit(point, Qt.KeyboardModifier.NoModifier)
        return self.canvas._begin_edit(point)

    def test_point_precedes_line(self):
        self.canvas._update_hover_edit_target(QPoint(200, 200))
        if self.seg:
            self.assertEqual(self.canvas.hover_vertex_index, 0)
            self.assertIsNone(self.canvas.hover_segment_index)
        else:
            self.assertEqual(self.canvas.hover_edit_kind, "resize")

    def test_hover_displays_line(self):
        self.canvas.hover_point = QPoint(500, 200)
        self.canvas._update_hover_edit_target(self.canvas.hover_point)
        painter = MagicMock()
        self.canvas._draw_mode_guides(painter, self.canvas.target_rect())
        self.assertTrue(any(call.args[-1] == "Line" for call in painter.drawText.call_args_list))

    def test_point_precedes_overlapping_label_line(self):
        """위쪽 라벨의 선분보다 아래쪽 라벨의 꼭짓점을 우선합니다."""
        if self.seg:
            other = models.LabelPolygon(0, [(0.1, 0.2), (0.9, 0.2), (0.9, 0.9), (0.1, 0.9)], "#00ff00")
        else:
            other = models.LabelBox(0, 0.5, 0.55, 0.8, 0.7, "#00ff00")
        self.canvas.labels.append(other)
        self.canvas.hover_point = QPoint(202, 201)
        self.canvas._update_hover_edit_target(self.canvas.hover_point)
        self.assertEqual(self.canvas.hover_edit_index, 0)
        if self.seg:
            self.assertEqual(self.canvas.hover_vertex_index, 0)
            self.assertIsNone(self.canvas.hover_segment_index)
        else:
            self.assertEqual(self.canvas.hover_edit_kind, "resize")
        painter = MagicMock()
        self.canvas._draw_mode_guides(painter, self.canvas.target_rect())
        self.assertTrue(any(call.args[-1] == "Point" for call in painter.drawText.call_args_list))

    def test_line_moves_endpoints_and_saves(self):
        self.assertTrue(self.begin(QPoint(500, 200)))
        self.canvas._update_edit(QPoint(530, 300))
        if self.seg:
            self.assertAlmostEqual(self.label.points[0][0], 0.23)
            self.assertAlmostEqual(self.label.points[0][1], 0.3)
            self.assertAlmostEqual(self.label.points[1][1], 0.3)
            self.assertEqual(self.label.points[2:], [(0.8, 0.8), (0.2, 0.8)])
        else:
            self.assertAlmostEqual(self.label.y_center - self.label.height / 2, 0.2 + 100 / 1001)
            self.assertAlmostEqual(self.label.y_center + self.label.height / 2, 0.8)
            self.assertAlmostEqual(self.label.x_center, 0.5)
            self.assertAlmostEqual(self.label.width, 0.6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            self.canvas.label_edited.connect(lambda *args: save_labels(path, self.canvas.labels))
            self.canvas._finish_edit()
            self.assertEqual(len(load_labels(path)), 1)
            self.assertIsNone(self.canvas.edit_label_index)

    def test_closing_edge_or_left_side(self):
        self.assertTrue(self.begin(QPoint(200, 500)))
        self.canvas._update_edit(QPoint(100, 500))
        if self.seg:
            self.assertAlmostEqual(self.label.points[0][0], 0.1)
            self.assertAlmostEqual(self.label.points[3][0], 0.1)
            self.assertEqual(self.label.points[1:3], [(0.8, 0.2), (0.8, 0.8)])
        else:
            self.assertAlmostEqual(self.label.x_center - self.label.width / 2, 0.2 - 100 / 1001)
            self.assertAlmostEqual(self.label.x_center + self.label.width / 2, 0.8)

    def test_boundary_and_no_inversion(self):
        self.begin(QPoint(500, 200))
        self.canvas._update_edit(QPoint(500, -10000))
        if self.seg:
            self.assertAlmostEqual(self.label.points[0][1], 0)
            self.assertAlmostEqual(self.label.points[1][1], 0)
        else:
            self.assertAlmostEqual(self.label.y_center - self.label.height / 2, 0)
            self.canvas._update_edit(QPoint(500, 10000))
            self.assertGreaterEqual(self.label.height, 1 / 1001 - 1e-10)

    def test_interior_still_moves_whole_label(self):
        self.begin(QPoint(500, 500))
        if self.seg:
            self.assertIsNone(self.canvas.edit_segment_index)
            self.assertIsNone(self.canvas.edit_vertex_index)
        else:
            self.assertEqual(self.canvas.edit_mode_kind, "move")


if __name__ == "__main__":
    unittest.main()
