# -*- coding: utf-8 -*-
"""세그멘테이션 캔버스의 폴리곤, 브러시, 지우개, 편집 동작을 처리합니다."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import QApplication, QWidget

from ..models import LabelPolygon


class ImageCanvas(QWidget):
    """YOLO 세그멘테이션 폴리곤 표시와 입력 모드를 담당하는 캔버스입니다."""

    polygon_created = pyqtSignal(object)
    mask_requested = pyqtSignal(object)
    label_edited = pyqtSignal(int, object)
    label_deleted = pyqtSignal(int)
    interaction_finished = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.current_mode = "hand"
        self.input_mode = "click"
        self.image_path: Path | None = None
        self.image = QImage()
        self.labels: list[LabelPolygon] = []
        self.hover_point: QPoint | None = None
        self.zoom_factor = 1.0
        self.class_color_hex = "#7f7f7f"
        self.class_name = ""
        self.pan_offset = QPointF(0.0, 0.0)
        self.pan_anchor: QPoint | None = None
        self.pan_start_offset = QPointF(0.0, 0.0)
        self.click_points: list[QPointF] = []
        self.preview_point: QPointF | None = None
        self.brush_points: list[QPoint] = []
        self.brush_size = 18
        self.polygon_point_count = 24
        self.edit_label_index: int | None = None
        self.edit_vertex_index: int | None = None
        self.edit_last_point: QPointF | None = None
        self.hover_edit_index: int | None = None
        self.hover_vertex_index: int | None = None
        self.edit_is_clone = False
        self.eraser_target_index: int | None = None

    def set_input_mode(self, mode: str) -> None:
        """세그 입력 방식을 갱신합니다."""
        self.input_mode = mode
        self.click_points.clear()
        self.preview_point = None
        self.brush_points.clear()
        self.update()

    def set_brush_size(self, brush_size: int) -> None:
        """브러시/지우개 공용 원형 붓 크기를 갱신합니다."""
        self.brush_size = max(1, brush_size)
        self.update()

    def set_polygon_point_count(self, point_count: int) -> None:
        """브러시 기반 윤곽선을 단순화할 최대 좌표 수를 갱신합니다."""
        self.polygon_point_count = max(3, point_count)

    def load_image(self, image_path: Path, labels: list[LabelPolygon]) -> None:
        """새 이미지를 불러오고 화면 상태를 초기화합니다."""
        self.image_path = image_path
        self.image = QImage(str(image_path))
        self.labels = labels
        self.hover_point = None
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self.pan_anchor = None
        self.click_points.clear()
        self.preview_point = None
        self.brush_points.clear()
        self._clear_edit_state()
        self.update()

    def set_mode(self, mode: str) -> None:
        """현재 모드를 손, 그리기, 지우개, 편집 중 하나로 변경합니다."""
        self.current_mode = mode
        self.click_points.clear()
        self.preview_point = None
        self.brush_points.clear()
        self.pan_anchor = None
        self.eraser_target_index = None
        self._clear_edit_state()
        self._update_cursor()
        self.update()

    def set_active_class_info(self, color_hex: str, class_name: str) -> None:
        """현재 선택 클래스의 색상과 이름을 표시용으로 저장합니다."""
        self.class_color_hex = color_hex
        self.class_name = class_name
        self.update()

    def set_labels(self, labels: list[LabelPolygon]) -> None:
        """현재 라벨 목록을 갱신합니다."""
        self.labels = labels
        self.update()

    def reset_view(self) -> None:
        """이미지를 화면 맞춤 상태로 되돌립니다."""
        self.zoom_factor = 1.0
        self.pan_offset = QPointF(0.0, 0.0)
        self._update_cursor()
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """휠 입력으로 이미지를 확대/축소합니다."""
        if self.image.isNull():
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        scale_step = 1.1 if delta > 0 else 0.9
        self.zoom_factor = max(0.2, min(8.0, self.zoom_factor * scale_step))
        self._update_cursor()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """이미지, 기존 폴리곤, 편집 핸들, 입력 미리보기를 그립니다."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101010"))

        if self.image.isNull():
            painter.end()
            return

        target_rect = self.target_rect()
        painter.drawImage(target_rect, self.image)

        for index, label in enumerate(self.labels):
            polygon = self._normalized_points_to_widget_polygon(label.points)
            if polygon.count() < 3:
                continue
            fill_color = QColor(label.color_hex)
            fill_color.setAlpha(60)
            pen_width = 3 if self.current_mode == "edit" and self.edit_label_index == index else 2
            painter.setPen(QPen(QColor(label.color_hex), pen_width))
            painter.setBrush(QBrush(fill_color))
            painter.drawPolygon(polygon)
            if self.current_mode == "edit":
                self._draw_vertex_handles(painter, polygon, QColor(label.color_hex))

        self._draw_click_preview(painter)
        self._draw_brush_preview(painter)
        self._draw_mode_guides(painter, target_rect)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """모드별 입력 시작을 처리합니다."""
        if self.image.isNull():
            return

        widget_point = self.clamp_to_image(event.position().toPoint())
        image_point = self.widget_to_image_point(widget_point)
        if widget_point is None or image_point is None:
            return

        if self.current_mode == "edit" and event.button() == Qt.MouseButton.RightButton:
            self._delete_label_at_point(image_point)
            return

        if self.current_mode in {"draw", "mask"} and self.input_mode == "click":
            if event.button() == Qt.MouseButton.RightButton:
                self._finalize_click_polygon()
                return
            if event.button() != Qt.MouseButton.LeftButton:
                return
            self._append_click_point(image_point)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.current_mode == "hand":
            if self.zoom_factor > 1.0:
                self.pan_anchor = widget_point
                self.pan_start_offset = QPointF(self.pan_offset)
                self._update_cursor(True)
            return

        if self.current_mode == "edit":
            modifiers = event.modifiers()
            # 편집 모드의 클릭 계열 단축 동작은 드래그 시작보다 먼저 처리합니다.
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                self._insert_vertex_on_nearest_segment(image_point)
                return
            if modifiers & Qt.KeyboardModifier.AltModifier:
                self._remove_nearest_vertex(image_point)
                return
            if not self._begin_edit(image_point, modifiers) and self.zoom_factor > 1.0:
                # 편집 대상이 없는 빈 영역은 확대 상태에서 손 도구처럼 화면 이동에 사용합니다.
                self.pan_anchor = widget_point
                self.pan_start_offset = QPointF(self.pan_offset)
                self._update_cursor(True)
            return

        if self.current_mode in {"draw", "mask"} and self.input_mode == "brush":
            self.brush_points = [image_point]
            self.update()
            return

        if self.current_mode == "erase":
            self.brush_points = [image_point]
            self.eraser_target_index = self._find_topmost_polygon_index(image_point)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """모드별 이동 중 갱신 처리를 수행합니다."""
        widget_point = self.clamp_to_image(event.position().toPoint())
        self.hover_point = widget_point
        image_point = self.widget_to_image_point(widget_point)

        if self.current_mode == "hand":
            if self.pan_anchor is not None and widget_point is not None:
                delta = widget_point - self.pan_anchor
                self.pan_offset = QPointF(self.pan_start_offset.x() + delta.x(), self.pan_start_offset.y() + delta.y())
            self.update()
            return

        if image_point is None:
            self.update()
            return

        if self.current_mode in {"draw", "mask"} and self.input_mode == "click":
            self.preview_point = QPointF(image_point)
            self.update()
            return

        if self.current_mode == "edit":
            if self.pan_anchor is not None and widget_point is not None:
                delta = widget_point - self.pan_anchor
                self.pan_offset = QPointF(self.pan_start_offset.x() + delta.x(), self.pan_start_offset.y() + delta.y())
                self.update()
                return
            self._update_hover_edit_target(image_point)
            self._update_edit(image_point)
            return

        if self.current_mode in {"draw", "mask", "erase"} and event.buttons() & Qt.MouseButton.LeftButton:
            self.brush_points.append(image_point)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """모드별 마우스 버튼 해제 처리를 수행합니다."""
        if self.current_mode == "hand":
            self.pan_anchor = None
            self._update_cursor()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self.current_mode == "edit":
            if self.pan_anchor is not None:
                self.pan_anchor = None
                self._update_cursor()
                return
            self._finish_edit()
            return

        if self.current_mode in {"draw", "mask"} and self.input_mode == "brush":
            self._finalize_brush_polygon()
            return

        if self.current_mode == "erase":
            self._apply_eraser_to_target()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """클릭 입력 모드에서는 더블클릭으로 폴리곤을 마감합니다."""
        if self.current_mode in {"draw", "mask"} and self.input_mode == "click":
            self._finalize_click_polygon()

    def target_rect(self) -> QRect:
        """현재 배율과 이동량을 반영한 실제 이미지 표시 영역을 반환합니다."""
        if self.image.isNull():
            return QRect()

        fit_size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        scaled_width = max(1, int(fit_size.width() * self.zoom_factor))
        scaled_height = max(1, int(fit_size.height() * self.zoom_factor))
        x = int((self.width() - scaled_width) // 2 + self.pan_offset.x())
        y = int((self.height() - scaled_height) // 2 + self.pan_offset.y())
        return QRect(x, y, scaled_width, scaled_height)

    def clamp_to_image(self, point: QPoint | None) -> QPoint | None:
        """포인터가 이미지 밖으로 나가도 가장자리 좌표로 보정해 반환합니다."""
        if point is None:
            return None
        target = self.target_rect()
        if target.isNull():
            return None
        x = min(max(point.x(), target.left()), target.right())
        y = min(max(point.y(), target.top()), target.bottom())
        return QPoint(x, y)

    def widget_to_image_point(self, point: QPoint | None) -> QPoint | None:
        """위젯 좌표를 실제 이미지 픽셀 좌표로 변환합니다."""
        if point is None or self.image.isNull():
            return None
        target = self.target_rect()
        if target.isNull():
            return None
        x_value = int(round((point.x() - target.left()) * self.image.width() / max(1, target.width())))
        y_value = int(round((point.y() - target.top()) * self.image.height() / max(1, target.height())))
        x_value = min(max(x_value, 0), max(0, self.image.width() - 1))
        y_value = min(max(y_value, 0), max(0, self.image.height() - 1))
        return QPoint(x_value, y_value)

    def image_to_widget_point(self, point: QPointF) -> QPointF:
        """실제 이미지 픽셀 좌표를 위젯 좌표로 변환합니다."""
        target = self.target_rect()
        return QPointF(
            target.left() + point.x() * target.width() / max(1, self.image.width()),
            target.top() + point.y() * target.height() / max(1, self.image.height()),
        )

    def _normalized_points_to_widget_polygon(self, points: list[tuple[float, float]]) -> QPolygonF:
        """정규화 좌표 폴리곤을 현재 화면 좌표 폴리곤으로 환산합니다."""
        target = self.target_rect()
        polygon = QPolygonF()
        for x_value, y_value in points:
            polygon.append(
                QPointF(
                    target.left() + x_value * target.width(),
                    target.top() + y_value * target.height(),
                )
            )
        return polygon

    def _image_points_to_widget_polygon(self, points: list[QPoint]) -> QPolygonF:
        """이미지 픽셀 좌표 폴리곤을 위젯 좌표 폴리곤으로 변환합니다."""
        polygon = QPolygonF()
        for point in points:
            polygon.append(self.image_to_widget_point(QPointF(point)))
        return polygon

    def _append_click_point(self, image_point: QPoint) -> None:
        """클릭 모드의 새 꼭짓점을 추가하고 필요 시 닫힌 폴리곤으로 확정합니다."""
        pointf = QPointF(image_point)
        if len(self.click_points) >= 3 and self._distance(pointf, self.click_points[0]) <= max(4.0, self.brush_size):
            self._finalize_click_polygon()
            return
        self.click_points.append(pointf)
        self.preview_point = pointf
        self.update()

    def _finalize_click_polygon(self) -> None:
        """클릭 모드에서 모아 둔 꼭짓점을 폴리곤으로 확정합니다."""
        if len(self.click_points) < 3:
            self.click_points.clear()
            self.preview_point = None
            self.update()
            return
        normalized_points = self._image_pointf_list_to_normalized(self.click_points)
        self.click_points.clear()
        self.preview_point = None
        if self.current_mode == "mask":
            self.mask_requested.emit(normalized_points)
        else:
            self.polygon_created.emit(normalized_points)
        self.interaction_finished.emit()
        self.update()

    def _finalize_brush_polygon(self) -> None:
        """브러시 스트로크를 가장 큰 외곽선 폴리곤으로 변환합니다."""
        if len(self.brush_points) < 2 or self.image.isNull():
            self.brush_points.clear()
            self.update()
            return
        mask = self._empty_mask()
        self._draw_brush_stroke(mask, self.brush_points, fill_value=255)
        normalized_points = self._mask_to_normalized_polygon(mask, self.polygon_point_count)
        self.brush_points.clear()
        if normalized_points:
            if self.current_mode == "mask":
                self.mask_requested.emit(normalized_points)
            else:
                self.polygon_created.emit(normalized_points)
            self.interaction_finished.emit()
        self.update()

    def _apply_eraser_to_target(self) -> None:
        """지우개 스트로크를 선택된 폴리곤에 적용합니다."""
        if self.eraser_target_index is None or self.eraser_target_index >= len(self.labels):
            self.brush_points.clear()
            self.eraser_target_index = None
            self.update()
            return

        target_label = self.labels[self.eraser_target_index]
        mask = self._polygon_to_mask(target_label.points)
        self._draw_brush_stroke(mask, self.brush_points, fill_value=0)
        normalized_points = self._mask_to_normalized_polygon(mask, self.polygon_point_count)
        self.brush_points.clear()

        if normalized_points:
            self.label_edited.emit(self.eraser_target_index, normalized_points)
        else:
            self.label_deleted.emit(self.eraser_target_index)
        self.eraser_target_index = None
        self.interaction_finished.emit()
        self.update()

    def _draw_brush_stroke(self, mask: np.ndarray, points: list[QPoint], fill_value: int) -> None:
        """원형 붓 스트로크를 마스크에 채웁니다."""
        # 실제 칠해지는 반경 계산을 한 곳으로 모아 미리보기와 결과의 기준을 일치시킵니다.
        radius = self._brush_radius_image()
        for index, point in enumerate(points):
            cv2.circle(mask, (point.x(), point.y()), radius, fill_value, thickness=-1)
            if index == 0:
                continue
            previous = points[index - 1]
            cv2.line(
                mask,
                (previous.x(), previous.y()),
                (point.x(), point.y()),
                fill_value,
                thickness=max(2, radius * 2),
            )

    def _empty_mask(self) -> np.ndarray:
        """현재 이미지 크기의 빈 마스크를 생성합니다."""
        return np.zeros((self.image.height(), self.image.width()), dtype=np.uint8)

    def _polygon_to_mask(self, points: list[tuple[float, float]]) -> np.ndarray:
        """정규화 폴리곤을 이미지 크기의 바이너리 마스크로 변환합니다."""
        mask = self._empty_mask()
        contour = []
        for x_value, y_value in points:
            contour.append(
                [
                    int(round(x_value * max(1, self.image.width() - 1))),
                    int(round(y_value * max(1, self.image.height() - 1))),
                ]
            )
        if len(contour) >= 3:
            cv2.fillPoly(mask, [np.array(contour, dtype=np.int32)], 255)
        return mask

    def _mask_to_normalized_polygon(self, mask: np.ndarray, target_point_count: int) -> list[tuple[float, float]]:
        """바이너리 마스크에서 가장 큰 외곽선을 추출해 YOLO 정규화 폴리곤으로 변환합니다."""
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) <= 1.0:
            return []

        perimeter = cv2.arcLength(contour, True)
        approximated = contour
        for factor in (0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016, 0.024, 0.032, 0.05):
            candidate = cv2.approxPolyDP(contour, factor * perimeter, True)
            if len(candidate) >= 3:
                approximated = candidate
            if 3 <= len(candidate) <= target_point_count:
                approximated = candidate
                break

        points = approximated.reshape(-1, 2).tolist()
        if len(points) > target_point_count:
            points = self._resample_points(points, target_point_count)
        if len(points) < 3:
            return []

        normalized_points: list[tuple[float, float]] = []
        for x_value, y_value in points:
            normalized_points.append(
                (
                    min(max(float(x_value) / max(1, self.image.width() - 1), 0.0), 1.0),
                    min(max(float(y_value) / max(1, self.image.height() - 1), 0.0), 1.0),
                )
            )
        return normalized_points

    def _resample_points(self, points: list[list[float]], target_point_count: int) -> list[list[float]]:
        """외곽선 좌표 수가 너무 많으면 균등 간격으로 줄입니다."""
        if len(points) <= target_point_count:
            return points
        sampled: list[list[float]] = []
        for index in range(target_point_count):
            source_index = int(round(index * (len(points) - 1) / max(1, target_point_count - 1)))
            sampled.append(points[source_index])
        return sampled

    def _image_pointf_list_to_normalized(self, points: list[QPointF]) -> list[tuple[float, float]]:
        """이미지 픽셀 좌표 목록을 정규화 좌표 목록으로 변환합니다."""
        normalized_points: list[tuple[float, float]] = []
        for point in points:
            normalized_points.append(
                (
                    min(max(point.x() / max(1, self.image.width() - 1), 0.0), 1.0),
                    min(max(point.y() / max(1, self.image.height() - 1), 0.0), 1.0),
                )
            )
        return normalized_points

    def _draw_click_preview(self, painter: QPainter) -> None:
        """클릭 입력 중인 폴리곤의 가이드 선을 그립니다."""
        if self.current_mode not in {"draw", "mask"} or self.input_mode != "click" or not self.click_points:
            return

        preview_polygon = QPolygonF()
        for point in self.click_points:
            preview_polygon.append(self.image_to_widget_point(point))
        if self.preview_point is not None:
            preview_polygon.append(self.image_to_widget_point(self.preview_point))

        preview_color = QColor("#ffffff") if self.current_mode == "mask" else QColor(self.class_color_hex)
        painter.setPen(QPen(preview_color, 2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        if preview_polygon.count() >= 2:
            painter.drawPolyline(preview_polygon)

        painter.setBrush(QBrush(preview_color))
        for point in self.click_points:
            painter.drawEllipse(self.image_to_widget_point(point), 3, 3)

    def _draw_brush_preview(self, painter: QPainter) -> None:
        """브러시/지우개 스트로크 미리보기를 그립니다."""
        if not self.brush_points:
            return

        path = QPainterPath()
        first_point = self.image_to_widget_point(QPointF(self.brush_points[0]))
        path.moveTo(first_point)
        for point in self.brush_points[1:]:
            path.lineTo(self.image_to_widget_point(QPointF(point)))

        color = QColor("#ffffff") if self.current_mode in {"mask", "erase"} else QColor(self.class_color_hex)
        color.setAlpha(180)
        # 미리보기 선의 굵기도 실제 브러시 반경을 위젯 좌표로 변환한 값과 동일하게 사용합니다.
        painter.setPen(
            QPen(
                color,
                max(2, self._brush_radius_widget() * 2),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawPath(path)

    def _draw_mode_guides(self, painter: QPainter, target_rect: QRect) -> None:
        """현재 모드에 맞는 십자선과 붓 원형 가이드를 그립니다."""
        if self.hover_point is None:
            return

        color = QColor("#9a9a9a")
        text = ""
        draw_brush_circle = False

        if self.current_mode == "draw":
            color = QColor(self.class_color_hex)
            text = "Polygon" if self.input_mode == "click" else "Brush"
            draw_brush_circle = self.input_mode == "brush"
        elif self.current_mode == "mask":
            color = QColor("#ffffff")
            text = "Mask"
            draw_brush_circle = self.input_mode == "brush"
        elif self.current_mode == "erase":
            color = QColor("#ffffff")
            text = "Eraser"
            draw_brush_circle = True
        elif self.current_mode == "edit":
            color = QColor("#ff9f1c")
            text = "Edit"

        painter.setPen(QPen(color, 1))
        painter.drawLine(self.hover_point.x(), target_rect.top(), self.hover_point.x(), target_rect.bottom())
        painter.drawLine(target_rect.left(), self.hover_point.y(), target_rect.right(), self.hover_point.y())

        if draw_brush_circle:
            radius = max(2, self._brush_radius_widget())
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawEllipse(self.hover_point, radius, radius)
        else:
            painter.setBrush(QBrush(color))
            painter.drawEllipse(self.hover_point, 2, 2)

        painter.drawText(self.hover_point + QPoint(6, -6), text)

    def _draw_vertex_handles(self, painter: QPainter, polygon: QPolygonF, color: QColor) -> None:
        """편집 모드에서 꼭짓점 핸들을 그립니다."""
        painter.setBrush(color)
        for point in polygon:
            painter.drawEllipse(point, 4, 4)

    def _begin_edit(self, image_point: QPoint, modifiers: Qt.KeyboardModifiers) -> bool:
        """편집 모드에서 꼭짓점 이동, 전체 이동, Ctrl 복제 이동을 시작합니다."""
        self._update_hover_edit_target(image_point)
        if self.hover_edit_index is None:
            return False

        # Ctrl+드래그는 현재 세그 폴리곤을 복제한 뒤 복제본을 바로 이동시키는 동작입니다.
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            source_label = self.labels[self.hover_edit_index]
            self.labels.append(
                LabelPolygon(
                    class_index=source_label.class_index,
                    points=list(source_label.points),
                    color_hex=source_label.color_hex,
                )
            )
            self.edit_label_index = len(self.labels) - 1
            self.edit_vertex_index = None
            self.edit_is_clone = True
            self.edit_last_point = QPointF(image_point)
            self.update()
            return True

        self.edit_label_index = self.hover_edit_index
        self.edit_vertex_index = self.hover_vertex_index
        self.edit_is_clone = False
        self.edit_last_point = QPointF(image_point)
        self.update()
        return True

    def _insert_vertex_on_nearest_segment(self, image_point: QPoint) -> None:
        """Shift+클릭 시 가장 가까운 선분 위에 새 꼭짓점을 삽입합니다."""
        nearest = self._find_nearest_segment_projection(image_point)
        if nearest is None:
            return

        label_index, insert_after_index, projected_point = nearest
        target_label = self.labels[label_index]
        updated_points = list(target_label.points)
        updated_points.insert(insert_after_index + 1, self._image_point_to_normalized_tuple(projected_point))
        target_label.points = updated_points

        # 즉시 저장 이벤트를 보내면 새 점이 들어간 상태로 바로 이어서 편집할 수 있습니다.
        self.label_edited.emit(label_index, list(updated_points))
        self.interaction_finished.emit()
        self.update()

    def _remove_nearest_vertex(self, image_point: QPoint) -> None:
        """Alt+클릭 시 가장 가까운 꼭짓점을 제거하되 최소 3개 점은 유지합니다."""
        self._update_hover_edit_target(image_point)
        if self.hover_edit_index is None or self.hover_vertex_index is None:
            return

        target_label = self.labels[self.hover_edit_index]
        if len(target_label.points) <= 3:
            return

        updated_points = list(target_label.points)
        updated_points.pop(self.hover_vertex_index)
        target_label.points = updated_points

        self.label_edited.emit(self.hover_edit_index, list(updated_points))
        self.interaction_finished.emit()
        self.update()

    def _update_edit(self, image_point: QPoint) -> None:
        """편집 중인 폴리곤을 현재 포인터 위치에 맞춰 갱신합니다."""
        if self.edit_label_index is None or self.edit_last_point is None:
            self.update()
            return

        points = list(self.labels[self.edit_label_index].points)
        if self.edit_vertex_index is not None:
            points[self.edit_vertex_index] = self._image_point_to_normalized_tuple(image_point)
        else:
            delta_x = image_point.x() - self.edit_last_point.x()
            delta_y = image_point.y() - self.edit_last_point.y()
            normalized_delta_x = delta_x / max(1, self.image.width() - 1)
            normalized_delta_y = delta_y / max(1, self.image.height() - 1)
            moved_points = []
            for x_value, y_value in points:
                moved_points.append(
                    (
                        min(max(x_value + normalized_delta_x, 0.0), 1.0),
                        min(max(y_value + normalized_delta_y, 0.0), 1.0),
                    )
                )
            points = moved_points

        self.labels[self.edit_label_index].points = points
        self.edit_last_point = QPointF(image_point)
        self.update()

    def _finish_edit(self) -> None:
        """편집이 끝나면 변경 내용을 외부로 알립니다."""
        if self.edit_label_index is None:
            return
        self.label_edited.emit(self.edit_label_index, list(self.labels[self.edit_label_index].points))
        self._clear_edit_state()
        self.update()

    def _clear_edit_state(self) -> None:
        """편집 중 임시 상태를 초기화합니다."""
        self.edit_label_index = None
        self.edit_vertex_index = None
        self.edit_last_point = None
        self.hover_edit_index = None
        self.hover_vertex_index = None
        self.edit_is_clone = False

    def _delete_label_at_point(self, image_point: QPoint) -> None:
        """편집 모드에서 우클릭한 폴리곤을 삭제합니다."""
        delete_index = self._find_topmost_polygon_index(image_point)
        if delete_index is None:
            return
        self.label_deleted.emit(delete_index)
        self.update()

    def _update_hover_edit_target(self, image_point: QPoint) -> None:
        """현재 포인터 위치에서 편집 가능한 꼭짓점 또는 폴리곤을 탐색합니다."""
        self.hover_edit_index = None
        self.hover_vertex_index = None
        nearest_label_index: int | None = None
        nearest_vertex_index: int | None = None
        nearest_distance: float | None = None
        vertex_hit_radius = self._edit_vertex_hit_radius_image()

        for index in range(len(self.labels) - 1, -1, -1):
            polygon = self._normalized_points_to_image_polygon(self.labels[index].points)
            for vertex_index, vertex_point in enumerate(polygon):
                distance = self._distance(QPointF(image_point), QPointF(vertex_point))
                if distance <= vertex_hit_radius and (nearest_distance is None or distance < nearest_distance):
                    nearest_label_index = index
                    nearest_vertex_index = vertex_index
                    nearest_distance = distance
            # 점 편집보다 우선하지는 않지만, 점 선택 반경에 걸리지 않을 때는 내부 클릭을 전체 이동으로 사용합니다.
            if self._polygon_contains_point(self.labels[index].points, image_point):
                if nearest_label_index is not None:
                    self.hover_edit_index = nearest_label_index
                    self.hover_vertex_index = nearest_vertex_index
                    return
                self.hover_edit_index = index
                self.hover_vertex_index = None
                return

        if nearest_label_index is not None:
            self.hover_edit_index = nearest_label_index
            self.hover_vertex_index = nearest_vertex_index

    def _find_topmost_polygon_index(self, image_point: QPoint) -> int | None:
        """현재 포인터 아래에 있는 가장 위 폴리곤의 인덱스를 반환합니다."""
        for index in range(len(self.labels) - 1, -1, -1):
            if self._polygon_contains_point(self.labels[index].points, image_point):
                return index
        return None

    def _find_nearest_segment_projection(self, image_point: QPoint) -> tuple[int, int, QPoint] | None:
        """포인터와 가장 가까운 선분 및 그 선분 위 투영 좌표를 찾습니다."""
        best_match: tuple[int, int, QPoint] | None = None
        best_distance: float | None = None
        max_distance = self._edit_vertex_hit_radius_image() * 2.0

        for label_index in range(len(self.labels) - 1, -1, -1):
            polygon = self._normalized_points_to_image_polygon(self.labels[label_index].points)
            if len(polygon) < 2:
                continue

            for segment_index in range(len(polygon)):
                start_point = polygon[segment_index]
                end_point = polygon[(segment_index + 1) % len(polygon)]
                projected_point, distance = self._project_point_to_segment(image_point, start_point, end_point)
                if distance > max_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_match = (label_index, segment_index, projected_point)
                    best_distance = distance

            # 화면상 가장 위의 폴리곤을 우선하기 위해 충분히 가까운 선분을 찾으면 바로 종료합니다.
            if best_match is not None and best_match[0] == label_index:
                return best_match

        return best_match

    def _normalized_points_to_image_polygon(self, points: list[tuple[float, float]]) -> list[QPoint]:
        """정규화 좌표 폴리곤을 이미지 픽셀 좌표 목록으로 환산합니다."""
        polygon: list[QPoint] = []
        for x_value, y_value in points:
            polygon.append(
                QPoint(
                    int(round(x_value * max(1, self.image.width() - 1))),
                    int(round(y_value * max(1, self.image.height() - 1))),
                )
            )
        return polygon

    def _polygon_contains_point(self, points: list[tuple[float, float]], image_point: QPoint) -> bool:
        """정규화 좌표 폴리곤 안에 이미지 포인트가 포함되는지 검사합니다."""
        polygon = QPolygonF()
        for point in self._normalized_points_to_image_polygon(points):
            polygon.append(QPointF(point))
        return polygon.containsPoint(QPointF(image_point), Qt.FillRule.OddEvenFill)

    def _image_point_to_normalized_tuple(self, image_point: QPoint) -> tuple[float, float]:
        """이미지 픽셀 좌표를 정규화 좌표로 변환합니다."""
        return (
            min(max(image_point.x() / max(1, self.image.width() - 1), 0.0), 1.0),
            min(max(image_point.y() / max(1, self.image.height() - 1), 0.0), 1.0),
        )

    def _distance(self, point_a: QPointF, point_b: QPointF) -> float:
        """두 점 사이의 유클리드 거리를 계산합니다."""
        delta_x = point_a.x() - point_b.x()
        delta_y = point_a.y() - point_b.y()
        return float((delta_x * delta_x + delta_y * delta_y) ** 0.5)

    def _project_point_to_segment(self, point: QPoint, start_point: QPoint, end_point: QPoint) -> tuple[QPoint, float]:
        """한 점을 선분에 직교 투영한 최근접 좌표와 거리를 계산합니다."""
        start_x = float(start_point.x())
        start_y = float(start_point.y())
        end_x = float(end_point.x())
        end_y = float(end_point.y())
        point_x = float(point.x())
        point_y = float(point.y())

        delta_x = end_x - start_x
        delta_y = end_y - start_y
        segment_length_squared = delta_x * delta_x + delta_y * delta_y
        if segment_length_squared <= 0.0:
            projected = QPoint(int(round(start_x)), int(round(start_y)))
            return projected, self._distance(QPointF(point), QPointF(projected))

        ratio = ((point_x - start_x) * delta_x + (point_y - start_y) * delta_y) / segment_length_squared
        ratio = min(max(ratio, 0.0), 1.0)
        projected_x = start_x + ratio * delta_x
        projected_y = start_y + ratio * delta_y
        projected = QPoint(int(round(projected_x)), int(round(projected_y)))
        return projected, self._distance(QPointF(point), QPointF(projected))

    def _brush_radius_widget(self) -> int:
        """현재 붓 크기를 위젯 표시 반경으로 환산합니다."""
        if self.image.isNull():
            return max(2, self._brush_radius_image())
        target = self.target_rect()
        scale = target.width() / max(1, self.image.width())
        return max(2, int(round(self._brush_radius_image() * scale)))

    def _brush_radius_image(self) -> int:
        """실제 마스크 연산에 사용하는 이미지 좌표 기준 브러시 반경을 반환합니다."""
        return max(1, int(round(self.brush_size / 2)))

    def _edit_vertex_hit_radius_image(self) -> float:
        """꼭짓점 편집 판정에 사용할 이미지 좌표 기준 히트 반경을 반환합니다."""
        if self.image.isNull():
            return 8.0
        target = self.target_rect()
        scale = target.width() / max(1, self.image.width())
        if scale <= 0:
            return 8.0
        # 폴리곤 내부 클릭과 구분되도록 꼭짓점 히트 범위는 화면 기준 약 10px 정도로 제한합니다.
        return max(4.0, 10.0 / scale)

    def _update_cursor(self, dragging: bool = False) -> None:
        """현재 모드와 줌 상태에 맞는 마우스 커서를 적용합니다."""
        if self.current_mode == "hand" and self.zoom_factor > 1.0 and dragging:
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        elif self.current_mode == "hand" and self.zoom_factor > 1.0:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        elif self.current_mode == "erase":
            self.setCursor(QCursor(Qt.CursorShape.ForbiddenCursor))
        else:
            self.setCursor(
                QCursor(
                    Qt.CursorShape.CrossCursor
                    if self.current_mode in {"draw", "mask", "erase"}
                    else Qt.CursorShape.ArrowCursor
                )
            )
