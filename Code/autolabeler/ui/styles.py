# -*- coding: utf-8 -*-
"""회색조 중심의 라이트/다크 테마 스타일시트를 제공합니다."""

from __future__ import annotations


def build_stylesheet(theme_mode: str) -> str:
    """테마 모드에 맞는 전체 애플리케이션 스타일시트를 반환합니다."""
    if theme_mode == "dark":
        return """
        QWidget {
            background-color: #1f1f1f;
            color: #f2f2f2;
            font-family: 'Malgun Gothic';
            font-size: 9pt;
        }
        QMainWindow, QFrame, QListWidget, QPlainTextEdit, QLineEdit, QComboBox {
            background-color: #2a2a2a;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 2px;
        }
        QPushButton {
            background-color: #3a3a3a;
            border: 1px solid #4a4a4a;
            border-radius: 4px;
            padding: 2px;
            margin: 2px;
        }
        QPushButton:hover {
            background-color: #4a4a4a;
        }
        /* 다크 모드에서도 라디오 버튼 텍스트와 선택 상태가 배경과 강한 대비를 갖도록 명시합니다. */
        QRadioButton {
            background-color: #2f2f2f;
            color: #f7f7f7;
            border: 1px solid #4b4b4b;
            border-radius: 6px;
            padding: 6px 8px;
            spacing: 8px;
        }
        QRadioButton:checked {
            background-color: #23466f;
            border: 1px solid #5ea8ff;
            color: #ffffff;
        }
        QRadioButton:hover {
            background-color: #393939;
        }
        QRadioButton::indicator {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 2px solid #d0d0d0;
            background-color: #161616;
        }
        QRadioButton::indicator:checked {
            border: 2px solid #f9f9f9;
            background-color: #4da3ff;
        }
        QRadioButton::indicator:unchecked {
            border: 2px solid #b5b5b5;
            background-color: #1b1b1b;
        }
        QScrollBar:vertical {
            background-color: #2a2a2a;
            width: 10px;
            margin: 0px;
            border: none;
        }
        QScrollBar::handle:vertical {
            background-color: #5a5a5a;
            min-height: 24px;
            border-radius: 4px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
            border: none;
            background: transparent;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: transparent;
        }
        QScrollBar:horizontal {
            background-color: #2a2a2a;
            height: 10px;
            margin: 0px;
            border: none;
        }
        QScrollBar::handle:horizontal {
            background-color: #5a5a5a;
            min-width: 24px;
            border-radius: 4px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
            border: none;
            background: transparent;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: transparent;
        }
        QListWidget::item:selected {
            background-color: #595959;
        }
        """

    return """
    QWidget {
        background-color: #f4f4f4;
        color: #202020;
        font-family: 'Malgun Gothic';
        font-size: 9pt;
    }
    QMainWindow, QFrame, QListWidget, QPlainTextEdit, QLineEdit, QComboBox {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 2px;
    }
    QPushButton {
        background-color: #e5e5e5;
        border: 1px solid #c8c8c8;
        border-radius: 4px;
        padding: 2px;
        margin: 2px;
    }
    QPushButton:hover {
        background-color: #d8d8d8;
    }
    /* 라이트 모드에서는 체크된 라디오 버튼이 흰 배경에서도 확실히 드러나도록 파란 계열 강조색을 사용합니다. */
    QRadioButton {
        background-color: #ffffff;
        color: #141414;
        border: 1px solid #c6c6c6;
        border-radius: 6px;
        padding: 6px 8px;
        spacing: 8px;
    }
    QRadioButton:checked {
        background-color: #dcebff;
        border: 1px solid #6aa8ff;
        color: #0f2747;
    }
    QRadioButton:hover {
        background-color: #f2f6ff;
    }
    QRadioButton::indicator {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 2px solid #6a6a6a;
        background-color: #ffffff;
    }
    QRadioButton::indicator:checked {
        border: 2px solid #004ea8;
        background-color: #2f80ed;
    }
    QRadioButton::indicator:unchecked {
        border: 2px solid #7c7c7c;
        background-color: #ffffff;
    }
    QScrollBar:vertical {
        background-color: #f0f0f0;
        width: 10px;
        margin: 0px;
        border: none;
    }
    QScrollBar::handle:vertical {
        background-color: #b8b8b8;
        min-height: 24px;
        border-radius: 4px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
        border: none;
        background: transparent;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: transparent;
    }
    QScrollBar:horizontal {
        background-color: #f0f0f0;
        height: 10px;
        margin: 0px;
        border: none;
    }
    QScrollBar::handle:horizontal {
        background-color: #b8b8b8;
        min-width: 24px;
        border-radius: 4px;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
        border: none;
        background: transparent;
    }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: transparent;
    }
    QListWidget::item:selected {
        background-color: #cfcfcf;
    }
    """
