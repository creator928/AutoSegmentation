# -*- coding: utf-8 -*-
"""작업 폴더의 클래스 파일과 이미지 목록을 관리합니다."""

from __future__ import annotations

from pathlib import Path

from ..constants import IMAGE_EXTENSIONS


def ensure_classes_file(work_dir: Path, class_names: list[str] | None = None) -> list[str]:
    """classes.txt가 없으면 전달받은 클래스 목록으로 생성하고 결과를 반환합니다."""
    classes_path = work_dir / "classes.txt"
    if classes_path.exists():
        return read_classes(work_dir)

    if not class_names:
        raise FileNotFoundError("classes.txt가 없고 생성할 클래스 정보도 없습니다.")

    sanitized = [line.strip() for line in class_names if line.strip()]
    classes_path.write_text("\n".join(sanitized), encoding="utf-8")
    return sanitized


def read_classes(work_dir: Path) -> list[str]:
    """작업 폴더의 classes.txt를 읽어 클래스 목록을 반환합니다."""
    classes_path = work_dir / "classes.txt"
    if not classes_path.exists():
        return []
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_images(work_dir: Path) -> list[Path]:
    """작업 폴더 안의 이미지 파일을 이름순으로 정렬해 반환합니다."""
    return sorted(
        [
            path
            for path in work_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda item: item.name.lower(),
    )
