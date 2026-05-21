# -*- coding: utf-8 -*-
"""작업 폴더의 worklog.txt 기반 작업 상태를 관리합니다."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


WorkStatus = Literal["v", "a", "n"]
WORKLOG_PATTERN = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s+=\s+\[([van])\]\s*$")


def worklog_path(work_dir: Path) -> Path:
    """작업 폴더 기준 worklog.txt 경로를 반환합니다."""
    return work_dir / "worklog.txt"


def normalize_work_status(value: str) -> WorkStatus:
    """허용된 상태 문자만 유지하고 나머지는 미검토 상태로 보정합니다."""
    return value if value in {"v", "a", "n"} else "n"


def default_work_statuses(image_paths: list[Path]) -> dict[Path, WorkStatus]:
    """새 worklog를 만들 때 사용할 기본 상태 맵을 생성합니다."""
    return {
        image_path: "v" if image_path.with_suffix(".txt").exists() else "n"
        for image_path in image_paths
    }


def save_worklog_statuses(
    work_dir: Path,
    image_paths: list[Path],
    statuses: dict[Path, WorkStatus],
) -> dict[Path, WorkStatus]:
    """이미지 순서에 맞춘 worklog.txt를 다시 기록합니다."""
    normalized_statuses = {
        image_path: normalize_work_status(statuses.get(image_path, "n"))
        for image_path in image_paths
    }
    total_count = len(image_paths)
    lines: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        lines.append(f"[{index}/{total_count}] {image_path.stem} = [{normalized_statuses[image_path]}]")
    worklog_path(work_dir).write_text("\n".join(lines), encoding="utf-8")
    return normalized_statuses


def load_worklog_statuses(work_dir: Path, image_paths: list[Path]) -> dict[Path, WorkStatus]:
    """기존 worklog를 읽고 현재 이미지 목록 기준 상태 맵으로 정규화합니다."""
    status_path = worklog_path(work_dir)
    statuses = default_work_statuses(image_paths)
    if status_path.exists():
        name_to_path = {image_path.stem: image_path for image_path in image_paths}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            match = WORKLOG_PATTERN.match(line.strip())
            if match is None:
                continue
            image_stem = match.group(3)
            image_path = name_to_path.get(image_stem)
            if image_path is None:
                continue
            statuses[image_path] = normalize_work_status(match.group(4))
    return save_worklog_statuses(work_dir, image_paths, statuses)


def work_status_for_image(statuses: dict[Path, WorkStatus], image_path: Path) -> WorkStatus:
    """상태 맵에서 개별 이미지의 작업 상태를 반환합니다."""
    return normalize_work_status(statuses.get(image_path, "n"))


def set_work_status(
    work_dir: Path,
    image_paths: list[Path],
    statuses: dict[Path, WorkStatus],
    image_path: Path,
    status: WorkStatus,
) -> dict[Path, WorkStatus]:
    """특정 이미지의 작업 상태를 변경하고 worklog.txt를 다시 저장합니다."""
    updated_statuses = dict(statuses)
    updated_statuses[image_path] = normalize_work_status(status)
    return save_worklog_statuses(work_dir, image_paths, updated_statuses)
