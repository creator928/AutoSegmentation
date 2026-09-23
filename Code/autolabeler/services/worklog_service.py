# -*- coding: utf-8 -*-
"""작업 폴더의 worklog.txt 기반 작업 상태를 관리합니다."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


WorkStatus = Literal["v", "a", "n"]

# 프로세스 간 상태 갱신과 라벨 저장을 직렬화합니다.
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from functools import wraps

_worklog_mutex = threading.RLock()
_worklog_local = threading.local()


@contextmanager
def worklog_lock(work_dir: Path):
    """GUI와 자동 추론의 worklog 갱신이 서로 덮어쓰지 않도록 잠급니다."""
    with _worklog_mutex:
        if getattr(_worklog_local, "depth", 0):
            _worklog_local.depth += 1
            try:
                yield
            finally:
                _worklog_local.depth -= 1
            return
        import msvcrt
        with (work_dir / ".worklog.lock").open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            _worklog_local.depth = 1
            try:
                yield
            finally:
                _worklog_local.depth = 0
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def locked_worklog(function):
    """work_dir를 받는 상태 함수에 동일한 잠금을 적용합니다."""
    @wraps(function)
    def wrapped(work_dir, *args, **kwargs):
        with worklog_lock(work_dir):
            return function(work_dir, *args, **kwargs)
    return wrapped


def atomic_write_text(path: Path, text: str) -> None:
    """완성된 임시 파일로 교체하여 빈 파일이 관찰되지 않도록 합니다."""
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_explicit_statuses(work_dir: Path) -> dict[str, WorkStatus]:
    """누락·손상·중복 상태는 자동 처리 허가로 해석하지 않습니다."""
    statuses = {}
    for line in worklog_path(work_dir).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        match = WORKLOG_PATTERN.fullmatch(line.strip())
        if match is None or match.group(3) in statuses:
            raise ValueError("worklog 형식 오류 또는 중복 항목: 자동 처리를 중단합니다.")
        statuses[match.group(3)] = match.group(4)
    return statuses


@locked_worklog
def auto_label_allowed(work_dir: Path, image_path: Path) -> bool:
    """명시적인 a/n 상태만 자동 추론하도록 허용합니다."""
    return read_explicit_statuses(work_dir).get(image_path.stem) in {"a", "n"}


@locked_worklog
def commit_auto_label(work_dir: Path, image_paths: list[Path], image_path: Path,
                      text: str, backup_dir: Path, previous: bytes | None) -> bool:
    """저장 직전 상태·수동 편집을 재검사하고 백업 후 결과를 저장합니다."""
    explicit = read_explicit_statuses(work_dir)
    if explicit.get(image_path.stem) not in {"a", "n"}:
        return False
    label_path = image_path.with_suffix(".txt")
    current = label_path.read_bytes() if label_path.exists() else None
    if current != previous:
        return False
    backup_dir.mkdir(parents=True, exist_ok=True)
    if current is not None:
        shutil.copy2(label_path, backup_dir / label_path.name)
    atomic_write_text(label_path, text)
    # 현재 worklog 전체를 그대로 유지하며 처리한 한 항목만 a로 변경합니다.
    log_path = worklog_path(work_dir)
    lines = log_path.read_text(encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        match = WORKLOG_PATTERN.fullmatch(line.strip())
        if match and match.group(3) == image_path.stem:
            lines[index] = line[:line.rfind("[")] + "[a]"
    atomic_write_text(log_path, "\n".join(lines))
    return True


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


@locked_worklog
def save_worklog_statuses(
    work_dir: Path,
    image_paths: list[Path],
    statuses: dict[Path, WorkStatus],
    allow_verified_change: Path | None = None,
) -> dict[Path, WorkStatus]:
    """이미지 순서에 맞춘 worklog.txt를 다시 기록합니다."""
    normalized_statuses = {
        image_path: normalize_work_status(statuses.get(image_path, "n"))
        for image_path in image_paths
    }
    # 오래된 GUI 상태가 이미 검토 완료된 항목을 되돌리지 않도록 합니다.
    if worklog_path(work_dir).exists():
        explicit = read_explicit_statuses(work_dir)
        for image_path in image_paths:
            if explicit.get(image_path.stem) == "v" and image_path != allow_verified_change:
                normalized_statuses[image_path] = "v"
    total_count = len(image_paths)
    lines: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        lines.append(f"[{index}/{total_count}] {image_path.stem} = [{normalized_statuses[image_path]}]")
    atomic_write_text(worklog_path(work_dir), "\n".join(lines))
    return normalized_statuses


@locked_worklog
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


@locked_worklog
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
    return save_worklog_statuses(work_dir, image_paths, updated_statuses, allow_verified_change=image_path)
