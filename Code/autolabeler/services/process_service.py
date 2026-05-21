# -*- coding: utf-8 -*-
"""외부 Python 프로세스를 안정적으로 실행하기 위한 공통 유틸리티입니다."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from collections.abc import Iterator
from pathlib import Path


def build_clean_python_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """외부 Python이 현재 exe의 Python 경로를 잘못 물려받지 않도록 환경 변수를 정리합니다."""
    process_env = dict(base_env or os.environ)
    process_env.pop("PYTHONHOME", None)
    process_env.pop("PYTHONPATH", None)
    process_env["PYTHONIOENCODING"] = "utf-8"
    process_env["PYTHONUTF8"] = "1"

    conda_env_dir = process_env.get("AUTOSEGMENTATION_CONDA_ENV_DIR", "").strip()
    if conda_env_dir:
        env_path = Path(conda_env_dir)
        # conda activate 없이도 외부 Python이 필요한 DLL을 찾도록 자식 프로세스 PATH만 보정합니다.
        conda_paths = [
            env_path,
            env_path / "Library" / "bin",
            env_path / "Scripts",
        ]
        existing_path = process_env.get("PATH", "")
        process_env["PATH"] = os.pathsep.join(
            [str(path) for path in conda_paths if path.exists()] + [existing_path]
        )
    return process_env


@contextlib.contextmanager
def clean_windows_dll_search_path() -> Iterator[None]:
    """PyInstaller의 DLL 검색 경로가 외부 Python에 상속되지 않도록 잠시 초기화합니다."""
    if os.name != "nt":
        yield
        return

    kernel32 = ctypes.windll.kernel32
    restore_path = str(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else ""
    kernel32.SetDllDirectoryW(None)
    try:
        yield
    finally:
        if restore_path:
            kernel32.SetDllDirectoryW(restore_path)


def hidden_subprocess_kwargs() -> dict[str, object]:
    """Windows GUI 앱에서 자식 프로세스 콘솔 창이 뜨지 않도록 옵션을 반환합니다."""
    if os.name != "nt":
        return {}

    import subprocess

    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup_info.wShowWindow = 0
    return {
        "startupinfo": startup_info,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }
