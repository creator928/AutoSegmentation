# -*- coding: utf-8 -*-
"""학습용 Python 런타임과 하드웨어 사용 가능 여부를 판별합니다."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .process_service import build_clean_python_env, clean_windows_dll_search_path, hidden_subprocess_kwargs


@dataclass
class HardwareStatus:
    """학습 장치 선택에 필요한 하드웨어 상태를 담습니다."""

    python_command: list[str]
    gpu_available: bool
    gpu_name: str
    cpu_name: str
    cuda_runtime_available: bool


def resolve_python_command() -> list[str]:
    """외부 학습 실행에 사용할 Python 명령을 결정합니다."""
    candidates: list[list[str]] = []
    configured_python = os.environ.get("AUTOSEGMENTATION_PYTHON_EXE", "").strip()
    if configured_python and Path(configured_python).exists():
        candidates.append([configured_python])
    if shutil.which("py"):
        candidates.extend((["py", "-3.10"], ["py", "-3"]))
    if shutil.which("python"):
        candidates.append(["python"])
    if Path(sys.executable).name.lower().startswith("python"):
        candidates.append([sys.executable])

    probe_script = (
        "import socket\n"
        "import importlib.metadata\n"
        "import torch\n"
        "import ultralytics\n"
        "import cv2\n"
    )
    for candidate in candidates:
        try:
            with clean_windows_dll_search_path():
                completed = subprocess.run(
                    [*candidate, "-c", probe_script],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=build_clean_python_env(),
                    **hidden_subprocess_kwargs(),
                )
            if completed.returncode == 0:
                return candidate
        except Exception:
            continue
    return []


def detect_hardware() -> HardwareStatus:
    """실제 GPU 장치 존재 여부와 CPU 이름을 확인합니다."""
    python_command = resolve_python_command()
    cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "CPU"

    gpu_name = ""
    gpu_available = False
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            **hidden_subprocess_kwargs(),
        )
        first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
        if first_line:
            gpu_available = True
            gpu_name = first_line
    except Exception:
        gpu_available = False
        gpu_name = ""

    if not python_command:
        return HardwareStatus([], gpu_available, gpu_name, cpu_name, False)

    script = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " gpu=bool(torch.cuda.is_available())\n"
        " gpu_name=torch.cuda.get_device_name(0) if gpu else ''\n"
        " print(json.dumps({'gpu': gpu, 'gpu_name': gpu_name}, ensure_ascii=False))\n"
        "except Exception:\n"
        " print(json.dumps({'gpu': False, 'gpu_name': ''}, ensure_ascii=False))\n"
    )
    try:
        with clean_windows_dll_search_path():
            completed = subprocess.run(
                [*python_command, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                env=build_clean_python_env(),
                **hidden_subprocess_kwargs(),
            )
        payload = json.loads(completed.stdout.strip())
        cuda_runtime_available = bool(payload.get("gpu", False))
        if not gpu_available and bool(payload.get("gpu", False)):
            gpu_available = True
        if not gpu_name:
            gpu_name = str(payload.get("gpu_name", ""))
        return HardwareStatus(
            python_command=python_command,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            cpu_name=cpu_name,
            cuda_runtime_available=cuda_runtime_available,
        )
    except Exception:
        return HardwareStatus(
            python_command=python_command,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            cpu_name=cpu_name,
            cuda_runtime_available=False,
        )
