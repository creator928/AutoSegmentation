# -*- coding: utf-8 -*-
"""백그라운드 학습 워커를 정의합니다."""

from __future__ import annotations

import locale
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .process_service import build_clean_python_env, clean_windows_dll_search_path, hidden_subprocess_kwargs
from .training_service import recreate_temp_dataset


@dataclass
class TrainingRequest:
    """백그라운드 학습에 필요한 입력값입니다."""

    python_command: list[str]
    work_dir: Path
    image_paths: list[Path]
    class_names: list[str]
    dataset_size: int
    epochs: int
    image_size: int
    batch_size: int
    project_name: str
    model_path: Path
    use_gpu: bool
    ultralytics_dir: Path
    runner_script_path: Path


class TrainingWorker(QObject):
    """외부 Python 프로세스로 YOLO 학습을 백그라운드 실행합니다."""

    progress_changed = pyqtSignal(int, int)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, request: TrainingRequest) -> None:
        super().__init__()
        self.request = request

    def _decode_output_line(self, raw_line: bytes) -> str:
        """외부 프로세스 로그를 UTF-8 우선, 실패 시 시스템 로캘로 복구 디코딩합니다."""
        for encoding in ("utf-8", locale.getpreferredencoding(False), "cp949"):
            try:
                return raw_line.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw_line.decode("utf-8", errors="replace").strip()

    def run(self) -> None:
        """Temp 데이터셋 생성과 학습 프로세스 실행을 순차 수행합니다."""
        try:
            dataset_info = recreate_temp_dataset(
                self.request.work_dir,
                self.request.image_paths,
                self.request.dataset_size,
                self.request.class_names,
            )
            command = [
                *self.request.python_command,
                str(self.request.runner_script_path),
                "--model",
                str(self.request.model_path),
                "--data",
                str(dataset_info.dataset_yaml_path),
                "--result-dir",
                str(dataset_info.result_dir),
                "--project-name",
                self.request.project_name,
                "--epochs",
                str(self.request.epochs),
                "--imgsz",
                str(self.request.image_size),
                "--batch",
                str(self.request.batch_size),
                "--device",
                "0" if self.request.use_gpu else "cpu",
                "--ultralytics-dir",
                str(self.request.ultralytics_dir),
            ]
            self.status_changed.emit("Temp 데이터셋 생성 완료")
            # 자식 Python 표준 입출력을 UTF-8로 유도하되, 부모 쪽에서는 추가 복구 디코딩도 수행합니다.
            process_env = build_clean_python_env()
            with clean_windows_dll_search_path():
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=False,
                    env=process_env,
                    **hidden_subprocess_kwargs(),
                )

            if process.stdout is None:
                raise RuntimeError("학습 로그를 읽을 수 없습니다.")

            recent_lines: list[str] = []
            for raw_line in process.stdout:
                line = self._decode_output_line(raw_line)
                if not line:
                    continue
                recent_lines.append(line)
                if len(recent_lines) > 20:
                    recent_lines.pop(0)
                if line.startswith("AUTO_LABELER_PROGRESS|"):
                    _, current_text, total_text = line.split("|", 2)
                    self.progress_changed.emit(int(current_text), int(total_text))
                else:
                    self.status_changed.emit(line)

            return_code = process.wait()
            if return_code != 0:
                error_tail = "\n".join(recent_lines[-10:])
                raise RuntimeError(
                    f"학습 프로세스가 비정상 종료되었습니다. code={return_code}\n{error_tail}"
                )

            self.finished.emit(str(dataset_info.result_dir / "result.onnx"))
        except Exception as exc:
            self.failed.emit(str(exc))
