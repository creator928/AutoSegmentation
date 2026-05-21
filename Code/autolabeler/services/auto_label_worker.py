# -*- coding: utf-8 -*-
"""학습 결과 PT를 이용한 백그라운드 오토 라벨 워커를 정의합니다."""

from __future__ import annotations

import locale
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from .process_service import build_clean_python_env, clean_windows_dll_search_path, hidden_subprocess_kwargs


@dataclass
class AutoLabelRequest:
    """오토 라벨 실행에 필요한 입력값을 모읍니다."""

    python_command: list[str]
    work_dir: Path
    image_paths: list[Path]
    model_path: Path
    image_size: int
    conf_threshold: float
    use_gpu: bool
    ultralytics_dir: Path
    runner_script_path: Path
    stop_index_path: Path
    worklog_path: Path


class AutoLabelWorker(QObject):
    """외부 Python 프로세스로 PT 기반 오토 라벨을 백그라운드 실행합니다."""

    progress_changed = pyqtSignal(int, int)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(int, int)
    failed = pyqtSignal(str)

    def __init__(self, request: AutoLabelRequest) -> None:
        super().__init__()
        self.request = request

    def _decode_output_line(self, raw_line: bytes) -> str:
        """외부 프로세스 로그를 UTF-8 우선으로 복구 디코딩합니다."""
        for encoding in ("utf-8", locale.getpreferredencoding(False), "cp949"):
            try:
                return raw_line.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw_line.decode("utf-8", errors="replace").strip()

    def run(self) -> None:
        """작업 폴더의 전체 이미지 목록을 기반으로 오토 라벨을 실행합니다."""
        try:
            manifest_path = self.request.work_dir / "Temp" / "Result" / "auto_label_targets.txt"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                "\n".join(str(path) for path in self.request.image_paths),
                encoding="utf-8",
            )

            command = [
                *self.request.python_command,
                str(self.request.runner_script_path),
                "--model",
                str(self.request.model_path),
                "--manifest",
                str(manifest_path),
                "--imgsz",
                str(self.request.image_size),
                "--conf",
                str(self.request.conf_threshold),
                "--device",
                "0" if self.request.use_gpu else "cpu",
                "--ultralytics-dir",
                str(self.request.ultralytics_dir),
                "--stop-index-path",
                str(self.request.stop_index_path),
                "--worklog-path",
                str(self.request.worklog_path),
            ]
            self.status_changed.emit("오토 라벨 실행 준비 완료")
            # 자식 Python 출력 인코딩을 UTF-8로 고정해 한글 로그가 깨지지 않도록 합니다.
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
                raise RuntimeError("오토 라벨 로그를 읽을 수 없습니다.")

            processed_count = 0
            labeled_count = 0
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
                elif line.startswith("AUTO_LABELER_DONE|"):
                    _, processed_text, labeled_text = line.split("|", 2)
                    processed_count = int(processed_text)
                    labeled_count = int(labeled_text)
                else:
                    self.status_changed.emit(line)

            return_code = process.wait()
            if return_code != 0:
                error_tail = "\n".join(recent_lines[-10:])
                raise RuntimeError(
                    f"오토 라벨 프로세스가 비정상 종료되었습니다. code={return_code}\n{error_tail}"
                )

            self.finished.emit(processed_count, labeled_count)
        except Exception as exc:
            self.failed.emit(str(exc))
