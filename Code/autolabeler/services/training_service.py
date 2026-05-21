# -*- coding: utf-8 -*-
"""학습 전 검증과 Temp 데이터셋 복제를 담당합니다."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .worklog_service import load_worklog_statuses, work_status_for_image


TRAIN_SPLIT_SEED = 42


@dataclass
class TrainingDatasetInfo:
    """복제된 학습 데이터셋 경로 정보를 모읍니다."""

    temp_dir: Path
    dataset_yaml_path: Path
    result_dir: Path


def result_onnx_path(work_dir: Path) -> Path:
    """작업 폴더 기준 학습 결과 ONNX 파일 경로를 반환합니다."""
    return work_dir / "Temp" / "Result" / "result.onnx"


def result_pt_path(work_dir: Path) -> Path:
    """작업 폴더 기준 학습 결과 PT 파일 경로를 반환합니다."""
    return work_dir / "Temp" / "Result" / "result.pt"


def temp_dataset_image_paths(work_dir: Path) -> list[Path]:
    """Temp 데이터셋에 복제된 이미지 경로를 split 순서대로 반환합니다."""
    image_paths: list[Path] = []
    for split_name in ("train", "val", "test"):
        split_dir = work_dir / "Temp" / "images" / split_name
        if not split_dir.exists():
            continue
        image_paths.extend(sorted(path for path in split_dir.iterdir() if path.is_file()))
    return image_paths


def verify_image_paths(work_dir: Path) -> list[Path]:
    """Temp/Verify 아래에 저장된 검증 결과 이미지를 이름순으로 반환합니다."""
    verify_dir = work_dir / "Temp" / "Verify"
    if not verify_dir.exists():
        return []
    return sorted(path for path in verify_dir.iterdir() if path.is_file())


def training_target_images(image_paths: list[Path], dataset_size: int) -> list[Path]:
    """이름순 이미지 목록 중 학습에 사용할 앞부분만 반환합니다."""
    return image_paths[: max(0, dataset_size)]


def front_labeled_boundary_index(image_paths: list[Path]) -> int:
    """앞에서부터 연속으로 라벨 txt가 존재하는 마지막 이미지 인덱스를 반환합니다."""
    boundary_index = -1
    for index, image_path in enumerate(image_paths):
        if not image_path.with_suffix(".txt").exists():
            break
        boundary_index = index
    return boundary_index


def is_training_ready(work_dir: Path | None, image_paths: list[Path], dataset_size: int) -> bool:
    """앞에서부터 N장의 이미지가 모두 수동 검토 완료되었는지 확인합니다."""
    if work_dir is None or dataset_size <= 0 or len(image_paths) < dataset_size:
        return False
    statuses = load_worklog_statuses(work_dir, image_paths)
    for image_path in training_target_images(image_paths, dataset_size):
        if work_status_for_image(statuses, image_path) != "v":
            return False
        if not image_path.with_suffix(".txt").exists():
            return False
    return True


def recreate_temp_dataset(
    work_dir: Path,
    image_paths: list[Path],
    dataset_size: int,
    class_names: list[str],
) -> TrainingDatasetInfo:
    """Temp 폴더를 덮어쓰며 학습용 복제 데이터셋을 생성합니다."""
    temp_dir = work_dir / "Temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    result_dir = temp_dir / "Result"

    targets = training_target_images(image_paths, dataset_size)
    shuffled = list(targets)
    rng = random.Random(TRAIN_SPLIT_SEED)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = max(1, int(total * 0.8))
    val_count = max(1, int(total * 0.1))
    test_count = total - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1

    split_map = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }

    for split_name in ("train", "val", "test"):
        (temp_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, split_images in split_map.items():
        for image_path in split_images:
            label_path = image_path.with_suffix(".txt")
            shutil.copy2(image_path, temp_dir / "images" / split_name / image_path.name)
            shutil.copy2(label_path, temp_dir / "labels" / split_name / label_path.name)

    dataset_yaml_path = temp_dir / "dataset.yaml"
    yaml_text = "\n".join(
        [
            f"path: {temp_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(class_names)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(class_names)],
        ]
    )
    dataset_yaml_path.write_text(yaml_text, encoding="utf-8")
    result_dir.mkdir(parents=True, exist_ok=True)
    return TrainingDatasetInfo(temp_dir=temp_dir, dataset_yaml_path=dataset_yaml_path, result_dir=result_dir)
