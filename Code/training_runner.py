# -*- coding: utf-8 -*-
"""외부 Python 환경에서 Ultralytics 세그멘테이션 학습과 ONNX export를 수행합니다."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from itertools import repeat
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """명령행 인자를 해석합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ultralytics-dir", required=True)
    return parser.parse_args()


def patch_ultralytics_label_cache() -> None:
    """Windows 권한 제약 환경에서도 라벨 캐시를 만들 수 있도록 직렬 처리로 우회합니다."""
    from ultralytics.data import dataset as dataset_module

    def serial_cache_labels(self, path: Path = Path("./labels.cache")) -> dict:
        """Ultralytics 기본 ThreadPool 대신 단일 스레드로 라벨 캐시를 생성합니다."""
        cache_payload = {"labels": []}
        nm, nf, ne, nc, messages = 0, 0, 0, 0, []
        description = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be [keypoints, dims]."
            )

        iterator = zip(
            self.im_files,
            self.label_files,
            repeat(self.prefix),
            repeat(self.use_keypoints),
            repeat(len(self.data["names"])),
            repeat(nkpt),
            repeat(ndim),
            repeat(self.single_cls),
        )
        progress = dataset_module.TQDM(iterator, desc=description, total=total)
        for item in progress:
            im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, message = dataset_module.verify_image_label(item)
            nm += nm_f
            nf += nf_f
            ne += ne_f
            nc += nc_f
            if im_file:
                cache_payload["labels"].append(
                    {
                        "im_file": im_file,
                        "shape": shape,
                        "cls": lb[:, 0:1],
                        "bboxes": lb[:, 1:],
                        "segments": segments,
                        "keypoints": keypoint,
                        "normalized": True,
                        "bbox_format": "xywh",
                    }
                )
            if message:
                messages.append(message)
            progress.desc = f"{description} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
        progress.close()

        if messages:
            dataset_module.LOGGER.info("\n".join(messages))
        if nf == 0:
            dataset_module.LOGGER.warning(
                f"{self.prefix}No labels found in {path}. {dataset_module.HELP_URL}"
            )
        cache_payload["hash"] = dataset_module.get_hash(self.label_files + self.im_files)
        cache_payload["results"] = nf, nm, ne, nc, len(self.im_files)
        cache_payload["msgs"] = messages
        dataset_module.save_dataset_cache_file(
            self.prefix,
            path,
            cache_payload,
            dataset_module.DATASET_CACHE_VERSION,
        )
        return cache_payload

    dataset_module.YOLODataset.cache_labels = serial_cache_labels


def patch_ultralytics_onnx_export() -> None:
    """torch 2.11 환경에서 안정적인 legacy ONNX exporter를 사용하도록 강제합니다."""
    import torch
    from ultralytics.engine import exporter as exporter_module
    from ultralytics.utils import export as export_module

    def legacy_export_onnx(
        torch_model: torch.nn.Module,
        im: torch.Tensor,
        onnx_file: str,
        opset: int = 14,
        input_names: list[str] = ["images"],
        output_names: list[str] = ["output0"],
        dynamic: bool | dict = False,
    ) -> None:
        """torch.onnx.export를 legacy 경로(dynamo=False)로 호출합니다."""
        torch.onnx.export(
            torch_model,
            im,
            onnx_file,
            verbose=False,
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic or None,
        )

    export_module.export_onnx = legacy_export_onnx
    exporter_module.export_onnx = legacy_export_onnx


def create_verify_images(result_root: Path, model_path: Path, device: str) -> None:
    """Temp/images/test 기준으로 세그멘테이션 검증 이미지를 Temp/Verify에 미리 생성합니다."""
    import cv2
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    LOGGER.setLevel(logging.ERROR)

    temp_dir = result_root.parent
    test_dir = temp_dir / "images" / "test"
    verify_dir = temp_dir / "Verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    verify_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        return

    verify_model = YOLO(str(model_path))
    test_images = sorted(path for path in test_dir.iterdir() if path.is_file())
    for image_path in test_images:
        results = verify_model.predict(
            source=str(image_path),
            imgsz=640,
            task="segment",
            device=device,
            conf=0.01,
            iou=0.45,
            max_det=300,
            verbose=False,
        )
        if results:
            rendered = results[0].plot(labels=True, conf=True)
            cv2.imwrite(str(verify_dir / image_path.name), rendered)
        else:
            shutil.copy2(image_path, verify_dir / image_path.name)


def main() -> int:
    """Ultralytics 세그멘테이션 학습 후 ONNX를 export합니다."""
    args = parse_args()
    os.environ["YOLO_CONFIG_DIR"] = args.ultralytics_dir

    from ultralytics import YOLO
    patch_ultralytics_label_cache()
    patch_ultralytics_onnx_export()

    model = YOLO(args.model)

    def on_train_epoch_end(trainer) -> None:
        """epoch 단위 진행률을 표준 출력으로 전달합니다."""
        current_epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", args.epochs))
        print(f"AUTO_LABELER_PROGRESS|{current_epoch}|{total_epochs}", flush=True)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    result_root = Path(args.result_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(result_root),
        name=args.project_name,
        exist_ok=True,
    )

    best_path = Path(getattr(model.trainer, "best", ""))
    if not best_path.exists():
        best_path = result_root / args.project_name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError("학습 완료 후 best.pt를 찾지 못했습니다.")

    # 검증과 오토 라벨링은 PT를 사용하므로 best.pt를 결과 폴더의 고정 이름으로 복사합니다.
    target_pt = result_root / "result.pt"
    if best_path.resolve() != target_pt.resolve():
        shutil.copy2(best_path, target_pt)
    else:
        target_pt = best_path

    exported_path = YOLO(str(best_path)).export(format="onnx")
    exported_path = Path(str(exported_path))
    target_onnx = result_root / "result.onnx"
    if exported_path.resolve() != target_onnx.resolve():
        shutil.copy2(exported_path, target_onnx)
        # 내보내기 과정에서 결과 폴더 밖에 생성된 임시 ONNX는 루트 오염을 막기 위해 정리합니다.
        exported_path.unlink(missing_ok=True)
    # 검증 창은 Temp/Verify의 저장 이미지만 표시하므로 학습 직후 테스트셋 기준으로 미리 생성합니다.
    create_verify_images(result_root, target_pt, args.device)
    print(f"최종 ONNX 저장 완료: {target_onnx}", flush=True)
    print(f"최종 PT 저장 완료: {target_pt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
