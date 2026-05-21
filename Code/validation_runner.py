# -*- coding: utf-8 -*-
"""외부 Python 환경에서 ONNX 단일 이미지 세그멘테이션 검증 추론을 수행합니다."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """명령행 인자를 해석합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", required=True, type=int)
    parser.add_argument("--conf", required=True, type=float)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ultralytics-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """단일 이미지의 세그멘테이션 결과를 JSON으로 출력합니다."""
    args = parse_args()
    os.environ["YOLO_CONFIG_DIR"] = args.ultralytics_dir

    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    # 검증 창은 JSON 한 줄만 받도록 Ultralytics 로그를 최대한 억제합니다.
    LOGGER.setLevel(logging.ERROR)

    model = YOLO(args.model)
    results = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        task="segment",
        device=args.device,
        conf=args.conf,
        iou=0.45,
        max_det=300,
        verbose=False,
    )

    detections: list[dict[str, object]] = []
    if results:
        boxes = getattr(results[0], "boxes", None)
        masks = getattr(results[0], "masks", None)
        if (
            boxes is not None
            and masks is not None
            and boxes.xyxy is not None
            and boxes.cls is not None
            and boxes.conf is not None
            and masks.xyn is not None
        ):
            xyxy_list = boxes.xyxy.tolist()
            class_list = boxes.cls.tolist()
            conf_list = boxes.conf.tolist()
            polygon_list = masks.xyn
            for xyxy, class_value, conf_value, polygon in zip(xyxy_list, class_list, conf_list, polygon_list):
                detections.append(
                    {
                        "class_index": int(class_value),
                        "conf": float(conf_value),
                        "xyxy": [float(value) for value in xyxy],
                        "polygon": [[float(x_value), float(y_value)] for x_value, y_value in polygon],
                    }
                )

    print(json.dumps({"detections": detections}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
