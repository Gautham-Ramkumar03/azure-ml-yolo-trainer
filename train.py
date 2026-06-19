"""Single-node YOLO training script (runs on the Azure ML compute target).

This script is generic: the model and hyperparameters are passed as arguments
by submit_job.py, so it works with any Ultralytics model and YOLO task
(detection, OBB, segmentation, classification) without edits.

It auto-detects all local GPUs and trains across them, logs diagnostics, and
saves outputs to `outputs/` (which Azure ML captures in the run history).
"""

import argparse
import logging
import os
import sys
import time

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train")


def log_diagnostics() -> None:
    logger.info("PyTorch %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA version: %s | devices: %d", torch.version.cuda, torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            logger.info("  Device %d: %s", i, torch.cuda.get_device_name(i))


def main(args: argparse.Namespace) -> None:
    start = time.time()
    log_diagnostics()

    # Avoid Weights & Biases prompting for an API key on the cluster.
    os.environ.setdefault("WANDB_MODE", "disabled")

    if torch.cuda.is_available():
        device = list(range(torch.cuda.device_count()))  # use every GPU on the node
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    logger.info("Training on device(s): %s", device)

    logger.info("Loading model: %s", args.model)
    model = YOLO(args.model)

    model.train(
        data=os.path.join(args.data_path, "data.yaml"),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        project="outputs",
        name=args.name,
        device=device,
        save=True,
        save_period=args.save_period,
        val=True,
        workers=args.workers,
        verbose=True,
        exist_ok=True,
    )

    logger.info("Training completed in %.2f hours", (time.time() - start) / 3600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to mounted dataset (contains data.yaml)")
    parser.add_argument("--model", default="yolov8m.pt", help="Ultralytics model name or .pt path")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--save_period", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="train_run", help="Run/output folder name under outputs/")
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)
