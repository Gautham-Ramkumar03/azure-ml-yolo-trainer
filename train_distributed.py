"""Multi-node distributed YOLO training (DDP over NCCL/MPI).

Runs on the Azure ML compute target when launched via submit_distributed.py.
The distributed setup is resilient to the various rank/master env vars Azure ML
and OpenMPI expose. Model and hyperparameters are passed as arguments, so this
script is generic across Ultralytics models and YOLO tasks.
"""

import argparse
import datetime
import logging
import os
import socket
import sys

import torch
import torch.distributed as dist
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_distributed")


def setup_distributed():
    """Initialise the process group from whichever env vars are available."""
    # Prefer PyTorch vars; fall back to OpenMPI vars set by the AML MPI launcher.
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))

    # Resolve the master address from the several names Azure ML may use.
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        for var in ("AZ_BATCH_MASTER_NODE", "AZUREML_COMPUTE_MASTER_NODE"):
            if var in os.environ:
                raw = os.environ[var]
                if ":" in raw:
                    master_addr, port = raw.split(":", 1)
                    os.environ.setdefault("MASTER_PORT", port)
                else:
                    master_addr = raw
                break
    os.environ["MASTER_ADDR"] = master_addr or "localhost"
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(local_rank))

    logger.info(
        "host=%s rank=%d world_size=%d local_rank=%d master=%s:%s",
        socket.gethostname(), rank, world_size, local_rank,
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"],
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        timeout=datetime.timedelta(hours=2),
        init_method="env://",
    )
    return dist.get_rank(), dist.get_world_size(), device


def main(args: argparse.Namespace) -> None:
    rank, world_size, device = setup_distributed()
    os.environ.setdefault("WANDB_MODE", "disabled")

    try:
        logger.info("Rank %d loading model %s", rank, args.model)
        model = YOLO(args.model)

        model.train(
            data=os.path.join(args.data_path, "data.yaml"),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            project="outputs",
            name="distributed_run",
            device=device,
            verbose=(rank == 0),
            workers=4,
            exist_ok=True,
        )
        logger.info("Rank %d: training completed", rank)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    try:
        main(args)
    except Exception:
        logger.exception("Distributed training failed")
        sys.exit(1)
