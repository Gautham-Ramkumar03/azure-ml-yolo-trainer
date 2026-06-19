"""Download trained model weights from a completed Azure ML run.

Pass the run ID printed by submit_job.py (or copied from Azure ML Studio).

Examples:
    python download_model.py --run-id yolo-training_1700000000_abcdef12
    python download_model.py --run-id <id> --all   # download the whole outputs/ tree
"""

import argparse
import os

from azureml.core import Run

from common import get_workspace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="Azure ML run ID to download from")
    p.add_argument("--output-dir", default="./downloaded_models", help="Local destination directory")
    p.add_argument("--all", action="store_true", help="Download every file under outputs/, not just weights")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ws = get_workspace()
    run = Run.get(ws, args.run_id)
    os.makedirs(args.output_dir, exist_ok=True)

    files = run.get_file_names()
    if args.all:
        targets = [f for f in files if f.startswith("outputs/")]
    else:
        targets = [f for f in files if f.startswith("outputs/") and f.endswith(".pt")]

    if not targets:
        print("No matching files found in this run. Files available:")
        for f in files:
            print(f"  {f}")
        return

    print(f"Downloading {len(targets)} file(s) from run {args.run_id}...")
    for remote in targets:
        # Preserve the path under outputs/ in the local destination.
        rel = remote[len("outputs/"):] if remote.startswith("outputs/") else remote
        local_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        run.download_file(remote, local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"  ✅ {local_path} ({size_mb:.1f} MB)")

    print(f"Done. Files saved under {args.output_dir}")


if __name__ == "__main__":
    main()
