"""Submit a single-node YOLO training job to Azure ML.

Everything (cluster, environment, dataset, model, hyperparameters) is read from
`config.yaml`. Override the target cluster on the fly with --cluster.

Examples:
    python submit_job.py                  # use default_cluster from config.yaml
    python submit_job.py --cluster big    # use the 'big' cluster alias
    python submit_job.py --no-wait        # submit and return immediately

When the run finishes, the cluster's min_nodes is set back to 0 so it scales
down automatically and you stop paying for idle GPUs.
"""

import argparse

from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig
from azureml.core.runconfig import RunConfiguration

from common import get_workspace, load_config, resolve_cluster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", help="Cluster alias from config.yaml (default: default_cluster)")
    p.add_argument("--model", help="Override training.model from config.yaml")
    p.add_argument("--epochs", type=int, help="Override training.epochs")
    p.add_argument("--no-wait", action="store_true", help="Do not stream logs / wait for completion")
    p.add_argument("--keep-warm", action="store_true", help="Do NOT scale the cluster down after the run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ws = get_workspace()

    cluster = resolve_cluster(config, args.cluster)
    cluster_name = cluster["name"]
    compute_target = ws.compute_targets[cluster_name]
    print(f"Using compute target: {cluster_name}")

    env_cfg = config["environment"]
    env = Environment.get(workspace=ws, name=env_cfg["name"])
    print(f"Using environment: {env.name}, version: {env.version}")

    ds_name = config["dataset"]["name"]
    dataset: Dataset = ws.datasets[ds_name]
    print(f"Using dataset: {ds_name}")

    train_cfg = config["training"]
    model = args.model or train_cfg["model"]
    epochs = args.epochs or train_cfg["epochs"]

    run_config = RunConfiguration()
    run_config.environment = env
    run_config.target = compute_target
    run_config.environment_variables = {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "PYTHONUNBUFFERED": "1",
        "AZUREML_COMPUTE_USE_COMMON_RUNTIME": "true",
    }

    src = ScriptRunConfig(
        source_directory=".",
        script="train.py",
        arguments=[
            "--data_path", dataset.as_mount(),
            "--model", model,
            "--epochs", epochs,
            "--batch", train_cfg["batch"],
            "--imgsz", train_cfg["imgsz"],
            "--patience", train_cfg["patience"],
        ],
        compute_target=compute_target,
        environment=env,
        run_config=run_config,
    )

    experiment = Experiment(ws, train_cfg["experiment"])
    tags = {"model": model, "dataset": ds_name, "cluster": cluster_name}
    run = experiment.submit(config=src, tags=tags)
    print(f"Submitted run ID: {run.id}")
    print(f"Run URL: {run.get_portal_url()}")

    if args.no_wait:
        print("Submitted. Use download_model.py with the run ID above once it completes.")
        return

    try:
        run.wait_for_completion(show_output=True)
        print(f"Final status: {run.get_status()}")
    finally:
        if not args.keep_warm:
            print("Scaling cluster down (min_nodes=0) to stop idle billing...")
            compute_target.update(min_nodes=0, max_nodes=cluster["max_nodes"])
            print("✅ Cluster will scale down automatically when idle.")


if __name__ == "__main__":
    main()
