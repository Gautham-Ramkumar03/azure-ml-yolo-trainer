"""Submit a multi-node distributed (DDP over MPI) YOLO training job.

Use this when one node isn't enough and you want to train across several nodes
of a cluster. Node count and processes-per-node come from the `distributed`
section of config.yaml; everything else is shared with submit_job.py.

Examples:
    python submit_distributed.py --cluster big
    python submit_distributed.py --cluster big --nodes 4
"""

import argparse

from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration, RunConfiguration

from common import get_workspace, load_config, resolve_cluster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cluster", help="Cluster alias from config.yaml (default: default_cluster)")
    p.add_argument("--nodes", type=int, help="Override distributed.node_count")
    p.add_argument("--keep-warm", action="store_true", help="Do NOT scale the cluster down after the run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ws = get_workspace()

    cluster = resolve_cluster(config, args.cluster)
    cluster_name = cluster["name"]
    compute_target = ws.compute_targets[cluster_name]

    dist_cfg = config["distributed"]
    node_count = args.nodes or dist_cfg["node_count"]
    procs_per_node = dist_cfg["process_count_per_node"]
    print(f"Distributed run: {node_count} node(s) x {procs_per_node} process(es) on {cluster_name}")

    env = Environment.get(workspace=ws, name=config["environment"]["name"])
    dataset: Dataset = ws.datasets[config["dataset"]["name"]]
    train_cfg = config["training"]

    run_config = RunConfiguration()
    run_config.environment = env
    run_config.target = compute_target
    run_config.node_count = node_count
    run_config.mpi = MpiConfiguration(process_count_per_node=procs_per_node, node_count=node_count)
    run_config.environment_variables = {"PYTHONUNBUFFERED": "1", "NCCL_DEBUG": "INFO"}

    src = ScriptRunConfig(
        source_directory=".",
        script="train_distributed.py",
        arguments=[
            "--data_path", dataset.as_mount(),
            "--model", train_cfg["model"],
            "--epochs", train_cfg["epochs"],
            "--batch", train_cfg["batch"],
            "--imgsz", train_cfg["imgsz"],
            "--patience", train_cfg["patience"],
        ],
        compute_target=compute_target,
        environment=env,
        run_config=run_config,
        distributed_job_config=MpiConfiguration(
            process_count_per_node=procs_per_node, node_count=node_count
        ),
    )

    experiment = Experiment(ws, train_cfg["experiment"] + "-distributed")
    run = experiment.submit(config=src, tags={"mode": "distributed", "nodes": str(node_count)})
    print(f"Submitted run ID: {run.id}")
    print(f"Run URL: {run.get_portal_url()}")

    try:
        run.wait_for_completion(show_output=True)
        print(f"Final status: {run.get_status()}")
    finally:
        if not args.keep_warm:
            print("Scaling cluster down (min_nodes=0)...")
            compute_target.update(min_nodes=0, max_nodes=cluster["max_nodes"])
            print("✅ Cluster will scale down automatically when idle.")


if __name__ == "__main__":
    main()
