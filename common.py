"""Shared helpers for the Azure ML YOLO training pipeline.

Every control-plane script reads its settings from `config.yaml` (workspace
clusters, dataset, model, hyperparameters) so that you never have to edit code
to point the pipeline at your own resources.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from azureml.core import Workspace

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load and return the pipeline configuration from `config.yaml`."""
    path = Path(path)
    if not path.exists():
        sys.exit(
            f"❌ {path.name} not found.\n"
            "   Copy the template first:  cp config.example.yaml config.yaml\n"
            "   then edit it with your cluster name(s), dataset and model."
        )
    with path.open() as f:
        return yaml.safe_load(f)


def get_workspace() -> Workspace:
    """Connect to the Azure ML workspace defined in `config.json`."""
    try:
        return Workspace.from_config()
    except Exception as exc:  # noqa: BLE001 - surface a friendly hint
        sys.exit(
            f"❌ Could not connect to the Azure ML workspace: {exc}\n"
            "   Make sure config.json exists (cp config.json.template config.json)\n"
            "   and that you have run `az login`."
        )


def resolve_cluster(config: dict, cluster_alias: str | None = None) -> dict:
    """Return the cluster entry for `cluster_alias` (or the default).

    The returned dict has at least `name` (the Azure ML compute target name)
    and `max_nodes`.
    """
    clusters = config.get("clusters") or {}
    alias = cluster_alias or config.get("default_cluster", "default")
    if alias not in clusters:
        available = ", ".join(clusters) or "(none defined)"
        sys.exit(
            f"❌ Cluster alias '{alias}' is not defined in config.yaml.\n"
            f"   Available aliases: {available}"
        )
    cluster = clusters[alias]
    cluster.setdefault("max_nodes", 1)
    return cluster
