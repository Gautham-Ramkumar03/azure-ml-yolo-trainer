"""Inspect and control your Azure ML compute clusters.

Accepts a cluster *alias* from config.yaml (preferred) or a raw compute target
name. Handy for keeping costs down — scale clusters to zero when idle and warm
them up right before a big job.

Examples:
    python manage_cluster.py default status
    python manage_cluster.py big prepare --nodes 4
    python manage_cluster.py big scale-down
    python manage_cluster.py list
"""

import argparse
import time

from common import get_workspace, load_config


def cluster_name_from_alias(config: dict, alias_or_name: str) -> str:
    """Map a config alias to its compute target name, or pass the value through."""
    clusters = config.get("clusters") or {}
    if alias_or_name in clusters:
        return clusters[alias_or_name]["name"]
    return alias_or_name  # treat as a raw compute target name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cluster", help="Cluster alias from config.yaml or a raw compute target name")
    parser.add_argument(
        "action",
        choices=["status", "scale-down", "prepare", "list"],
        help="Action to perform ('list' shows clusters from config.yaml)",
    )
    parser.add_argument("--nodes", type=int, default=2, help="Node count for 'prepare'")
    parser.add_argument("--max-nodes", type=int, default=4, help="Max nodes for 'scale-down'")
    args = parser.parse_args()

    config = load_config()

    if args.action == "list":
        print("Clusters defined in config.yaml:")
        for alias, c in (config.get("clusters") or {}).items():
            print(f"  {alias:12s} -> {c['name']} (max_nodes={c.get('max_nodes', '?')})")
        return

    ws = get_workspace()
    name = cluster_name_from_alias(config, args.cluster)
    compute_target = ws.compute_targets[name]
    print(f"Cluster: {name}")

    if args.action == "status":
        status = compute_target.get_status()
        print(f"  State: {status.provisioning_state}")
        print(f"  Current nodes: {status.current_node_count}")
        print(f"  Min/Max nodes: {status.scale_settings.minimum_node_count}/{status.scale_settings.maximum_node_count}")

    elif args.action == "scale-down":
        print("  Setting min_nodes=0 (scales down when idle)...")
        compute_target.update(min_nodes=0, max_nodes=args.max_nodes)
        print("  ✅ Done.")

    elif args.action == "prepare":
        print(f"  Warming up to {args.nodes} nodes...")
        compute_target.update(min_nodes=args.nodes, max_nodes=args.nodes)
        compute_target.wait_for_completion(show_output=True)
        print("  ✅ Cluster ready.")


if __name__ == "__main__":
    main()
