"""Register the training conda environment with Azure ML.

Reads the environment name and conda file from `config.yaml`. Run this once
(and again whenever you change conda.yml) before submitting training jobs.
"""

from azureml.core import Environment

from common import get_workspace, load_config


def main() -> None:
    config = load_config()
    ws = get_workspace()

    env_cfg = config["environment"]
    env = Environment.from_conda_specification(
        name=env_cfg["name"],
        file_path=env_cfg["conda_file"],
    )
    env.register(workspace=ws)
    print(f"✅ Registered environment '{env_cfg['name']}' from {env_cfg['conda_file']}")


if __name__ == "__main__":
    main()
