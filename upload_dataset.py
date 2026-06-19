"""Upload a local YOLO-format dataset to Azure ML and register it.

Source folder and dataset name come from `config.yaml` (the `dataset` section).
The folder should contain your images/labels and a `data.yaml` at its root.

Tip: keep large datasets OUTSIDE this project directory (or delete them after
upload) so they are not swept into the job snapshot on submit.
"""

from azureml.core import Dataset

from common import get_workspace, load_config


def main() -> None:
    config = load_config()
    ws = get_workspace()

    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    local_dir = ds_cfg["local_dir"]
    datastore = ws.get_default_datastore()

    print(f"Uploading '{local_dir}' -> datastore path 'datasets/{name}' ...")
    dataset = Dataset.File.upload_directory(
        src_dir=local_dir,
        target=(datastore, f"datasets/{name}"),
        overwrite=True,
    )

    dataset.register(
        workspace=ws,
        name=name,
        description=f"Uploaded by upload_dataset.py from {local_dir}",
        create_new_version=True,
    )
    print(f"✅ Dataset '{name}' uploaded and registered.")


if __name__ == "__main__":
    main()
