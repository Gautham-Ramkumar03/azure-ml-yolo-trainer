<div align="center">

# 🚀 Azure ML YOLO Training Pipeline

**Train YOLO models on Azure ML GPU clusters without touching a single line of code.**

Fill in a config file with your cluster name, dataset, and model — then run one command.

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-0078D4.svg?logo=microsoftazure&logoColor=white)](https://learn.microsoft.com/azure/machine-learning/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-purple.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Why this exists

Getting a YOLO model training on Azure Machine Learning is *annoying*. The model code is
the easy part — the pain is wiring up the workspace, registering an environment,
uploading and registering datasets, picking a compute cluster, configuring distributed
training, and remembering to **scale the cluster back down so you stop paying for idle
GPUs**. Most example repos hard-code all of this, so adapting them means editing five
different scripts.

This pipeline moves **all** of that into a single `config.yaml`. You declare your
cluster(s), dataset, model, and hyperparameters once, and every script reads from it. No
code edits to point it at your own resources.

```bash
# The entire workflow, once configured:
python register_env.py          # register the training environment
python upload_dataset.py        # upload + register your dataset
python submit_job.py            # train (auto-scales the cluster down when done)
python download_model.py --run-id <id>
```

## ✨ Features

- 🧩 **Config-driven** — declare clusters, dataset, model & hyperparameters in `config.yaml`; never edit code.
- 🖧 **Multi-cluster** — define any number of clusters and target any of them with `--cluster <alias>`.
- 🧠 **Any YOLO model/task** — detection, OBB, segmentation, classification (YOLOv8, YOLO11, or your own `.pt`).
- ⚡ **Single- and multi-node** — one-GPU, multi-GPU, or multi-node distributed (DDP over MPI).
- 💸 **Cost-safe by default** — clusters auto scale to zero after every job; `manage_cluster.py` for manual control.
- 🔌 **Roboflow integration** — pull datasets straight from Roboflow.
- 🔒 **No secrets in code** — credentials live in gitignored `config.json` / `.env`.

## 🏗️ Architecture

```
        YOUR MACHINE (control plane)                 AZURE ML (compute plane)
   ┌──────────────────────────────────┐         ┌──────────────────────────────┐
   │  config.yaml  (clusters/dataset/ │         │                              │
   │               model/hyperparams) │         │   GPU Cluster(s)             │
   │  config.json  (AML workspace)    │         │   ┌────────┐   ┌────────┐    │
   │  .env         (Roboflow key)     │         │   │ node 0 │…  │ node N │    │
   └──────────────┬───────────────────┘         │   └────────┘   └────────┘    │
                  │                              │        ▲                     │
   register_env.py│  ── registers env ──────────┼────────┘                     │
   upload_dataset.py ─ uploads dataset ─────────┼──▶  Datastore                │
   submit_job.py / submit_distributed.py ───────┼──▶  runs train.py /          │
   manage_cluster.py ─ scale up/down ───────────┼──▶  train_distributed.py     │
   download_model.py ◀─ pulls weights ──────────┼──   outputs/ (best.pt)       │
   └──────────────────────────────────┘         └──────────────────────────────┘
```

## 📂 Repository structure

```text
.
├── config.example.yaml      # ← copy to config.yaml; the ONE file you edit
├── config.json.template     # ← copy to config.json; your AML workspace details
├── .env.example             # ← copy to .env; Roboflow credentials (optional)
├── conda.yml                # training environment that runs on the cluster
├── environment.lock.yml     # fully pinned local env (exact reproduction, optional)
├── requirements.txt         # minimal local dependencies
├── common.py                # shared config/workspace/cluster helpers
│
├── register_env.py          # register conda.yml as an AML environment
├── upload_dataset.py        # upload + register a local dataset
├── roboflow_dataset.py      # download a dataset from Roboflow
│
├── submit_job.py            # submit single-node training  (+ auto scale-down)
├── submit_distributed.py    # submit multi-node DDP training (+ auto scale-down)
├── train.py                 # single-node training script (runs on the cluster)
├── train_distributed.py     # distributed training script (runs on the cluster)
│
├── manage_cluster.py        # status / scale-up / scale-down any cluster
├── download_model.py        # download trained weights from a finished run
└── test/
    ├── test_workspace.py    # verify AML workspace connection
    ├── test_local_gpu.py    # verify local CUDA / GPU
    └── test_roboflow.py     # verify Roboflow credentials
```

## ✅ Prerequisites

- An **Azure ML workspace** with at least one **GPU compute cluster** already created.
- A **YOLO-format dataset** (images + labels + a `data.yaml`), e.g. exported from Roboflow.
- Local machine with **Python 3.9+** and the **[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)**.

## ⚡ Quickstart

### 1. Install local dependencies

```bash
pip install -r requirements.txt
```

> Want a byte-for-byte reproduction of the development environment?
> `conda env create -f environment.lock.yml`

### 2. Connect to your Azure ML workspace

```bash
cp config.json.template config.json   # then fill in your workspace details
az login
python test/test_workspace.py         # should print your workspace name
```

`config.json` (find it in Azure ML Studio → workspace → **Download config.json**):

```json
{
    "subscription_id": "<YOUR_AZURE_SUBSCRIPTION_ID>",
    "resource_group": "<YOUR_RESOURCE_GROUP>",
    "workspace_name": "<YOUR_AML_WORKSPACE_NAME>"
}
```

### 3. Configure the pipeline

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` — at minimum set your **cluster name**, **dataset name**, and **model**:

```yaml
clusters:
  default:
    name: gpu-cluster        # ← your compute target name in Azure ML Studio
    max_nodes: 4
default_cluster: default

dataset:
  name: my-dataset
  local_dir: ./my-dataset

training:
  model: yolov8m.pt          # any Ultralytics model or a path to your own .pt
  epochs: 150
  batch: 32
  imgsz: 640
```

### 4. Run the pipeline

```bash
python register_env.py                       # one-time (re-run when conda.yml changes)
python upload_dataset.py                      # upload + register your dataset
python submit_job.py                          # train on the default cluster
python download_model.py --run-id <run-id>    # fetch weights (run id printed on submit)
```

That's it. 🎉

## 🔄 Flow of operation

| # | Step | Command | What happens |
|---|------|---------|--------------|
| 1 | **Get data** | `python roboflow_dataset.py` | (Optional) download a dataset from Roboflow into a local folder. |
| 2 | **Register env** | `python register_env.py` | Builds & registers `conda.yml` as a reusable Azure ML environment. |
| 3 | **Upload data** | `python upload_dataset.py` | Uploads `dataset.local_dir` to the datastore and registers it as `dataset.name`. |
| 4 | **Submit** | `python submit_job.py` | Mounts the dataset, launches `train.py` on the chosen cluster, streams logs. |
| 5 | **Auto scale-down** | *(automatic)* | When the run ends, the cluster's `min_nodes` is reset to 0 → no idle GPU billing. |
| 6 | **Download** | `python download_model.py --run-id <id>` | Pulls `best.pt` / weights from the run's `outputs/`. |

## 🖧 Working with multiple clusters

Declare as many clusters as you like in `config.yaml`:

```yaml
clusters:
  cheap:   { name: gpu-cluster-t4,   max_nodes: 2 }
  default: { name: gpu-cluster,      max_nodes: 4 }
  big:     { name: gpu-cluster-a100, max_nodes: 8 }
default_cluster: default
```

Then target any of them by alias:

```bash
python manage_cluster.py list                 # show configured clusters
python submit_job.py --cluster cheap          # quick experiment on the cheap cluster
python submit_job.py --cluster big            # full run on the big cluster
python manage_cluster.py big status           # inspect node count / state
python manage_cluster.py big prepare --nodes 4   # warm up before a job
python manage_cluster.py big scale-down       # send it back to zero
```

## ⚙️ Distributed (multi-node) training

For large datasets, train across multiple nodes with DDP over MPI:

```yaml
distributed:
  node_count: 2
  process_count_per_node: 4   # GPUs per node
```

```bash
python submit_distributed.py --cluster big
python submit_distributed.py --cluster big --nodes 4   # override node count
```

`train_distributed.py` handles rank/master discovery across Azure ML's MPI launcher and
falls back gracefully, so you don't have to manage `RANK`/`MASTER_ADDR` yourself.

## 🎛️ Customization

| Want to… | Where |
|----------|-------|
| Switch YOLO variant or task (OBB/seg/cls) | `training.model` in `config.yaml` (e.g. `yolov8m-obb.pt`, `yolov8m-seg.pt`) |
| Use your own checkpoint | set `training.model` to a `.pt` path |
| Tune epochs / batch / image size | `training.*` in `config.yaml` |
| Override per-run without editing config | `python submit_job.py --model yolo11m.pt --epochs 50` |
| Add training packages | `conda.yml`, then re-run `register_env.py` |

## 🛠️ Troubleshooting

- **Snapshot too large on submit** — keep your dataset outside the repo (or delete it after upload). It's already uploaded to the datastore; the job mounts it from there.
- **Cluster won't start** — check quota/availability in Azure ML Studio, or try a different `--cluster`.
- **Dependency errors on the cluster** — pin versions in `conda.yml` and re-run `register_env.py`.
- **`KeyError: ROBOFLOW_API_KEY`** — copy `.env.example` to `.env` and fill it in.
- **`config.yaml not found`** — copy `config.example.yaml` to `config.yaml`.

## 🔒 Security

No credentials are committed to this repository. `config.json` (Azure subscription),
`config.yaml` (your resource names), and `.env` (Roboflow API key) are all **gitignored**
and provided only as templates. Never commit real keys or subscription IDs.

## 📄 License

Released under the [MIT License](LICENSE).
