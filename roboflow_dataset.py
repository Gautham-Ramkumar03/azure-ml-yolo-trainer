"""Download a YOLO-format dataset from Roboflow.

Credentials and project coordinates are read from environment variables so
that no secrets are committed to source control. Copy `.env.example` to
`.env`, fill in your values, and they will be loaded automatically.
"""

import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "your-workspace")
PROJECT = os.getenv("ROBOFLOW_PROJECT", "your-project")
VERSION = int(os.getenv("ROBOFLOW_VERSION", "1"))
FORMAT = os.getenv("ROBOFLOW_FORMAT", "yolov8-obb")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download(FORMAT)

print(f"Dataset downloaded to: {dataset.location}")
