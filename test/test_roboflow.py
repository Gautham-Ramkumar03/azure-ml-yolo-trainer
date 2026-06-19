"""Smoke test for Roboflow connectivity.

Reads credentials from environment variables (loaded from `.env` if present)
so that no API key is hard-coded. Run with:

    python test/test_roboflow.py
"""

import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE_NAME = os.getenv("ROBOFLOW_WORKSPACE", "your-workspace")
PROJECT_NAME = os.getenv("ROBOFLOW_PROJECT", "your-project")


def test_roboflow_connection():
    print("Testing Roboflow connection...")

    try:
        # Initialize Roboflow with API key
        rf = Roboflow(api_key=API_KEY)
        print("✅ Successfully authenticated with Roboflow API")

        # Try to access the workspace
        try:
            workspace = rf.workspace(WORKSPACE_NAME)
            print(f"✅ Successfully accessed workspace: {WORKSPACE_NAME}")

            # Try to access the project
            try:
                project = workspace.project(PROJECT_NAME)
                print(f"✅ Successfully accessed project: {PROJECT_NAME}")
                print(f"   Project type: {project.type}")
                print(f"   Project versions: {len(project.versions())}")

            except Exception as e:
                print(f"❌ Failed to access project: {PROJECT_NAME}")
                print(f"   Error: {str(e)}")

        except Exception as e:
            print(f"❌ Failed to access workspace: {WORKSPACE_NAME}")
            print(f"   Error: {str(e)}")

    except Exception as e:
        print("❌ Failed to authenticate with Roboflow API")
        print(f"   Error: {str(e)}")


if __name__ == "__main__":
    test_roboflow_connection()
