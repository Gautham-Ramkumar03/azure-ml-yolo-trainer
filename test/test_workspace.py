from azureml.core import Workspace

ws = Workspace.from_config()  # Loads config.json from current directory
print("Workspace name:", ws.name)