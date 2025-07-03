import sys
from pathlib import Path

# APP_ROOT is the src/ folder in the lab template, resolved based on the full path of the __init__.py file
APP_ROOT = Path(__file__).resolve().parent

# PROJECT_ROOT is the root of the repository, one level above the src/ module
PROJECT_ROOT = APP_ROOT.parent

# The rest are paths for the different folders of the repository
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_PATH = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# TODO: if os is mac, find different scratch path vs linux
if sys.platform == "linux":
    SCRATCH_DIR = Path.home() / "scratch"
    print("Detected Linux system, using scratch directory: ", SCRATCH_DIR)
else:
    SCRATCH_DIR = PROJECT_ROOT.parent / "scratch"  # hardcoded for Christina's local machine
    print("Detected non-Linux system, using scratch directory: ", SCRATCH_DIR)
