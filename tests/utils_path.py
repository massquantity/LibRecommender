import os
import shutil
from pathlib import Path

SAVE_PATH = os.path.join(str(Path(os.path.realpath(__file__)).parent), "save_path")


def remove_path(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
