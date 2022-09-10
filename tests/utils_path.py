import os
from pathlib import Path

SAVE_PATH = os.path.join(
    str(Path(os.path.realpath(__file__)).parent), "save_path",
)
