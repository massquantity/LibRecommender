import os
import shutil
from pathlib import Path

from libreco.data import TransformedSet

SAVE_PATH = os.path.join(str(Path(os.path.realpath(__file__)).parent), "save_path")


def remove_path(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def set_ranking_labels(data: TransformedSet):
    original_labels = data._labels.copy()
    data._labels[original_labels >= 4] = 1
    data._labels[original_labels < 4] = 0
