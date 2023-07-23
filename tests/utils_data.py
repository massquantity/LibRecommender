import os
import shutil
from pathlib import Path

import numpy as np

from libreco.data import TransformedEvalSet, TransformedSet

SAVE_PATH = os.path.join(str(Path(os.path.realpath(__file__)).parent), "save_path")


def remove_path(path):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def set_ranking_labels(data):
    if isinstance(data, TransformedSet):
        original_labels = data._labels.copy()
        data._labels[original_labels >= 4] = 1
        data._labels[original_labels < 4] = 0
    elif isinstance(data, TransformedEvalSet):
        original_labels = np.copy(data.labels)
        data.labels[original_labels >= 4] = 1
        data.labels[original_labels < 4] = 0
