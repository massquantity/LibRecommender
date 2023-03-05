from enum import Enum


class FeatType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"


class Backend(Enum):
    TF = "tensorflow"
    TORCH = "torch"
