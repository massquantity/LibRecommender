import torch


def device_config(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def hidden_units_config(hidden_units):
    if isinstance(hidden_units, int):
        return [hidden_units]
    elif not isinstance(hidden_units, (list, tuple)):
        raise ValueError(
            f"`hidden_units` must be one of (int, list of int, tuple of int), "
            f"got: {type(hidden_units)}, {hidden_units}"
        )
    for i in hidden_units:
        if not isinstance(i, int):
            raise ValueError(f"`hidden_units` contains not int value: {hidden_units}")
    return list(hidden_units)
