import numpy as np
import torch


def to_tensor(data, device=None, dtype=None):
    if isinstance(data, (list, tuple)):
        return torch.tensor(data, dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device=device, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        if data.device != device:
            return data.to(device=device)
        else:
            return data
    else:
        raise ValueError(f"unknown data type when converting tensor: {type(data)}")


def feat_to_tensor(ids, sparse_indices, dense_values, device):
    id_tensor = to_tensor(ids, device=device, dtype=torch.long)
    sparse_tensor = (
        to_tensor(sparse_indices, device=device, dtype=torch.long)
        if sparse_indices is not None
        else None
    )
    dense_tensor = (
        to_tensor(dense_values, device=device, dtype=torch.float)
        if dense_values is not None
        else None
    )
    return id_tensor, sparse_tensor, dense_tensor


def user_unique_to_tensor(users, data_info, device):
    user_tensor = to_tensor(users, device=device, dtype=torch.long)
    if isinstance(users, torch.Tensor):
        users = users.cpu().numpy()
    sparse_tensor = (
        to_tensor(data_info.user_sparse_unique[users], device=device, dtype=torch.long)
        if data_info.user_sparse_unique is not None
        else None
    )
    dense_tensor = (
        to_tensor(data_info.user_dense_unique[users], device=device, dtype=torch.float)
        if data_info.user_dense_unique is not None
        else None
    )
    return user_tensor, sparse_tensor, dense_tensor


def item_unique_to_tensor(items, data_info, device):
    item_tensor = to_tensor(items, device=device, dtype=torch.long)
    if isinstance(items, torch.Tensor):
        items = items.cpu().numpy()
    sparse_tensor = (
        to_tensor(data_info.item_sparse_unique[items], device=device, dtype=torch.long)
        if data_info.item_sparse_unique is not None
        else None
    )
    dense_tensor = (
        to_tensor(data_info.item_dense_unique[items], device=device, dtype=torch.float)
        if data_info.item_dense_unique is not None
        else None
    )
    return item_tensor, sparse_tensor, dense_tensor
