import torch


def feat_to_tensor(ids, sparse_indices, dense_values, device):
    id_tensor = torch.LongTensor(ids, device=device)
    sparse_tensor = (
        torch.LongTensor(sparse_indices, device=device)
        if sparse_indices is not None
        else None
    )
    dense_tensor = (
        torch.FloatTensor(dense_values, device=device)
        if dense_values is not None
        else None
    )
    return id_tensor, sparse_tensor, dense_tensor


def user_unique_to_tensor(users, data_info, device):
    user_tensor = torch.LongTensor(users, device=device)
    sparse_tensor = (
        torch.LongTensor(data_info.user_sparse_unique[users], device=device)
        if data_info.user_sparse_unique is not None
        else None
    )
    dense_tensor = (
        torch.FloatTensor(data_info.user_dense_unique[users], device=device)
        if data_info.user_dense_unique is not None
        else None
    )
    return user_tensor, sparse_tensor, dense_tensor


def item_unique_to_tensor(items, data_info, device):
    item_tensor = torch.LongTensor(items, device=device)
    sparse_tensor = (
        torch.LongTensor(data_info.item_sparse_unique[items], device=device)
        if data_info.item_sparse_unique is not None
        else None
    )
    dense_tensor = (
        torch.FloatTensor(data_info.item_dense_unique[items], device=device)
        if data_info.item_dense_unique is not None
        else None
    )
    return item_tensor, sparse_tensor, dense_tensor
