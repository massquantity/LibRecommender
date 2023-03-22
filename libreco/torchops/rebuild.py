"""Rebuild PyTorch models."""
from dataclasses import astuple

import torch
from torch import nn

from ..utils.save_load import load_torch_state_dict
from ..utils.validate import sparse_feat_size


# noinspection PyIncorrectDocstring
@torch.no_grad()
def rebuild_torch_model(self, path, model_name):
    """Assign the saved model variables to the newly initialized model.

    This method is used before retraining the new model, in order to avoid training
    from scratch every time we get some new data.

    Parameters
    ----------
    path : str
        File folder path for the saved model variables.
    model_name : str
        Name of the saved model file.
    """
    from ..training.dispatch import get_trainer

    self.model_built = True
    self.build_model()
    self.trainer = get_trainer(self)

    model_state_dict, optimizer_state_dict = load_torch_state_dict(
        path, model_name, self.device
    )
    user_param_indices, item_param_indices = list(), list()
    user_params, item_params = dict(), dict()
    sparse_param_indices, sparse_params = list(), dict()
    # remove user, item and sparse parameters from state_dict and assign later
    for i, name in enumerate(list(model_state_dict.keys())):
        if "user" in name and "embed" in name:
            user_param_indices.append(i)
            user_params[name] = model_state_dict.pop(name)
        elif "item" in name and "embed" in name:
            item_param_indices.append(i)
            item_params[name] = model_state_dict.pop(name)
        elif "sparse" in name and "embed" in name:
            sparse_param_indices.append(i)
            sparse_params[name] = model_state_dict.pop(name)

    # `strict=False` ignores non-matching keys
    self.torch_model.load_state_dict(model_state_dict, strict=False)

    sparse_offset = self.data_info.sparse_offset
    old_n_users, old_n_items, old_sparse_len, old_sparse_oov, _ = astuple(
        self.data_info.old_info
    )

    for name, param in self.torch_model.named_parameters():
        if name in user_params:
            index = torch.arange(old_n_users, device=self.device)
            param.index_copy_(0, index, user_params[name])
        elif name in item_params:
            index = torch.arange(old_n_items, device=self.device)
            param.index_copy_(0, index, item_params[name])
        elif name in sparse_params:
            old_indices, old_values = get_sparse_indices_values(
                sparse_offset,
                old_sparse_len,
                old_sparse_oov,
                sparse_params[name],
                self.device,
            )
            param.index_copy_(0, old_indices, old_values)

    for i in user_param_indices:
        optimizer_user_state = optimizer_state_dict["state"][i]
        assign_adam_optimizer_states(
            optimizer_user_state,
            old_n_users,
            self.n_users,
            self.embed_size,
            self.device,
        )
    for i in item_param_indices:
        optimizer_item_state = optimizer_state_dict["state"][i]
        assign_adam_optimizer_states(
            optimizer_item_state,
            old_n_items,
            self.n_items,
            self.embed_size,
            self.device,
        )
    for i in sparse_param_indices:
        optimizer_sparse_state = optimizer_state_dict["state"][i]
        assign_adam_sparse_states(
            optimizer_sparse_state, self.data_info, self.embed_size, self.device
        )

    self.trainer.optimizer.load_state_dict(optimizer_state_dict)


def get_sparse_indices_values(
    sparse_offset, old_sparse_len, old_sparse_oov, sparse_param_tensor, device
):
    indices = list(range(len(sparse_param_tensor)))
    # remove oov indices
    for i in old_sparse_oov:
        indices.remove(i)
    sparse_param_tensor = sparse_param_tensor[indices]

    indices = []
    for offset, size in zip(sparse_offset, old_sparse_len):
        if size != -1:
            indices.extend(range(offset, offset + size))
    indices = torch.tensor(indices, dtype=torch.long, device=device)
    return indices, sparse_param_tensor


def assign_adam_optimizer_states(state, old_num, new_num, embed_size, device):
    index = torch.arange(old_num, device=device)
    new_first_moment = nn.init.zeros_(torch.empty(new_num, embed_size, device=device))
    new_first_moment.index_copy_(0, index, state["exp_avg"])
    state["exp_avg"] = new_first_moment
    new_second_moment = nn.init.zeros_(torch.empty(new_num, embed_size, device=device))
    new_second_moment.index_copy_(0, index, state["exp_avg_sq"])
    state["exp_avg_sq"] = new_second_moment


def assign_adam_sparse_states(state, data_info, embed_size, device):
    _, _, old_sparse_len, old_sparse_oov, _ = astuple(data_info.old_info)
    sparse_offset = data_info.sparse_offset
    sparse_size = sparse_feat_size(data_info)
    new_first_moment = nn.init.zeros_(
        torch.empty(sparse_size, embed_size, device=device)
    )
    old_indices, old_values = get_sparse_indices_values(
        sparse_offset, old_sparse_len, old_sparse_oov, state["exp_avg"], device
    )
    new_first_moment.index_copy_(0, old_indices, old_values)
    state["exp_avg"] = new_first_moment
    new_second_moment = nn.init.zeros_(
        torch.empty(sparse_size, embed_size, device=device)
    )
    old_indices, old_values = get_sparse_indices_values(
        sparse_offset, old_sparse_len, old_sparse_oov, state["exp_avg_sq"], device
    )
    new_second_moment.index_copy_(0, old_indices, old_values)
    state["exp_avg_sq"] = new_second_moment
