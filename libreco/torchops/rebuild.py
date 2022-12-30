import torch
from torch import nn

from ..utils.save_load import load_torch_state_dict
from ..utils.validate import sparse_feat_size


@torch.no_grad()
def rebuild_torch_model(self, path, model_name):
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

    for name, param in self.torch_model.named_parameters():
        if name in user_params:
            param.index_copy_(
                0, torch.arange(self.data_info.old_n_users), user_params[name]
            )
        elif name in item_params:
            param.index_copy_(
                0, torch.arange(self.data_info.old_n_items), item_params[name]
            )
        elif name in sparse_params:
            old_indices, old_values = get_sparse_indices_values(
                self.data_info, sparse_params[name]
            )
            param.index_copy_(0, old_indices, old_values)

    for i in user_param_indices:
        optimizer_user_state = optimizer_state_dict["state"][i]
        assign_adam_optimizer_states(
            optimizer_user_state,
            self.data_info.old_n_users,
            self.n_users,
            self.embed_size,
        )
    for i in item_param_indices:
        optimizer_item_state = optimizer_state_dict["state"][i]
        assign_adam_optimizer_states(
            optimizer_item_state,
            self.data_info.old_n_items,
            self.n_items,
            self.embed_size,
        )
    for i in sparse_param_indices:
        optimizer_sparse_state = optimizer_state_dict["state"][i]
        assign_adam_sparse_states(
            optimizer_sparse_state, self.data_info, self.embed_size
        )

    self.trainer.optimizer.load_state_dict(optimizer_state_dict)


def get_sparse_indices_values(data_info, sparse_param_tensor):
    indices = list(range(len(sparse_param_tensor)))
    # remove oov indices
    for i in data_info.old_sparse_oov:
        indices.remove(i)
    sparse_param_tensor = sparse_param_tensor[indices]

    indices = []
    for offset, size in zip(data_info.sparse_offset, data_info.old_sparse_len):
        if size != -1:
            indices.extend(range(offset, offset + size))
    indices = torch.tensor(indices, dtype=torch.long)
    return indices, sparse_param_tensor


def assign_adam_optimizer_states(state, old_num, new_num, embed_size):
    new_first_moment = nn.init.zeros_(torch.empty(new_num, embed_size))
    new_first_moment.index_copy_(0, torch.arange(old_num), state["exp_avg"])
    state["exp_avg"] = new_first_moment
    new_second_moment = nn.init.zeros_(torch.empty(new_num, embed_size))
    new_second_moment.index_copy_(0, torch.arange(old_num), state["exp_avg_sq"])
    state["exp_avg_sq"] = new_second_moment


def assign_adam_sparse_states(state, data_info, embed_size):
    sparse_size = sparse_feat_size(data_info)
    new_first_moment = nn.init.zeros_(torch.empty(sparse_size, embed_size))
    old_indices, old_values = get_sparse_indices_values(data_info, state["exp_avg"])
    new_first_moment.index_copy_(0, old_indices, old_values)
    state["exp_avg"] = new_first_moment
    new_second_moment = nn.init.zeros_(torch.empty(sparse_size, embed_size))
    old_indices, old_values = get_sparse_indices_values(data_info, state["exp_avg_sq"])
    new_second_moment.index_copy_(0, old_indices, old_values)
    state["exp_avg_sq"] = new_second_moment
