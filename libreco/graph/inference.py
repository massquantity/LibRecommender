import torch
from torch.nn import functional as F
from tqdm import tqdm

from .from_dgl import build_i2i_homo_graph


def full_neighbor_embeddings(model):
    """Full inference by aggregating over all neighbor embeddings on the homogeneous graph.

    original graph -> CPU
    `torch_model` -> CPU or GPU
    returned embeddings -> CPU
    """
    from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

    is_graphsage = model.model_name.startswith("GraphSage")
    is_pinsage = model.model_name.startswith("PinSage")
    if hasattr(model, "homo_g"):
        g = model.homo_g
    else:
        g = build_i2i_homo_graph(
            model.n_items, model.user_consumed, model.data_info.item_consumed
        )

    g.ndata["feat"] = _compute_features(
        model.torch_model,
        model.data_info,
        model.n_items,
        model.batch_size,
        model.device,
    )
    prefetch_node_feats = ["feat"]
    prefetch_edge_feats = None
    if is_pinsage:
        g.edata["weights"] = torch.ones(g.num_edges())
        prefetch_edge_feats = ["weights"]

    sampler = MultiLayerFullNeighborSampler(
        num_layers=1,
        prefetch_node_feats=prefetch_node_feats,
        prefetch_edge_feats=prefetch_edge_feats,
    )
    dataloader = DataLoader(
        g,
        torch.arange(model.n_items, device=g.device),
        sampler,
        device=model.device,
        batch_size=model.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    x, y = g.ndata["feat"], None
    conv_layers = model.torch_model.layers
    for i, layer in enumerate(conv_layers, start=1):
        y = torch.empty(model.n_items, model.embed_size)
        x = x.to(model.device)
        for input_nodes, output_nodes, blocks in tqdm(
            dataloader, desc=f"full inference on layer {i}"
        ):
            # all neighbor in one layer, i.e., one block
            h = layer(blocks[0], x[input_nodes])
            if is_graphsage and i != len(conv_layers):
                h = F.relu(h)
            elif is_pinsage and i == len(conv_layers):
                h = model.torch_model.final_linear(h)
            y[output_nodes] = h.cpu()
        x = y
    return y.cpu().numpy()


def _compute_features(torch_model, data_info, n_items, batch_size, device):
    features = []
    all_items = list(range(n_items))
    for i in range(0, n_items, batch_size):
        items = all_items[i : i + batch_size]
        sparse_indices, dense_values = None, None
        if data_info.item_sparse_unique is not None:
            sparse_indices = torch.from_numpy(data_info.item_sparse_unique[items])
            sparse_indices = sparse_indices.to(device)
        if data_info.item_dense_unique is not None:
            dense_values = torch.from_numpy(data_info.item_dense_unique[items])
            dense_values = dense_values.to(device)
        items = torch.tensor(items, dtype=torch.long, device=device)
        features.append(
            torch_model.get_raw_features(
                items, sparse_indices, dense_values, is_user=False
            )
        )
    return torch.cat(features, dim=0).cpu()
