import itertools
from importlib.util import find_spec

import numpy as np
import torch
import tqdm


# avoid exiting the program if dgl is not installed and user wants to use other algorithms
def check_dgl(cls: type) -> type:
    if find_spec("dgl") is None:
        dgl_model_name = cls.__name__
        torch_model_name = dgl_model_name.replace("DGL", "")
        cls.dgl_error = ModuleNotFoundError(
            f"Failed to import `dgl`, try using `{torch_model_name}` instead of "
            f"`{dgl_model_name}` if you have trouble installing DGL library"
        )
    else:
        cls.dgl_error = None
    return cls


def build_i2i_homo_graph(n_items, user_consumed, item_consumed):
    import dgl

    src_items, dst_items = [], []
    for i in tqdm.trange(n_items, desc="building homo graph"):
        neighbors = set()
        for u in item_consumed[i]:
            neighbors.update(user_consumed[u])
        src_items.extend(neighbors)
        dst_items.extend([i] * len(neighbors))
    src = torch.tensor(src_items, dtype=torch.long)
    dst = torch.tensor(dst_items, dtype=torch.long)
    return dgl.graph((src, dst), num_nodes=n_items)


def build_u2i_hetero_graph(n_users, n_items, user_consumed):
    import dgl

    items = [list(user_consumed[u]) for u in range(n_users)]
    counts = [len(i) for i in items]
    users = torch.arange(n_users).repeat_interleave(torch.tensor(counts))
    items = list(itertools.chain.from_iterable(items))
    items = torch.tensor(items, dtype=torch.long)
    graph_data = {
        ("user", "consumed", "item"): (users, items),
        ("item", "consumed-by", "user"): (items, users),
    }
    num_nodes = {"user": n_users, "item": n_items}
    return dgl.heterograph(graph_data, num_nodes)


def build_subgraphs(heads, item_pairs, paradigm, num_neg):
    import dgl

    heads_pos = torch.as_tensor(heads, dtype=torch.long)
    tails_pos = torch.as_tensor(item_pairs[0], dtype=torch.long)
    tails_neg = torch.as_tensor(item_pairs[1], dtype=torch.long)
    if num_neg > 1:
        heads_neg = heads_pos.repeat_interleave(num_neg)
    else:
        heads_neg = heads_pos
    h_name = "user" if paradigm == "u2i" else "item"
    pos_graph = dgl.heterograph({(h_name, "connect", "item"): (heads_pos, tails_pos)})
    neg_graph = dgl.heterograph({(h_name, "connect", "item"): (heads_neg, tails_neg)})
    pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
    return pos_graph, neg_graph, heads_pos, heads_neg, tails_pos, tails_neg


def compute_u2i_edge_scores(graph, user_reprs, item_reprs):
    import dgl.function as dfn

    with graph.local_scope():
        graph.nodes["user"].data["u"] = user_reprs
        graph.nodes["item"].data["i"] = item_reprs
        graph.apply_edges(dfn.u_dot_v("u", "i", "e"))
        scores = graph.edata["e"].squeeze()
    return scores


def compute_i2i_edge_scores(graph, item_reprs):
    import dgl.function as dfn

    with graph.local_scope():
        graph.ndata["h"] = item_reprs
        graph.apply_edges(dfn.u_dot_v("h", "h", "e"))
        scores = graph.edata["e"].squeeze()
    return scores


def pairs_from_dgl_graph(graph, start_nodes, num_walks, walk_len, focus_start):
    """
    initial:
    t u i u i
    t u i u i

    focus_start:
    t i i  ->  (t t), (i i i i)  repeat-> (t t t t), (i i i i)
    t i i

    not focus_start:
    t i i  repeat->  t t i i i i  slice[:, 1:-1]->  t i i i  reshape-> t i  -> (t i t i), (i i i i)
    t i i            t t i i i i                    t i i i            i i
                                                                       t i
                                                                       i i
    """
    import dgl

    metapath = ["consumed-by", "consumed"] * walk_len
    items, items_pos = [], []
    for _ in range(num_walks):
        walks = dgl.sampling.random_walk(graph, start_nodes, metapath=metapath)[0]
        if focus_start:
            tails = walks[:, 2::2]  # only select walked items
            heads = torch.repeat_interleave(start_nodes, tails.shape[1])
            tails = tails.ravel()
            mask = torch.logical_and(tails != -1, tails != heads)
        else:
            walks = torch.repeat_interleave(walks[:, ::2], repeats=2, dim=1)
            walks = walks[:, 1:-1].reshape(-1, 2)
            heads, tails = walks[:, 0], walks[:, 1]
            mask = (heads != -1) & (tails != -1) & (heads != tails)
        heads, tails = heads[mask], tails[mask]
        items.extend(heads.tolist())
        items_pos.extend(tails.tolist())
    return np.array(items), np.array(items_pos)
