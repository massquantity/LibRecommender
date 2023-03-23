import torch

from .message import ItemMessage, ItemMessageDGL, UserMessage
from ..sampling import bipartite_neighbors, bipartite_neighbors_with_weights


class NeighborWalker:
    def __init__(self, model, data_info):
        self.paradigm = model.paradigm
        self.num_layers = model.num_layers
        self.num_neighbors = model.num_neighbors
        self.remove_edges = model.remove_edges
        self.user_consumed = data_info.user_consumed
        self.item_consumed = data_info.item_consumed
        self.user_sparse_unique = data_info.user_sparse_unique
        self.user_dense_unique = data_info.user_dense_unique
        self.item_sparse_unique = data_info.item_sparse_unique
        self.item_dense_unique = data_info.item_dense_unique
        self.use_pinsage = "PinSage" in model.model_name
        if self.use_pinsage:
            self.num_walks = model.num_walks
            self.walk_length = model.neighbor_walk_len
            self.termination_prob = model.termination_prob

    def __call__(self, items, items_pos=None):
        nodes, sparse, dense = self.get_item_feats(items)
        nbs, nbs_sparse, nbs_dense, offsets, weights = (
            self.sample_pinsage(items, items_pos)
            if self.use_pinsage
            else self.sample_graphsage(items)
        )
        return ItemMessage(
            nodes, sparse, dense, nbs, nbs_sparse, nbs_dense, offsets, weights
        )

    def sample_graphsage(self, items):
        neighbors, neighbors_sparse, neighbors_dense, offsets = [], [], [], []
        nodes = items
        weights = None
        for _ in range(self.num_layers):
            nbs, offs = bipartite_neighbors(
                nodes, self.user_consumed, self.item_consumed, self.num_neighbors
            )
            nbs, sparse, dense = self.get_item_feats(nbs)
            neighbors.append(nbs)
            neighbors_sparse.append(sparse)
            neighbors_dense.append(dense)
            offsets.append(offs)
            nodes = nbs
        return neighbors, neighbors_sparse, neighbors_dense, offsets, weights

    def sample_pinsage(self, items, items_pos=None):
        neighbors, neighbors_sparse, neighbors_dense = [], [], []
        offsets, weights = [], []
        nodes = items
        if self.paradigm == "i2i" and self.remove_edges and items_pos is not None:
            item_indices = list(range(len(items)))
        else:
            item_indices = None
        for _ in range(self.num_layers):
            nbs, ws, offs, item_indices_in_samples = bipartite_neighbors_with_weights(
                nodes,
                self.user_consumed,
                self.item_consumed,
                self.num_neighbors,
                self.num_walks,
                self.walk_length,
                items,
                item_indices,
                items_pos,
                self.termination_prob,
            )
            nbs, sparse, dense = self.get_item_feats(nbs)
            neighbors.append(nbs)
            neighbors_sparse.append(sparse)
            neighbors_dense.append(dense)
            offsets.append(offs)
            weights.append(ws)
            nodes = nbs
            item_indices = item_indices_in_samples
        return neighbors, neighbors_sparse, neighbors_dense, offsets, weights

    def get_item_feats(self, items):
        if isinstance(items, torch.Tensor):
            items = items.detach().cpu().numpy()
        sparse, dense = None, None
        if self.item_sparse_unique is not None:
            sparse = self.item_sparse_unique[items]
        if self.item_dense_unique is not None:
            dense = self.item_dense_unique[items]
        return items, sparse, dense

    def get_user_feats(self, users):
        if isinstance(users, torch.Tensor):
            users = users.detach().cpu().numpy()
        sparse, dense = None, None
        if self.user_sparse_unique is not None:
            sparse = self.user_sparse_unique[users]
        if self.user_dense_unique is not None:
            dense = self.user_dense_unique[users]
        return UserMessage(users, sparse, dense)


class NeighborWalkerDGL(NeighborWalker):
    def __init__(self, model, data_info):
        super().__init__(model, data_info)
        self.graph = model.hetero_g if self.use_pinsage else model.homo_g

    def __call__(self, items, target_nodes=None):
        import dgl

        # use torch tensor
        blocks = self.transform_blocks(items, target_nodes)
        start_nb_nodes = blocks[0].srcdata[dgl.NID]
        start_nodes, nbs_sparse, nbs_dense = self.get_item_feats(start_nb_nodes)
        return ItemMessageDGL(blocks, start_nodes, nbs_sparse, nbs_dense)

    def sample_frontier(self, nodes):
        import dgl

        if self.use_pinsage:
            sampler = dgl.sampling.PinSAGESampler(
                self.graph,
                ntype="item",
                other_type="user",
                num_traversals=self.walk_length,
                termination_prob=self.termination_prob,
                num_random_walks=self.num_walks,
                num_neighbors=self.num_neighbors,
            )
            return sampler(nodes)
        else:
            return dgl.sampling.sample_neighbors(
                g=self.graph,
                nodes=nodes,
                fanout=self.num_neighbors,
                edge_dir="in",
            )

    def transform_blocks(self, nodes, target_nodes=None):
        r"""Bipartite graph block: items(nodes) -> sampled neighbor nodes

        -------------
        |     / ... |
        |    /  src |
        |dst -  src |
        |    \  src |
        |     \ ... |
        -------------
        """
        import dgl

        blocks = []
        for _ in range(self.num_layers):
            frontier = self.sample_frontier(nodes)
            if (
                self.paradigm == "i2i"
                and self.remove_edges
                and target_nodes is not None
            ):
                heads_pos, heads_neg, tails_pos, tails_neg = target_nodes
                eids = frontier.edge_ids(
                    torch.cat([heads_pos, heads_neg]),
                    torch.cat([tails_pos, tails_neg]),
                    return_uv=True,
                )[2]
                if len(eids) > 0:
                    frontier = dgl.remove_edges(frontier, eids)
            block = dgl.to_block(frontier, dst_nodes=nodes)
            nodes = block.srcdata[dgl.NID]
            blocks.append(block)
        blocks.reverse()
        return blocks
