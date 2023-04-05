import os

import numpy as np

from libreco.bases import EmbedBase

from .common import (
    check_path_exists,
    save_id_mapping,
    save_model_name,
    save_to_json,
    save_user_consumed,
)


def save_embed(path: str, model: EmbedBase):
    """Save Embed model to disk.

    Parameters
    ----------
    path : str
        Model saving path.
    model : EmbedBase
        Model to save.
    """
    check_path_exists(path)
    save_model_name(path, model)
    save_id_mapping(path, model.data_info)
    save_user_consumed(path, model.data_info)
    save_vectors(path, model.user_embed, model.n_users, "user_embed.json")
    save_vectors(path, model.item_embed, model.n_items, "item_embed.json")


def save_vectors(path: str, embeds: np.ndarray, num: int, name: str):
    embed_path = os.path.join(path, name)
    embed_dict = dict()
    for i in range(num):
        embed_dict[i] = embeds[i].tolist()
    save_to_json(embed_path, embed_dict)


def save_faiss_index(path: str, model: EmbedBase, nlist: int = 80, nprobe: int = 10):
    import faiss

    check_path_exists(path)
    index_path = os.path.join(path, "faiss_index.bin")
    item_embeds = model.item_embed[: model.n_items].astype(np.float32)
    d = item_embeds.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(item_embeds)
    index.add(item_embeds)
    index.nprobe = nprobe
    faiss.write_index(index, index_path)
