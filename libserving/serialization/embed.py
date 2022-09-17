import os

import numpy as np

from libreco.bases import EmbedBase
from .common import check_path_exists, save_id_mapping, save_to_json, save_user_consumed


def save_embed(path: str, model: EmbedBase):
    check_path_exists(path)
    save_id_mapping(path, model.data_info)
    save_user_consumed(path, model.data_info)
    save_embeds(path, model.user_embed, "user_embed")
    save_embeds(path, model.item_embed, "item_embed")


def save_embeds(path: str, embeds: np.ndarray, name: str):
    embed_path = os.path.join(path, name)
    embed_dict = dict()
    for i in range(len(embeds)):
        embed_dict[i] = embeds[i].tolist()
    save_to_json(embed_path, embed_dict)
