import json
import os
import faiss
import numpy as np


def load_item_vector(path):
    i_vec_path = os.path.join(path, "item_vector.json")
    with open(i_vec_path, "r") as f2:
        item_vector_json = json.load(f2)
    length = len(item_vector_json)
    embed_size = len(item_vector_json['1'])
    item_vectors = np.zeros((length, embed_size), dtype=np.float32)
    for i in range(length):
        item_vectors[i] = item_vector_json[str(i)]
    return item_vectors


def train_faiss(item_vector):
    quantizer = faiss.IndexFlatL2(item_vector.shape[1])
    index = faiss.IndexIVFFlat(quantizer, item_vector.shape[1], 80)
    index.train(item_vector)
    index.add(item_vector)
    return index


def save_faiss_index(path):
    item_vector = load_item_vector(path)
    index = train_faiss(item_vector)
    index_path = os.path.join(path, "faiss_index.bin")
    faiss.write_index(index, index_path)

