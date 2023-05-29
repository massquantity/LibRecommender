import os

import pytest

from libreco.algorithms import BPR
from libserving.serialization import save_faiss_index
from tests.utils_data import SAVE_PATH


def test_faiss_index(embed_model):
    import faiss

    save_faiss_index(SAVE_PATH, embed_model, 80, 10)
    index = faiss.read_index(os.path.join(SAVE_PATH, "faiss_index.bin"))
    _, ids = index.search(embed_model.user_embeds_np[0].reshape(1, -1), 10)
    assert ids.shape == (1, 10)
    assert index.ntotal == embed_model.n_items
    assert index.d == embed_model.embed_size + 1  # embed + bias


@pytest.fixture
def embed_model(prepare_pure_data):
    _, train_data, _, data_info = prepare_pure_data
    model = BPR(
        data_info=data_info,
        n_epochs=2,
        lr=1e-4,
        batch_size=2048,
        use_tf=False,
        optimizer="adam",
    )
    model.fit(train_data, neg_sampling=True, verbose=2)
    return model
