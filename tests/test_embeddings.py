"""Tests for the embedding cache (driftmon.rag.embeddings), using a fake model."""
import numpy as np

from driftmon.rag.embeddings import Embedder, document_text


class FakeModel:
    """Deterministic 4-d vectors from the text; records every text it sees."""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, batch):
        self.seen.extend(batch)
        return np.array([[len(t), t.count("a"), t.count("e"), 1.0] for t in batch])


def _embedder(tmp_path, model=None, prefix="Q: ", batch_size=2):
    return Embedder("org/fake-model", tmp_path, query_prefix=prefix, batch_size=batch_size,
                    encode_fn=model or FakeModel())


def test_document_text_joins_headline_and_description():
    assert document_text("Big News", "Details here.") == "Big News. Details here."
    assert document_text("What If?", "Then this.") == "What If? Then this."
    assert document_text("  Only Title ", "") == "Only Title"
    assert document_text("", "Only text") == "Only text"


def test_vectors_are_normalised_and_in_input_order(tmp_path):
    e = _embedder(tmp_path)
    v = e.embed_documents(["aaa", "e", "aaa"])
    assert v.shape == (3, 4) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(v[0], v[2]) and not np.allclose(v[0], v[1])


def test_cache_avoids_recomputing_within_and_across_instances(tmp_path):
    model = FakeModel()
    e = _embedder(tmp_path, model)
    first = e.embed_documents(["one", "two", "one", "three"])
    assert model.seen == ["one", "two", "three"]  # duplicate sent once, batches of 2
    e.embed_documents(["two", "three"])
    assert e.model_calls == 3

    model2 = FakeModel()
    e2 = _embedder(tmp_path, model2)  # new instance reads the cache from disk
    again = e2.embed_documents(["three", "one", "two"])
    assert model2.seen == []
    assert np.allclose(again, first[[3, 0, 1]])


def test_queries_get_prefix_and_differ_from_documents(tmp_path):
    model = FakeModel()
    e = _embedder(tmp_path, model, prefix="Q: ")
    e.embed_queries(["find me"])
    e.embed_documents(["find me"])
    assert model.seen == ["Q: find me", "find me"]


def test_models_have_separate_caches(tmp_path):
    a = Embedder("org/model-a", tmp_path, encode_fn=FakeModel())
    a.embed_documents(["text"])
    model_b = FakeModel()
    Embedder("org/model-b", tmp_path, encode_fn=model_b).embed_documents(["text"])
    assert model_b.seen == ["text"]


def test_empty_input(tmp_path):
    assert _embedder(tmp_path).embed_documents([]).shape[0] == 0
