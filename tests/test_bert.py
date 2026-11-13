import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from src.bert import encode, tokenize


def test_tokenizer_and_unknown_padding():
    assert tokenize("Claim 123 costs $5.") == ["claim", "123", "costs", "$", "5", "."]
    ids, mask = encode(["known mystery"], {"known": 4}, length=6)
    assert ids.tolist() == [[2, 4, 1, 3, 0, 0]]
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0]]
