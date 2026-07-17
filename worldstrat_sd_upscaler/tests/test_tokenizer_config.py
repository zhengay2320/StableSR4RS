from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.utils import normalize_tokenizer_max_length


def test_tokenizer_max_length_uses_text_encoder_context() -> None:
    tokenizer = SimpleNamespace(model_max_length=10**30)
    text_encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=77))

    assert normalize_tokenizer_max_length(tokenizer, text_encoder) == 77
    assert tokenizer.model_max_length == 77


def test_tokenizer_max_length_rejects_invalid_context() -> None:
    tokenizer = SimpleNamespace(model_max_length=10**30)
    text_encoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=0))

    with pytest.raises(ValueError, match="max_position_embeddings"):
        normalize_tokenizer_max_length(tokenizer, text_encoder)
