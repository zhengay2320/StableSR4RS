from __future__ import annotations

import torch

from src.condition_adapter import ConditionAdapter


def test_zero_initialization_is_identity() -> None:
    adapter = ConditionAdapter(adapter_scale=1.0)
    image = torch.rand(2, 3, 16, 12) * 2 - 1
    output = adapter(image)
    assert torch.equal(output, image)
    final = adapter.net[-1]
    assert torch.count_nonzero(final.weight).item() == 0
    assert torch.count_nonzero(final.bias).item() == 0


def test_adapter_safetensors_roundtrip(tmp_path) -> None:
    torch.manual_seed(3)
    adapter = ConditionAdapter(adapter_scale=0.75)
    with torch.no_grad():
        adapter.net[-1].weight.normal_(std=0.01)
    before = adapter(torch.rand(1, 3, 8, 8) * 2 - 1)
    path = adapter.save_pretrained(tmp_path)
    assert path.name == "condition_adapter.safetensors"
    restored = ConditionAdapter.from_pretrained(tmp_path)
    assert restored.adapter_scale == 0.75
    torch.manual_seed(12)
    sample = torch.rand(1, 3, 8, 8) * 2 - 1
    assert torch.allclose(adapter(sample), restored(sample))
    assert before.shape == (1, 3, 8, 8)
