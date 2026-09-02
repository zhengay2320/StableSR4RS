from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from scripts.evaluate_cas_all_backbone_only import run_backbone_only_pipeline


class _CountingIdentityAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.last_shape: tuple[int, ...] | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_shape = tuple(value.shape)
        return value


class _FakePipeline:
    vae_scale_factor = 4

    def __init__(self) -> None:
        self.calls = 0
        self.last_image_size: tuple[int, int] | None = None

    def __call__(self, **kwargs):
        self.calls += 1
        image = kwargs["image"]
        self.last_image_size = image.size
        return SimpleNamespace(
            images=[image.resize((image.width * 4, image.height * 4), Image.Resampling.BICUBIC)]
        )


def test_backbone_only_path_uses_adapter_and_original_pipeline_geometry() -> None:
    pipe = _FakePipeline()
    adapter = _CountingIdentityAdapter()
    image = Image.new("RGB", (7, 5), (20, 40, 60))
    output = run_backbone_only_pipeline(
        pipe,
        adapter,
        image,
        torch.Generator().manual_seed(42),
        torch.device("cpu"),
        torch.float32,
    )
    assert adapter.calls == 1
    assert adapter.last_shape == (1, 3, 8, 8)
    assert pipe.calls == 1
    assert pipe.last_image_size == (8, 8)
    assert output.size == (28, 20)


def test_standalone_script_does_not_import_latent_phi() -> None:
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_cas_all_backbone_only.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert "src.latent_phi" not in imported_modules
    assert "src.latent_phi" not in imported_from
