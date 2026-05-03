"""Tests for the SAE model definitions."""

from __future__ import annotations

import pytest
import torch

from geosae.models import SparseAutoencoder, TopKSparseAutoencoder


@pytest.mark.parametrize("activation", ["topk", "relu", "jumprelu"])
def test_forward_shapes(activation: str) -> None:
    model = SparseAutoencoder(input_dim=8, expansion_factor=2, topk=2, activation=activation)
    x = torch.randn(4, 8)
    outputs = model(x)
    assert outputs["latents"].shape == (4, 16)
    assert outputs["x_hat"].shape == x.shape
    assert outputs["mse"].shape == ()


def test_topk_exact_sparsity() -> None:
    model = SparseAutoencoder(input_dim=8, expansion_factor=2, topk=3, activation="topk")
    x = torch.randn(4, 8)
    latents = model(x)["latents"]
    assert (latents > 0).sum(dim=-1).max().item() <= 3


def test_relu_no_negatives() -> None:
    model = SparseAutoencoder(input_dim=8, expansion_factor=2, topk=2, activation="relu")
    latents = model(torch.randn(4, 8))["latents"]
    assert (latents < 0).sum().item() == 0


def test_invalid_activation_raises() -> None:
    with pytest.raises(ValueError):
        SparseAutoencoder(input_dim=4, expansion_factor=2, topk=2, activation="spade")


def test_topk_alias() -> None:
    model = TopKSparseAutoencoder(input_dim=8, expansion_factor=2, topk=2)
    assert model.activation == "topk"


def test_decoder_unit_norm() -> None:
    model = SparseAutoencoder(input_dim=8, expansion_factor=2, topk=2, activation="topk")
    column_norms = model.decoder.weight.norm(dim=0)
    assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-5)
