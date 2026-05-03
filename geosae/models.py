"""Model definitions for GeoSAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_ACTIVATIONS = ("topk", "relu", "jumprelu")


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with a switchable activation gate.

    Supported activations:
      - ``topk``: keep the ``k`` largest positive pre-activations (paper default).
      - ``relu``: standard ReLU SAE used in Table 2 of the paper.
      - ``jumprelu``: a learnable per-feature threshold above which ReLU passes
        through. Used here as the prior-analysis baseline; not the
        paper's main method.
    """

    def __init__(
        self,
        input_dim: int,
        expansion_factor: int,
        topk: int = 16,
        activation: str = "topk",
        jumprelu_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_VALID_ACTIVATIONS}, got {activation!r}.")
        self.input_dim = input_dim
        self.latent_dim = input_dim * expansion_factor
        self.topk = topk
        self.activation = activation

        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim, bias=False)

        if activation == "jumprelu":
            self.log_threshold = nn.Parameter(
                torch.full((self.latent_dim,), float(torch.log(torch.tensor(jumprelu_threshold)))))
        else:
            self.register_parameter("log_threshold", None)

        self.register_buffer(
            "neuron_activate_counts",
            torch.zeros(self.latent_dim, dtype=torch.long),
        )
        self.register_buffer("num_steps", torch.tensor(0, dtype=torch.long))
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        self.normalize_decoder()

    def pre_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encoder pre-activations, before any sparsity gate."""
        return self.encoder(inputs - self.pre_bias)

    def _apply_activation(self, pre_activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.activation == "topk":
            topk_values, topk_indices = torch.topk(pre_activations, self.topk, dim=-1)
            latents = torch.zeros_like(pre_activations)
            latents.scatter_(-1, topk_indices, F.relu(topk_values))
            return latents, topk_indices
        if self.activation == "relu":
            return F.relu(pre_activations), None
        # jumprelu
        threshold = torch.exp(self.log_threshold)
        gate = (pre_activations > threshold).float()
        return F.relu(pre_activations) * gate, None

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns sparse latents and (for TopK only) selected indices."""
        return self._apply_activation(self.pre_activations(inputs))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        latents, topk_indices = self.encode(inputs)
        reconstructions = self.decode(latents)
        mse = F.mse_loss(reconstructions, inputs)
        if self.training:
            self.num_steps += 1
            self.neuron_activate_counts += (latents > 0).any(dim=0).long()
        outputs: dict[str, torch.Tensor] = {
            "x_hat": reconstructions,
            "latents": latents,
            "mse": mse,
        }
        if topk_indices is not None:
            outputs["topk_indices"] = topk_indices
        return outputs

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    @torch.no_grad()
    def alive_feature_mask(self) -> torch.Tensor:
        return self.neuron_activate_counts > 0


class TopKSparseAutoencoder(SparseAutoencoder):
    """Backwards-compatible alias for the TopK variant used in the paper."""

    def __init__(self, input_dim: int, expansion_factor: int, topk: int) -> None:
        super().__init__(
            input_dim=input_dim,
            expansion_factor=expansion_factor,
            topk=topk,
            activation="topk",
        )
