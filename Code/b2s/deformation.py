from __future__ import annotations

import torch
from torch import nn

from .mesh_utils import Mesh


class VertexDeformationField(nn.Module):
    """Predict per-vertex offsets from a global latent scene/object code."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, vertices: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.view(1, -1).expand(vertices.shape[0], -1)
        return self.net(torch.cat([vertices, latent], dim=-1))


def deform_mesh(mesh: Mesh, latent: torch.Tensor, field: VertexDeformationField, scale: float = 0.05) -> Mesh:
    """Apply a smooth learned vertex offset field to the mesh."""
    offsets = field(mesh.vertices, latent.to(mesh.vertices.device))
    return Mesh(mesh.vertices + scale * offsets, mesh.faces.clone(), f"{mesh.name}_deformed")
