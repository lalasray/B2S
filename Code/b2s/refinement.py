from __future__ import annotations

from dataclasses import dataclass

import torch

from .assets import AssetLibrary, RetrievedAsset, retrieve_and_fit_asset
from .deformation import VertexDeformationField, deform_mesh
from .mesh_utils import Mesh, approximate_collision_score, mesh_surface_distance


@dataclass
class RefinementResult:
    """Result of iterative scene/object refinement."""

    fitted: RetrievedAsset
    refined_box: torch.Tensor
    iterations: int


class IterativeSceneRefiner:
    """Simple coordinate-descent style refiner over box, support, and mesh deformation."""

    def __init__(self, library: AssetLibrary, deformation_field: VertexDeformationField | None = None, steps: int = 3):
        self.library = library
        self.deformation_field = deformation_field
        self.steps = steps

    def refine(self, object_class: int, object_box: torch.Tensor, support_plane: torch.Tensor, environment_mesh: Mesh | None = None, latent: torch.Tensor | None = None) -> RefinementResult:
        current_box = object_box.clone()
        fitted = retrieve_and_fit_asset(object_class, current_box, support_plane, self.library, environment_mesh)
        best_score = fitted.collision_score + fitted.fit_distance + fitted.support_error
        for _ in range(self.steps):
            proposal = current_box.clone()
            proposal[:3] = torch.clamp(proposal[:3] * 0.97, min=0.2)
            proposal[5] = support_plane[2]
            candidate = retrieve_and_fit_asset(object_class, proposal, support_plane, self.library, environment_mesh)
            candidate_mesh = candidate.fitted_mesh
            if self.deformation_field is not None and latent is not None:
                candidate_mesh = deform_mesh(candidate_mesh, latent, self.deformation_field)
                candidate = RetrievedAsset(
                    label=candidate.label,
                    source_path=candidate.source_path,
                    fitted_mesh=candidate_mesh,
                    support_error=abs(float(candidate_mesh.bounds[0][2].item()) - float(support_plane[2].item())),
                    collision_score=approximate_collision_score(candidate_mesh, environment_mesh) if environment_mesh is not None else 0.0,
                    fit_distance=mesh_surface_distance(candidate_mesh, candidate.fitted_mesh),
                )
            score = candidate.collision_score + candidate.fit_distance + candidate.support_error
            if score <= best_score:
                best_score = score
                current_box = proposal
                fitted = candidate
        return RefinementResult(fitted=fitted, refined_box=current_box, iterations=self.steps)
