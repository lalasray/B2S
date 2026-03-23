from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .mesh_utils import (
    Mesh,
    approximate_collision_score,
    create_bed_mesh,
    create_box,
    create_chair_mesh,
    create_room_mesh,
    create_sofa_mesh,
    create_stairs_mesh,
    create_table_mesh,
    load_obj,
    mesh_surface_distance,
    snap_to_support,
    support_plane_gap,
    transform_mesh,
    write_obj,
)
from .retrieval import AssetProvider, LocalAssetProvider


OBJECT_LABELS = {
    # Maps classifier indices to asset categories used by retrieval.
    0: "chair",
    1: "table",
    2: "bed",
    3: "sofa",
    4: "stairs",
    5: "storage",
}


@dataclass
class RetrievedAsset:
    """Result bundle after retrieval, fitting, and consistency scoring."""

    label: str
    source_path: Path
    fitted_mesh: Mesh
    support_error: float
    collision_score: float
    fit_distance: float


class AssetLibrary:
    """Owns the default local assets and delegates path resolution to a provider."""

    def __init__(self, root: str = "Code/assets/library", provider: Optional[AssetProvider] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.provider = provider or LocalAssetProvider(str(self.root))

    def ensure_default_assets(self) -> Dict[str, Path]:
        """Generate the built-in procedural assets if they are missing."""
        builders = {
            "chair": create_chair_mesh,
            "table": create_table_mesh,
            "bed": create_bed_mesh,
            "sofa": create_sofa_mesh,
            "stairs": create_stairs_mesh,
            "storage": create_table_mesh,
            "room": create_room_mesh,
        }
        created = {}
        for label, builder in builders.items():
            path = self.root / f"{label}.obj"
            if not path.exists():
                write_obj(builder(), str(path))
            created[label] = path
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps({label: str(path) for label, path in created.items()}, indent=2))
        return created

    def get_asset_path(self, label: str) -> Path:
        """Resolve a label to a concrete OBJ path through the configured provider."""
        self.ensure_default_assets()
        path = Path(self.provider.resolve(label).path)
        if not path.exists():
            raise FileNotFoundError(f"No asset found for label {label}")
        return path

    def load_asset(self, label: str) -> Mesh:
        """Load the resolved mesh file into the local Mesh wrapper."""
        return load_obj(str(self.get_asset_path(label)))


def fit_asset_to_box(asset: Mesh, box_params: torch.Tensor, support_z: Optional[float] = None) -> Mesh:
    """Scale and place an asset according to the predicted oriented box."""
    # box_params follow the convention:
    # [extent_x, extent_y, extent_z, center_x, center_y, center_z, yaw]
    extents = torch.clamp(box_params[:3].abs(), min=0.2)
    center = box_params[3:6]
    yaw = float(box_params[6].item())
    base_extents = torch.clamp(asset.extents, min=1e-3)
    scale_xyz = extents / base_extents
    fitted = transform_mesh(asset, scale_xyz, yaw, center)
    if support_z is not None:
        # Snap the fitted object so its base actually rests on the support plane.
        fitted = snap_to_support(fitted, support_z)
    return fitted


def retrieve_and_fit_asset(
    object_class: int,
    object_box: torch.Tensor,
    support_plane: torch.Tensor,
    library: AssetLibrary,
    environment_mesh: Optional[Mesh] = None,
) -> RetrievedAsset:
    """Retrieve an asset, fit it to the predicted box, and score the result."""
    # 1. Convert the predicted class index into a human-readable asset label.
    label = OBJECT_LABELS.get(int(object_class), "chair")
    asset = library.load_asset(label)
    support_z = float(support_plane[2].item())

    # 2. Fit the retrieved asset to the predicted placement.
    fitted = fit_asset_to_box(asset, object_box, support_z=support_z)
    # Use a simple box proxy as the geometric target for refinement/fitting distance.
    target_proxy = transform_mesh(create_box(object_box[:3].abs().tolist(), f"{label}_target_proxy"), torch.ones(3), float(object_box[6].item()), object_box[3:6])
    target_proxy = snap_to_support(target_proxy, support_z)
    support_error = support_plane_gap(fitted, support_z)
    collision_score = approximate_collision_score(fitted, environment_mesh) if environment_mesh is not None else 0.0
    fit_distance = mesh_surface_distance(fitted, target_proxy)
    return RetrievedAsset(
        label=label,
        source_path=library.get_asset_path(label),
        fitted_mesh=fitted,
        support_error=support_error,
        collision_score=collision_score,
        fit_distance=fit_distance,
    )


def export_scene_bundle(
    output_dir: str,
    fitted_asset: RetrievedAsset,
    environment_mesh: Optional[Mesh] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Export the fitted object, optional environment mesh, and a JSON summary."""
    # Keeping exports together makes it easier to inspect one inference sample by folder.
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    asset_path = write_obj(fitted_asset.fitted_mesh, str(out / f"{fitted_asset.label}_fitted.obj"))
    outputs = {"asset": asset_path}
    if environment_mesh is not None:
        outputs["environment"] = write_obj(environment_mesh, str(out / "environment.obj"))
    meta_path = out / "scene_summary.json"
    summary = {
        "label": fitted_asset.label,
        "source_path": str(fitted_asset.source_path),
        "support_error": fitted_asset.support_error,
        "collision_score": fitted_asset.collision_score,
        "fit_distance": fitted_asset.fit_distance,
    }
    if metadata:
        summary.update(metadata)
    meta_path.write_text(json.dumps(summary, indent=2))
    outputs["summary"] = meta_path
    return outputs
