from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from pytorch3d.io import load_obj as p3d_load_obj
from pytorch3d.io import save_obj
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene


@dataclass
class Mesh:
    """Lightweight mesh container used across retrieval and fitting code."""

    vertices: torch.Tensor
    faces: torch.Tensor
    name: str = "mesh"

    def clone(self) -> "Mesh":
        """Return a detached copy so callers can transform meshes safely."""
        return Mesh(self.vertices.clone(), self.faces.clone(), self.name)

    def to(self, device: str | torch.device) -> "Mesh":
        """Move vertices/faces together onto a target device."""
        return Mesh(self.vertices.to(device), self.faces.to(device), self.name)

    def to_pytorch3d(self, device: str | torch.device | None = None) -> Meshes:
        """Convert to the PyTorch3D Meshes structure on demand."""
        verts = self.vertices if device is None else self.vertices.to(device)
        faces = self.faces if device is None else self.faces.to(device)
        return Meshes(verts=[verts], faces=[faces])

    @classmethod
    def from_pytorch3d(cls, mesh: Meshes, name: str = "mesh") -> "Mesh":
        return cls(mesh.verts_list()[0], mesh.faces_list()[0], name=name)

    @property
    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Axis-aligned min/max bounds used by simple geometric heuristics."""
        return self.vertices.min(dim=0).values, self.vertices.max(dim=0).values

    @property
    def extents(self) -> torch.Tensor:
        """Size of the mesh along each axis."""
        low, high = self.bounds
        return high - low


def rotation_matrix_z(yaw: float, device: torch.device | None = None) -> torch.Tensor:
    """Create a yaw-only rotation because the prototype uses box yaw as orientation."""
    c = math.cos(yaw)
    s = math.sin(yaw)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)


def transform_mesh(mesh: Mesh, scale_xyz: torch.Tensor, yaw: float, translation: torch.Tensor) -> Mesh:
    """Apply scale, yaw rotation, and translation to a mesh."""
    centered = mesh.vertices - mesh.vertices.mean(dim=0, keepdim=True)
    scaled = centered * scale_xyz.view(1, 3).to(centered.device)
    rotated = scaled @ rotation_matrix_z(yaw, centered.device).T
    translated = rotated + translation.view(1, 3).to(centered.device)
    return Mesh(translated, mesh.faces.clone(), name=mesh.name)


def write_obj(mesh: Mesh, path: str) -> Path:
    """Save a mesh as OBJ using PyTorch3D's writer."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_obj(str(output), mesh.vertices.cpu(), mesh.faces.cpu())
    return output


def load_obj(path: str, device: str | torch.device = "cpu") -> Mesh:
    """Load an OBJ mesh into the local Mesh wrapper."""
    verts, faces, _ = p3d_load_obj(path, device=device, load_textures=False)
    return Mesh(verts.float(), faces.verts_idx.long(), Path(path).stem)


def create_box(extents: Iterable[float], name: str) -> Mesh:
    """Create a simple box primitive used for scene proxies and default assets."""
    ex, ey, ez = [float(v) for v in extents]
    hx, hy, hz = ex / 2.0, ey / 2.0, ez / 2.0
    vertices = torch.tensor(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.long,
    )
    return Mesh(vertices, faces, name=name)


def merge_meshes(meshes: List[Mesh], name: str) -> Mesh:
    """Merge several mesh parts into one scene mesh."""
    # This is how we turn primitive parts like chair legs + seat into one object mesh.
    scene = join_meshes_as_scene(Meshes(verts=[mesh.vertices for mesh in meshes], faces=[mesh.faces for mesh in meshes]))
    return Mesh.from_pytorch3d(scene, name=name)


def create_chair_mesh() -> Mesh:
    """Build a simple procedural chair asset."""
    seat = transform_mesh(create_box((0.6, 0.6, 0.08), "chair_seat"), torch.ones(3), 0.0, torch.tensor([0.0, 0.0, 0.45]))
    back = transform_mesh(create_box((0.6, 0.08, 0.7), "chair_back"), torch.ones(3), 0.0, torch.tensor([0.0, -0.26, 0.75]))
    legs = []
    for sx in (-0.24, 0.24):
        for sy in (-0.24, 0.24):
            legs.append(transform_mesh(create_box((0.08, 0.08, 0.45), "chair_leg"), torch.ones(3), 0.0, torch.tensor([sx, sy, 0.225])))
    return merge_meshes([seat, back, *legs], "chair")


def create_table_mesh() -> Mesh:
    """Build a simple procedural table asset."""
    top = transform_mesh(create_box((1.1, 0.7, 0.08), "table_top"), torch.ones(3), 0.0, torch.tensor([0.0, 0.0, 0.72]))
    legs = []
    for sx in (-0.46, 0.46):
        for sy in (-0.26, 0.26):
            legs.append(transform_mesh(create_box((0.08, 0.08, 0.72), "table_leg"), torch.ones(3), 0.0, torch.tensor([sx, sy, 0.36])))
    return merge_meshes([top, *legs], "table")


def create_bed_mesh() -> Mesh:
    """Build a simple procedural bed asset."""
    base = transform_mesh(create_box((2.0, 1.5, 0.35), "bed_base"), torch.ones(3), 0.0, torch.tensor([0.0, 0.0, 0.175]))
    headboard = transform_mesh(create_box((1.5, 0.08, 0.9), "bed_headboard"), torch.ones(3), 0.0, torch.tensor([0.0, -0.71, 0.45]))
    return merge_meshes([base, headboard], "bed")


def create_sofa_mesh() -> Mesh:
    """Build a simple procedural sofa asset."""
    base = transform_mesh(create_box((1.8, 0.8, 0.45), "sofa_base"), torch.ones(3), 0.0, torch.tensor([0.0, 0.0, 0.225]))
    back = transform_mesh(create_box((1.8, 0.12, 0.8), "sofa_back"), torch.ones(3), 0.0, torch.tensor([0.0, -0.34, 0.65]))
    arms = [
        transform_mesh(create_box((0.12, 0.8, 0.6), "sofa_arm"), torch.ones(3), 0.0, torch.tensor([-0.84, 0.0, 0.3])),
        transform_mesh(create_box((0.12, 0.8, 0.6), "sofa_arm"), torch.ones(3), 0.0, torch.tensor([0.84, 0.0, 0.3])),
    ]
    return merge_meshes([base, back, *arms], "sofa")


def create_stairs_mesh() -> Mesh:
    """Build a short staircase used by the synthetic scene generator."""
    steps = []
    for i in range(4):
        steps.append(transform_mesh(create_box((1.2, 0.35, 0.16), f"step_{i}"), torch.ones(3), 0.0, torch.tensor([0.0, i * 0.35, 0.08 + i * 0.16])))
    return merge_meshes(steps, "stairs")


def create_room_mesh() -> Mesh:
    """Create a minimal room shell for default environment exports."""
    floor = transform_mesh(create_box((6.0, 6.0, 0.05), "room_floor"), torch.ones(3), 0.0, torch.tensor([0.0, 0.0, -0.025]))
    walls = [
        transform_mesh(create_box((6.0, 0.08, 2.6), "wall"), torch.ones(3), 0.0, torch.tensor([0.0, -3.0, 1.3])),
        transform_mesh(create_box((6.0, 0.08, 2.6), "wall"), torch.ones(3), 0.0, torch.tensor([0.0, 3.0, 1.3])),
        transform_mesh(create_box((0.08, 6.0, 2.6), "wall"), torch.ones(3), 0.0, torch.tensor([-3.0, 0.0, 1.3])),
        transform_mesh(create_box((0.08, 6.0, 2.6), "wall"), torch.ones(3), 0.0, torch.tensor([3.0, 0.0, 1.3])),
    ]
    return merge_meshes([floor, *walls], "room")


def aabb_intersection_volume(mesh_a: Mesh, mesh_b: Mesh) -> float:
    """Cheap overlap estimate used alongside point-sampled collision checks."""
    min_a, max_a = mesh_a.bounds
    min_b, max_b = mesh_b.bounds
    overlap_min = torch.maximum(min_a, min_b)
    overlap_max = torch.minimum(max_a, max_b)
    extent = torch.clamp(overlap_max - overlap_min, min=0.0)
    return float(torch.prod(extent).item())


def sample_surface_points(mesh: Mesh, num_samples: int = 2048) -> torch.Tensor:
    """Sample points from the mesh surface for fitting and collision heuristics."""
    samples = sample_points_from_meshes(mesh.to_pytorch3d(), num_samples=num_samples)
    return samples.squeeze(0)


def mesh_surface_distance(mesh_a: Mesh, mesh_b: Mesh, num_samples: int = 2048) -> float:
    """Compute a symmetric surface distance proxy between two meshes."""
    points_a = sample_surface_points(mesh_a, num_samples=num_samples).unsqueeze(0)
    points_b = sample_surface_points(mesh_b, num_samples=num_samples).unsqueeze(0)
    distance, _ = chamfer_distance(points_a, points_b)
    return float(distance.item())


def snap_to_support(mesh: Mesh, support_z: float) -> Mesh:
    """Translate a mesh vertically until its lowest point rests on the support plane."""
    low, _ = mesh.bounds
    translation = torch.tensor([0.0, 0.0, support_z - float(low[2].item())], dtype=torch.float32)
    return Mesh(mesh.vertices + translation.view(1, 3), mesh.faces.clone(), mesh.name)


def support_plane_gap(mesh: Mesh, support_z: float) -> float:
    """Measure how far the lowest mesh point is from the predicted support height."""
    low, _ = mesh.bounds
    return abs(float(low[2].item()) - support_z)


def approximate_collision_score(mesh_a: Mesh, mesh_b: Mesh, num_samples: int = 1024, collision_margin: float = 0.03) -> float:
    """Approximate collision severity using sampled surface proximity plus AABB overlap."""
    # This is not a full physics simulation.
    # It is a practical heuristic: if many points are very close and the boxes overlap,
    # the collision score becomes larger.
    points_a = sample_surface_points(mesh_a, num_samples=num_samples).unsqueeze(0)
    points_b = sample_surface_points(mesh_b, num_samples=num_samples).unsqueeze(0)
    pairwise = torch.cdist(points_a, points_b).squeeze(0)
    close_fraction = (pairwise.min(dim=1).values < collision_margin).float().mean()
    aabb_overlap = aabb_intersection_volume(mesh_a, mesh_b)
    return float(close_fraction.item() + aabb_overlap)
