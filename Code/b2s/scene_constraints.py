from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class SceneConstraints:
    """Voxel and support constraints extracted from predicted motion."""

    occupancy: torch.Tensor
    free_space: torch.Tensor
    support_volume: torch.Tensor
    reachability: torch.Tensor
    topology_cues: torch.Tensor


class SceneConstraintBuilder:
    """Build scene evidence from body trajectories and contacts."""

    def __init__(self, grid_size: int = 12, world_extent: float = 6.0):
        self.grid_size = grid_size
        self.world_extent = world_extent

    def _voxelize_points(self, points: torch.Tensor) -> torch.Tensor:
        coords = ((points + self.world_extent / 2) / self.world_extent * self.grid_size).long()
        coords = coords.clamp(0, self.grid_size - 1)
        grid = torch.zeros(self.grid_size, self.grid_size, self.grid_size, dtype=torch.float32, device=points.device)
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
        return grid

    def swept_body_free_space(self, root: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        """Mark the body trajectory and coarse joint cloud as traversed free-space evidence."""
        joint_points = joints.reshape(-1, 3) if joints.shape[-1] == 3 else root.repeat_interleave(max(joints.shape[-1] // 3, 1), dim=0)
        points = torch.cat([root, joint_points[:, :3]], dim=0)
        return self._voxelize_points(points)

    def support_constraints(self, root: torch.Tensor, contacts: torch.Tensor) -> torch.Tensor:
        """Create a coarse support volume around frames with strong support evidence."""
        supported = root[contacts.mean(dim=-1) > 0.5]
        if supported.numel() == 0:
            supported = root[:1]
        return self._voxelize_points(supported)

    def reachability_volume(self, root: torch.Tensor, reach: torch.Tensor) -> torch.Tensor:
        """Place reach points around the root trajectory."""
        reach_points = root + reach
        return self._voxelize_points(reach_points)

    def topology_from_motion(self, root: torch.Tensor, floor_event: torch.Tensor) -> torch.Tensor:
        """Summarize room-transition / floor-change cues into a compact vector."""
        extent = root[:, :2].max(dim=0).values - root[:, :2].min(dim=0).values
        vertical = root[:, 2].max() - root[:, 2].min()
        stairs = (floor_event != 1).float().mean()
        return torch.tensor([extent[0], extent[1], vertical, stairs], dtype=root.dtype, device=root.device)

    def build(self, root: torch.Tensor, joints: torch.Tensor, contacts: torch.Tensor, reach: torch.Tensor, floor_event: torch.Tensor) -> SceneConstraints:
        free_space = self.swept_body_free_space(root, joints)
        support = self.support_constraints(root, contacts)
        reachability = self.reachability_volume(root, reach)
        occupancy = torch.clamp(support + reachability * 0.5, max=1.0)
        return SceneConstraints(
            occupancy=occupancy,
            free_space=1.0 - free_space,
            support_volume=support,
            reachability=reachability,
            topology_cues=self.topology_from_motion(root, floor_event),
        )
