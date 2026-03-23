from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class MemoryEntry:
    """Accumulated scene memory for one scene or visit cluster."""

    fused_scene: torch.Tensor
    confidence: float
    visits: int = 1


@dataclass
class PersistentSceneMemory:
    """In-memory scene fusion store used for repeated visits."""

    entries: Dict[int, MemoryEntry] = field(default_factory=dict)

    def update(self, scene_id: int, scene_signature: torch.Tensor, confidence: float) -> MemoryEntry:
        if scene_id not in self.entries:
            self.entries[scene_id] = MemoryEntry(scene_signature.detach().clone(), confidence)
            return self.entries[scene_id]
        entry = self.entries[scene_id]
        alpha = min(max(confidence, 0.1), 0.9)
        entry.fused_scene = (1 - alpha) * entry.fused_scene + alpha * scene_signature.detach()
        entry.confidence = max(entry.confidence, confidence)
        entry.visits += 1
        return entry

    def query(self, scene_id: int) -> torch.Tensor | None:
        entry = self.entries.get(scene_id)
        return entry.fused_scene if entry else None
