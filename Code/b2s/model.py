from __future__ import annotations

import torch
from torch import nn

from .config import DataConfig, TrainConfig


class TemporalEncoder(nn.Module):
    """Shared temporal backbone for all downstream branches."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First project raw sensor features into the model's hidden size,
        # then let the bidirectional GRU build temporal context.
        x = torch.relu(self.proj(x))
        x, _ = self.gru(x)
        return self.out(x)


class TimeHead(nn.Module):
    """Per-frame prediction head used for temporal labels."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This head preserves the time dimension, so it is used for labels
        # that exist at every frame such as steps, contacts, and pose.
        return self.net(x)


class SequenceHead(nn.Module):
    """Sequence-level head that pools over time before prediction."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A simple mean-pool is enough for this prototype.
        # In a stronger model, this could become attention or learned pooling.
        return self.net(x.mean(dim=1))


class B2SModel(nn.Module):
    """Multi-branch prototype model that mirrors the architecture document."""

    def __init__(self, data_config: DataConfig, train_config: TrainConfig):
        super().__init__()
        h = train_config.hidden_dim
        self.data_config = data_config
        self.encoder = TemporalEncoder(data_config.sensor_dim, h)

        # Activity parser.
        self.regime_head = TimeHead(h, data_config.regime_classes)

        # Motion-specialized branches.
        # These branches all read from the same backbone but predict different aspects
        # of the body state. This is the model version of the architecture diagram.
        self.locomotion_head = nn.ModuleDict(
            {
                "step": TimeHead(h, 1),
                "heading": TimeHead(h, 1),
                "stride": TimeHead(h, 2),
                "floor_event": TimeHead(h, data_config.floor_event_classes),
                "baro": TimeHead(h, 1),
            }
        )
        self.transition_head = nn.ModuleDict(
            {
                "cls": TimeHead(h, data_config.transition_classes),
                "root": TimeHead(h, 3),
                "torso": TimeHead(h, 1),
            }
        )
        self.contact_head = nn.ModuleDict(
            {
                "contacts": TimeHead(h, 4),
                "reach": TimeHead(h, data_config.reach_dim),
                "affordance": TimeHead(h, data_config.affordance_classes),
            }
        )

        fusion_input_dim = h * 4
        # motion_fusion combines shared temporal features with branch-specific cues
        # before predicting the richer human state outputs.
        self.motion_fusion = nn.Linear(fusion_input_dim, h)
        self.motion_heads = nn.ModuleDict(
            {
                "pose": TimeHead(h, data_config.pose_dim),
                "joints": TimeHead(h, data_config.joint_dim),
                "root": TimeHead(h, 3),
                "behavior": TimeHead(h, data_config.behavior_classes),
                "uncertainty": TimeHead(h, 1),
            }
        )

        # Heads that translate motion features into scene constraints and scene predictions.
        self.constraint_heads = nn.ModuleDict(
            {
                "free_space": SequenceHead(h, data_config.occupancy_dim),
                "free_space_grid": SequenceHead(h, data_config.occupancy_grid_dim),
                "support_grid": SequenceHead(h, data_config.occupancy_grid_dim),
                "reachability_grid": SequenceHead(h, data_config.occupancy_grid_dim),
                "topology_cues": SequenceHead(h, 4),
                "support_height": SequenceHead(h, 1),
            }
        )
        self.scene_heads = nn.ModuleDict(
            {
                # Geometry-like descriptors.
                "occupancy": SequenceHead(h, data_config.occupancy_dim),
                "occupancy_grid": SequenceHead(h, data_config.occupancy_grid_dim),
                "floor_plane": SequenceHead(h, 4),
                "wall": SequenceHead(h, 4),
                "support_plane": SequenceHead(h, 4),
                "object_box": SequenceHead(h, 7),
                # Semantic and topology descriptors.
                "object_class": SequenceHead(h, data_config.object_classes),
                "semantic": SequenceHead(h, data_config.object_classes),
                "interaction_zone": SequenceHead(h, data_config.reach_dim),
                "room_graph": SequenceHead(h, data_config.room_graph_dim),
                "door": SequenceHead(h, 2),
                "stairs": SequenceHead(h, data_config.floor_event_classes),
                "topology_transition": SequenceHead(h, data_config.topology_transition_classes),
            }
        )

        # hypothesis_dim is the reduced scene signature size used by best-of-K training.
        hypothesis_dim = 8
        self.hypothesis_head = SequenceHead(h, data_config.scene_hypotheses * hypothesis_dim)
        self.hypothesis_score = SequenceHead(h, data_config.scene_hypotheses)
        self.refine_head = SequenceHead(h, hypothesis_dim)
        self.memory_head = SequenceHead(h, hypothesis_dim)
        self.retrieval_head = SequenceHead(h, 7)

        # Extra heads added after the initial prototype:
        # - mesh_refine nudges predicted boxes before fitting assets
        # - memory_embedding supports cross-visit alignment losses
        # - physics heads predict compact consistency energies
        self.mesh_refine_head = SequenceHead(h, 7)
        self.memory_embedding_head = SequenceHead(h, hypothesis_dim)
        self.physics_head = nn.ModuleDict(
            {
                "support_energy": SequenceHead(h, 1),
                "collision_energy": SequenceHead(h, 1),
                "stability_energy": SequenceHead(h, 1),
            }
        )

    def forward(self, sensor: torch.Tensor) -> dict:
        """Run the full model and return a structured dict of branch outputs."""
        # 1. Encode the fused sensor sequence once.
        encoded = self.encoder(sensor)
        regime_logits = self.regime_head(encoded)

        # 2. Predict branch-specific signals from the shared representation.
        locomotion = {name: head(encoded) for name, head in self.locomotion_head.items()}
        transition = {name: head(encoded) for name, head in self.transition_head.items()}
        contact = {name: head(encoded) for name, head in self.contact_head.items()}

        # Build a simple fused representation from the shared backbone and branch summaries.
        fused = torch.cat(
            [
                encoded,
                locomotion["stride"].repeat(1, 1, encoded.shape[-1] // 2),
                transition["root"].repeat(1, 1, encoded.shape[-1] // 3 + 1)[:, :, : encoded.shape[-1]],
                contact["contacts"].repeat(1, 1, encoded.shape[-1] // 4),
            ],
            dim=-1,
        )
        fused = torch.relu(self.motion_fusion(fused))

        # 3. Predict the richer human motion state from the fused representation.
        motion = {name: head(fused) for name, head in self.motion_heads.items()}

        # 4. Predict scene-related outputs from the sequence as a whole.
        constraints = {name: head(encoded) for name, head in self.constraint_heads.items()}
        scene = {name: head(encoded) for name, head in self.scene_heads.items()}

        batch_size = sensor.shape[0]
        hypo_raw = self.hypothesis_head(encoded).view(batch_size, self.data_config.scene_hypotheses, -1)
        return {
            # The output dict keeps the branches separate so the loss function can stay readable.
            "regime_logits": regime_logits,
            "locomotion": locomotion,
            "transition": transition,
            "contact": contact,
            "motion": motion,
            "constraints": constraints,
            "scene": scene,
            "hypotheses": hypo_raw,
            "hypothesis_scores": self.hypothesis_score(encoded),
            "refined_scene": self.refine_head(encoded),
            "memory": self.memory_head(encoded),
            "retrieval": self.retrieval_head(encoded),
            "mesh_refine": self.mesh_refine_head(encoded),
            "memory_embedding": self.memory_embedding_head(encoded),
            "physics": {name: head(encoded) for name, head in self.physics_head.items()},
        }
