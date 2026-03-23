from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .config import DataConfig


REGIME_TO_BEHAVIOR = {
    # The model predicts both "regime" and "behavior".
    # In this toy dataset we derive behavior from regime so the two targets stay consistent.
    0: 0,  # locomotion -> walking
    1: 1,  # transition
    2: 2,  # stationary use
    3: 4,  # resting
}


AFFORDANCE_BY_REGIME = {
    # Affordances describe what the nearby scene should allow the body to do.
    # Example: walking suggests "walkable", resting suggests "rest surface".
    0: 0,  # walkable
    1: 1,  # support transition
    2: 3,  # reachable/use
    3: 2,  # rest surface
}


TRANSITION_LABELS = {
    "none": 0,
    "sit": 1,
    "stand": 2,
    "lie": 3,
    "rise": 4,
}


def set_seed(seed: int) -> None:
    """Keep synthetic data generation reproducible across runs."""
    random.seed(seed)
    torch.manual_seed(seed)


def generate_split(config: DataConfig, size: int, seed: int) -> List[Dict[str, torch.Tensor]]:
    """Create one synthetic split with repeated scene templates for memory training."""
    set_seed(seed)
    return [generate_sample(config, seed + idx, idx % config.scene_templates) for idx in range(size)]


def generate_sample(config: DataConfig, seed: int, scene_id: int) -> Dict[str, torch.Tensor]:
    """Generate one full synthetic training sample.

    Think of one sample as one short recording window.

    It contains:
    - synthetic watch/phone/barometer sensor streams
    - frame-wise labels like steps, contacts, transitions, and behavior
    - scene-level targets like object box, room graph, and occupancy descriptor
    - a scene_id so different samples can pretend to come from repeated visits
    """
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    seq_len = config.seq_len
    # scene_bias is a tiny deterministic variation that makes each synthetic scene template
    # look slightly different. This helps the memory/retrieval heads learn something structured
    # instead of seeing identical targets every time.
    scene_bias = scene_id / max(config.scene_templates - 1, 1)

    # Allocate all tensors up front.
    # This makes the sample layout explicit and keeps the loop below easier to read.
    sensor = torch.zeros(seq_len, config.sensor_dim)
    regime = torch.zeros(seq_len, dtype=torch.long)
    step = torch.zeros(seq_len)
    heading = torch.zeros(seq_len)
    stride = torch.zeros(seq_len, 2)
    floor_event = torch.zeros(seq_len, dtype=torch.long)
    baro_delta = torch.zeros(seq_len)
    transition_cls = torch.zeros(seq_len, dtype=torch.long)
    transition_root = torch.zeros(seq_len, 3)
    transition_torso = torch.zeros(seq_len)
    contacts = torch.zeros(seq_len, 4)
    reach = torch.zeros(seq_len, config.reach_dim)
    affordance = torch.zeros(seq_len, dtype=torch.long)
    pose = torch.zeros(seq_len, config.pose_dim)
    joints = torch.zeros(seq_len, config.joint_dim)
    root = torch.zeros(seq_len, 3)
    behavior = torch.zeros(seq_len, dtype=torch.long)
    uncertainty = torch.zeros(seq_len, 1)

    # current_heading/current_root are the hidden state of the synthetic person.
    # We update them as we move through time and then derive labels from them.
    current_heading = rng.uniform(-math.pi, math.pi)
    current_root = torch.tensor([scene_bias * 0.5, -scene_bias * 0.3, 1.0 + 0.1 * scene_bias])

    # Instead of switching regime randomly every frame, we create short blocks.
    # That produces more realistic temporal structure such as "walk for a while, then sit".
    regime_blocks = []
    while sum(length for _, length in regime_blocks) < seq_len:
        remaining = seq_len - sum(length for _, length in regime_blocks)
        block_len = min(rng.randint(8, 18), remaining)
        block_regime = rng.choices([0, 1, 2, 3], weights=[0.35, 0.2, 0.3, 0.15])[0]
        regime_blocks.append((block_regime, block_len))

    t0 = 0
    for block_regime, block_len in regime_blocks:
        for local_t in range(block_len):
            t = t0 + local_t
            regime[t] = block_regime
            behavior[t] = REGIME_TO_BEHAVIOR[block_regime]
            affordance[t] = AFFORDANCE_BY_REGIME[block_regime]

            # phase gives us a smooth periodic signal inside a block.
            # We reuse it to synthesize gait-like or transition-like patterns.
            phase = 2 * math.pi * (local_t / max(block_len, 1))
            noise_watch = 0.05 * torch.randn(config.imu_watch_dim, generator=gen)
            noise_phone = 0.05 * torch.randn(config.imu_phone_dim, generator=gen)

            if block_regime == 0:
                # Locomotion examples are where we create the strongest motion signals:
                # steps, heading drift, horizontal stride, and occasional floor changes.
                current_heading += rng.uniform(-0.08, 0.08)
                step[t] = 1.0 if local_t % 2 == 0 else 0.0
                stride_mag = 0.45 + 0.1 * math.sin(phase)
                stride[t] = torch.tensor(
                    [stride_mag * math.cos(current_heading), stride_mag * math.sin(current_heading)]
                )
                height_change = rng.choice([-1, 0, 1]) if local_t == block_len // 2 else 0
                floor_event[t] = height_change + 1
                baro_delta[t] = 0.12 * height_change + rng.uniform(-0.01, 0.01)
                # Foot contact is active while moving in this toy setup.
                contacts[t, 0] = 1.0
                reach[t] = torch.tensor([0.2, 0.1, 0.0])
                watch = torch.tensor(
                    [math.sin(phase), math.cos(phase), 0.4, 0.2, current_heading / math.pi, step[t]]
                )
                phone = torch.tensor([0.8 * math.sin(phase), 0.6 * math.cos(phase), 0.2, 0.1, stride_mag + scene_bias, baro_delta[t]])
            elif block_regime == 1:
                # Transition windows model short actions like sitting down or standing up.
                # They change torso orientation and support, but usually not long-distance travel.
                transition_name = rng.choice(["sit", "stand", "lie", "rise"])
                transition_cls[t] = TRANSITION_LABELS[transition_name]
                root_delta_z = {"sit": -0.04, "stand": 0.04, "lie": -0.02, "rise": 0.03}[transition_name]
                transition_root[t] = torch.tensor([0.02 * math.sin(phase), 0.01 * math.cos(phase), root_delta_z])
                transition_torso[t] = {"sit": -0.6, "stand": 0.5, "lie": -1.0, "rise": 0.8}[transition_name]
                contacts[t] = torch.tensor(
                    [0.5 if transition_name in {"stand", "rise"} else 0.0, 0.0, 0.7, 0.3]
                )
                reach[t] = torch.tensor([0.5, 0.2, 0.1])
                baro_delta[t] = root_delta_z * 0.4 + rng.uniform(-0.01, 0.01)
                watch = torch.tensor([0.3, -0.2, transition_torso[t], 0.4, 0.1, 0.0])
                phone = torch.tensor([0.2, 0.1, transition_root[t, 2], 0.3, 0.0, baro_delta[t]])
            elif block_regime == 2:
                # Stationary-use windows imitate activities like using a desk or interacting
                # with something nearby. Reach is larger and hand contact appears more often.
                contacts[t] = torch.tensor([1.0, 0.2 + 0.4 * (local_t % 3 == 0), 0.4, 0.0])
                reach[t] = torch.tensor([0.8, 0.6, 0.3 + 0.1 * scene_bias])
                transition_cls[t] = 0
                watch = torch.tensor([0.1, 0.0, 0.0, 0.2, 0.7, 0.1])
                phone = torch.tensor([0.0, 0.1, 0.0, 0.1, 0.3, 0.0])
            else:
                # Resting keeps the body mostly still and shifts supervision toward
                # pelvis/back support instead of steps or reach.
                contacts[t] = torch.tensor([0.0, 0.0, 0.9, 0.8])
                reach[t] = torch.tensor([0.1, 0.0, 0.0])
                transition_cls[t] = 0
                watch = torch.tensor([0.0, 0.0, -0.2, 0.0, 0.0, 0.0])
                phone = torch.tensor([0.0, 0.0, -0.1, 0.0, 0.0, 0.0])
                uncertainty[t] = 0.2

            heading[t] = current_heading
            # This is our synthetic trajectory integrator.
            # Walking contributes stride/barometer deltas, transitions add short root changes.
            current_root = current_root + torch.tensor([stride[t, 0], stride[t, 1], baro_delta[t]])
            current_root = current_root + transition_root[t]
            root[t] = current_root

            # Pose/joint targets are not physically exact human poses.
            # They are just structured targets correlated with motion state so the model learns
            # multi-task behavior instead of unrelated random numbers.
            pose[t] = torch.cat(
                [
                    torch.sin(torch.linspace(0, 1, config.pose_dim // 2) + heading[t]),
                    torch.cos(torch.linspace(0, 1, config.pose_dim - config.pose_dim // 2) + transition_torso[t]),
                ]
            )[: config.pose_dim]
            joint_base = torch.linspace(0, 1, config.joint_dim)
            joints[t] = joint_base * (0.5 + contacts[t].mean()) + root[t].mean() * 0.05
            uncertainty[t] = uncertainty[t] + 0.05 * (block_regime == 1) + 0.02 * torch.rand(1, generator=gen)
            sensor[t] = torch.cat([watch + noise_watch, phone + noise_phone, baro_delta[t].view(1)])

        t0 += block_len

    # After the whole time sequence is built, we summarize it into scene-level targets.
    # Those feed the geometry/semantic/topology heads in the model.
    scene_features = build_scene_targets(config, regime, contacts, reach, root, floor_event, affordance, scene_id, scene_bias)
    return {
        "sensor": sensor.float(),
        "regime": regime,
        "step": step.unsqueeze(-1),
        "heading": heading.unsqueeze(-1),
        "stride": stride.float(),
        "floor_event": floor_event,
        "baro": baro_delta.unsqueeze(-1),
        "transition_cls": transition_cls,
        "transition_root": transition_root.float(),
        "transition_torso": transition_torso.unsqueeze(-1).float(),
        "contacts": contacts.float(),
        "reach": reach.float(),
        "affordance": affordance,
        "pose": pose.float(),
        "joints": joints.float(),
        "root": root.float(),
        "behavior": behavior,
        "uncertainty": uncertainty.float(),
        "scene_id": torch.tensor(scene_id, dtype=torch.long),
        **scene_features,
    }


def build_scene_targets(
    config: DataConfig,
    regime: torch.Tensor,
    contacts: torch.Tensor,
    reach: torch.Tensor,
    root: torch.Tensor,
    floor_event: torch.Tensor,
    affordance: torch.Tensor,
    scene_id: int,
    scene_bias: float,
) -> Dict[str, torch.Tensor]:
    """Collapse frame-level motion cues into scene-level supervision.

    These are compact targets, not full real-world scene annotations.

    The goal is to give the prototype something coherent to predict:
    - geometry-like descriptors
    - object/category targets
    - topology hints
    - retrieval/refinement targets
    - memory targets that repeat across samples from the same synthetic scene
    """
    pooled_regime = torch.bincount(regime, minlength=config.regime_classes).float() / len(regime)
    mean_contacts = contacts.mean(dim=0)
    mean_root = root.mean(dim=0)
    path_extent = root[:, :2].max(dim=0).values - root[:, :2].min(dim=0).values
    stairs_present = int((floor_event != 1).any().item())

    # In a full system, occupancy might be a voxel grid or implicit field.
    # Here we compress scene evidence into a short vector to keep training simple.
    occupancy = torch.cat([pooled_regime, mean_contacts, reach.mean(dim=0), mean_root, path_extent])[
        : config.occupancy_dim
    ]
    if occupancy.numel() < config.occupancy_dim:
        occupancy = torch.cat([occupancy, torch.zeros(config.occupancy_dim - occupancy.numel())])

    # These are tiny geometric summaries standing in for larger scene representations.
    floor_plane = torch.tensor([0.0, 0.0, 1.0, root[:, 2].min().item()])
    wall = torch.tensor([path_extent[0].item(), path_extent[1].item(), mean_root[0].item(), mean_root[1].item()])
    support_plane = torch.tensor(
        [mean_contacts[2].item(), mean_contacts[3].item(), root[:, 2].median().item(), contacts[:, 0].mean().item()]
    )
    object_box = torch.tensor(
        [
            path_extent[0].item() + 0.5,
            path_extent[1].item() + 0.5,
            0.8 + 0.4 * mean_contacts[2].item(),
            mean_root[0].item(),
            mean_root[1].item(),
            root[:, 2].mean().item(),
            root[:, 0].mean().item() * 0.1,
        ]
    )
    object_class = torch.tensor((scene_id + int(torch.mode(affordance).values.item())) % config.object_classes)
    semantic = object_class.clone()
    # Interaction zone is just the average reach volume proxy.
    interaction_zone = reach.mean(dim=0)
    room_graph = torch.zeros(config.room_graph_dim)
    room_graph[0] = 1.0
    room_graph[1] = float(path_extent[0] > 2.0)
    room_graph[2] = float(path_extent[1] > 2.0)
    room_graph[3] = float(stairs_present)
    room_graph[4:7] = pooled_regime[:3]
    door = torch.tensor(int(path_extent.norm().item() > 3.0))
    stairs = torch.tensor(2 if stairs_present else 1)
    topo_transition = torch.tensor(2 if stairs_present else 1)
    support_height = torch.tensor([root[:, 2].mean().item() - 0.45 * mean_contacts[2].item() + 0.05 * scene_bias])
    free_space = 1.0 - occupancy
    # "best_scene" is a compact scene signature used by the multi-hypothesis and memory losses.
    best_scene = torch.cat([occupancy[:8], room_graph[:8], object_box[:8]])[:8]
    if best_scene.numel() < 8:
        best_scene = torch.cat([best_scene, torch.zeros(8 - best_scene.numel())])
    # paired_memory_target gives another scene-consistent target for the memory head.
    paired_memory_target = best_scene.float() * (0.9 + 0.05 * scene_bias)

    # retrieval_embedding is a compact target used by the retrieval/refinement side.
    retrieval_embedding = torch.cat([object_box[:4], support_plane[:3]])[:7]
    return {
        "occupancy": occupancy.float(),
        "floor_plane": floor_plane.float(),
        "wall": wall.float(),
        "support_plane": support_plane.float(),
        "object_box": object_box.float(),
        "object_class": object_class.long(),
        "semantic": semantic.long(),
        "interaction_zone": interaction_zone.float(),
        "room_graph": room_graph.float(),
        "door": door.long(),
        "stairs": stairs.long(),
        "topology_transition": topo_transition.long(),
        "support_height": support_height.float(),
        "free_space": free_space.float(),
        "best_scene": best_scene.float(),
        "memory_target": best_scene.float() * 0.8 + 0.1,
        "paired_memory_target": paired_memory_target.float(),
        "retrieval_target": object_box.float(),
        "retrieval_embedding": retrieval_embedding.float(),
    }


class B2SDataset(Dataset):
    """Thin wrapper around a list of pre-generated tensor dictionaries."""

    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


def save_dataset(config: DataConfig, output_dir: str, seed: int = 7) -> Dict[str, Path]:
    """Generate and save the synthetic dataset splits to disk.

    The output files are plain torch `.pt` files containing a list of samples.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": generate_split(config, config.train_size, seed),
        "val": generate_split(config, config.val_size, seed + 10_000),
        "test": generate_split(config, config.test_size, seed + 20_000),
    }
    paths = {}
    for split_name, samples in splits.items():
        path = output_path / f"{split_name}.pt"
        torch.save(samples, path)
        paths[split_name] = path
    return paths


def load_dataset(split_path: str) -> B2SDataset:
    """Load one saved split into the dataset wrapper."""
    samples = torch.load(split_path)
    return B2SDataset(samples)
