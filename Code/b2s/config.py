from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Shape and label-space configuration shared by the dataset and model."""

    # Temporal setup for one training example.
    seq_len: int = 64

    # Synthetic split sizes used by the prototype generator.
    train_size: int = 512
    val_size: int = 128
    test_size: int = 128

    # Sensor feature dimensions: 3 accel + 3 gyro for each IMU, plus barometer.
    imu_watch_dim: int = 6
    imu_phone_dim: int = 6
    baro_dim: int = 1

    # Output spaces for downstream heads.
    reach_dim: int = 3
    pose_dim: int = 32
    joint_dim: int = 45
    occupancy_dim: int = 16
    room_graph_dim: int = 9
    object_classes: int = 6
    affordance_classes: int = 5
    behavior_classes: int = 6
    regime_classes: int = 4
    floor_event_classes: int = 3
    transition_classes: int = 5
    topology_transition_classes: int = 3
    scene_hypotheses: int = 3
    scene_templates: int = 12

    @property
    def sensor_dim(self) -> int:
        """Total fused sensor width presented to the model."""
        return self.imu_watch_dim + self.imu_phone_dim + self.baro_dim


@dataclass
class TrainConfig:
    """Optimization and runtime knobs for training."""

    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    device: str = "cpu"
    seed: int = 7
    log_every: int = 10
    dataset_dir: str = "Code/data"
    weights: dict = field(
        default_factory=lambda: {
            # Stage weights loosely follow the architecture blocks in arcitecture.md.
            "regime": 1.0,
            "locomotion": 1.0,
            "transition": 0.8,
            "contact": 0.8,
            "motion": 1.0,
            "constraints": 0.6,
            "scene": 0.8,
            "multi_hypothesis": 0.5,
            "refine": 0.4,
            "memory": 0.3,
            "retrieval": 0.3,
            "physics": 0.4,
        }
    )
