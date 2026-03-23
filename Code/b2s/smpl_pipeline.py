from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class MotionReconstructionResult:
    """Normalized 4D body-state result returned by an external reconstructor."""

    pose_body: Any
    root_world: Any
    trans_world: Any
    verts_world: Any
    metadata: Dict[str, Any]


class WHAMMotionReconstructor:
    """Adapter around the local WHAM checkout for 4D human reconstruction."""

    def __init__(self, wham_root: str = "WHAM"):
        self.wham_root = Path(wham_root).resolve()

    def reconstruct_from_video(self, video_path: str, output_dir: str) -> Dict[str, MotionReconstructionResult]:
        sys.path.insert(0, str(self.wham_root))
        try:
            from wham_api import WHAM_API
        except Exception as exc:
            raise RuntimeError("WHAM dependencies are not fully available in the current environment") from exc
        api = WHAM_API()
        results, tracking_results, slam_results = api(video_path, output_dir=output_dir)
        normalized = {}
        for track_id, result in results.items():
            normalized[track_id] = MotionReconstructionResult(
                pose_body=result.get("poses_body"),
                root_world=result.get("poses_root_world"),
                trans_world=result.get("trans_world"),
                verts_world=result.get("verts_cam"),
                metadata={"frame_id": result.get("frame_id"), "tracking": tracking_results, "slam": slam_results},
            )
        return normalized
