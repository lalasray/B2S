from __future__ import annotations

import argparse
import json

import torch

from b2s.assets import AssetLibrary, export_scene_bundle, retrieve_and_fit_asset
from b2s.config import DataConfig, TrainConfig
from b2s.dataset import B2SDataset
from b2s.deformation import VertexDeformationField
from b2s.memory_bank import PersistentSceneMemory
from b2s.mesh_utils import create_room_mesh, load_obj
from b2s.model import B2SModel
from b2s.refinement import IterativeSceneRefiner
from b2s.retrieval import make_asset_provider


def parse_args() -> argparse.Namespace:
    """Parse inference and retrieval options."""
    parser = argparse.ArgumentParser(description="Run B2S inference and fit a retrieved mesh asset.")
    parser.add_argument("--checkpoint", type=str, default="Code/checkpoints/best.pt")
    parser.add_argument("--dataset", type=str, default="Code/data/test.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--environment-mesh", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="Code/outputs/sample_0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--asset-source", type=str, default="local", choices=["local", "manifest"])
    parser.add_argument("--asset-manifest", type=str, default="")
    parser.add_argument("--iterative-refine", action="store_true")
    return parser.parse_args()


def load_environment(path: str):
    """Load a user-supplied environment mesh or fall back to the default room shell."""
    if path:
        return load_obj(path)
    return create_room_mesh()


def main() -> None:
    """Run one-sample inference, retrieve an asset, fit it, and export results."""
    args = parse_args()
    # Restore both the learned weights and the saved config objects from training.
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    data_config = ckpt.get("data_config", DataConfig())
    train_config = ckpt.get("train_config", TrainConfig(device=args.device))
    train_config.device = args.device

    model = B2SModel(data_config, train_config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = B2SDataset(torch.load(args.dataset))
    # Pick one sample from the chosen split and add the batch dimension expected by the model.
    sample = dataset[args.sample_index]
    sensor = sample["sensor"].unsqueeze(0).to(args.device)

    with torch.no_grad():
        outputs = model(sensor)

    object_class = int(outputs["scene"]["object_class"].argmax(dim=-1).item())
    # Apply the learned refinement delta before fitting the retrieved asset.
    object_box = (outputs["scene"]["object_box"] + outputs["mesh_refine"]).squeeze(0).cpu()
    support_plane = outputs["scene"]["support_plane"].squeeze(0).cpu()
    environment_mesh = load_environment(args.environment_mesh)

    # Retrieval can come from the checked-in local library or a manifest-backed provider.
    provider = make_asset_provider(args.asset_source, "Code/assets/library", args.asset_manifest or None)
    library = AssetLibrary(provider=provider)
    library.ensure_default_assets()
    if args.iterative_refine:
        refiner = IterativeSceneRefiner(library, deformation_field=VertexDeformationField(latent_dim=8), steps=3)
        refine_result = refiner.refine(object_class, object_box, support_plane, environment_mesh, outputs["memory_embedding"].squeeze(0).cpu())
        fitted = refine_result.fitted
        object_box = refine_result.refined_box
    else:
        fitted = retrieve_and_fit_asset(object_class, object_box, support_plane, library, environment_mesh)

    # Summarize the prediction into a human-readable JSON sidecar.
    regime_pred = outputs["regime_logits"].argmax(dim=-1).squeeze(0).cpu()
    memory = PersistentSceneMemory()
    memory_entry = memory.update(int(sample["scene_id"].item()), outputs["memory_embedding"].squeeze(0).cpu(), confidence=0.7)
    summary = {
        "sample_index": args.sample_index,
        "predicted_object_class": object_class,
        "predicted_regime_histogram": torch.bincount(regime_pred, minlength=data_config.regime_classes).tolist(),
        "predicted_box": object_box.tolist(),
        "predicted_support_plane": support_plane.tolist(),
        "physics": {key: float(value.squeeze().cpu()) for key, value in outputs["physics"].items()},
        "memory_visits": memory_entry.visits,
    }
    exports = export_scene_bundle(args.output_dir, fitted, environment_mesh, metadata=summary)
    print(json.dumps({key: str(value) for key, value in exports.items()}, indent=2))


if __name__ == "__main__":
    main()
