from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy helper that works on both sequence and batch predictions."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy helper for contact and event-style predictions."""
    return F.binary_cross_entropy_with_logits(logits, targets)


def _l1(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Simple regression helper used across the prototype."""
    return F.l1_loss(pred, targets)


def _batch_memory_alignment(embedding: torch.Tensor, scene_id: torch.Tensor) -> torch.Tensor:
    """Encourage samples from the same synthetic scene template to cluster together."""
    if embedding.shape[0] < 2:
        return embedding.new_tensor(0.0)
    # Compare every sample against every other sample in the batch.
    # Same-scene pairs should be similar, different-scene pairs should stay apart.
    normalized = F.normalize(embedding, dim=-1)
    similarity = normalized @ normalized.T
    same_scene = scene_id.unsqueeze(0) == scene_id.unsqueeze(1)
    eye = torch.eye(scene_id.shape[0], device=scene_id.device, dtype=torch.bool)
    positive_mask = same_scene & ~eye
    negative_mask = ~same_scene
    pos_loss = (1.0 - similarity[positive_mask]).mean() if positive_mask.any() else embedding.new_tensor(0.0)
    neg_loss = F.relu(similarity[negative_mask] - 0.2).mean() if negative_mask.any() else embedding.new_tensor(0.0)
    return pos_loss + neg_loss


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute grouped losses that roughly correspond to the architecture stages."""
    losses = {}

    # A. Activity-aware parsing.
    losses["regime"] = _ce(outputs["regime_logits"], batch["regime"])

    # B. Locomotion branch.
    locomotion = outputs["locomotion"]
    losses["locomotion"] = (
        _bce(locomotion["step"], batch["step"])
        + _l1(locomotion["heading"], batch["heading"])
        + _l1(locomotion["stride"], batch["stride"])
        + _ce(locomotion["floor_event"], batch["floor_event"])
        + _l1(locomotion["baro"], batch["baro"])
    )

    # C. Transition branch.
    transition = outputs["transition"]
    losses["transition"] = (
        _ce(transition["cls"], batch["transition_cls"])
        + _l1(transition["root"], batch["transition_root"])
        + _l1(transition["torso"], batch["transition_torso"])
    )

    # D. Stationary/contact branch.
    contact = outputs["contact"]
    # Encourage stable contacts during frames where the branch predicts support.
    foot_slide = (torch.sigmoid(contact["contacts"][:, :, :1]) * batch["stride"].norm(dim=-1, keepdim=True)).mean()
    contact_vel = (torch.sigmoid(contact["contacts"]) * batch["transition_root"].abs().mean(dim=-1, keepdim=True)).mean()
    losses["contact"] = (
        _bce(contact["contacts"], batch["contacts"])
        + _l1(contact["reach"], batch["reach"])
        + _ce(contact["affordance"], batch["affordance"])
        + foot_slide
        + contact_vel
    )

    # E. Human motion state estimation.
    motion = outputs["motion"]
    root_pred = motion["root"]
    root_true = batch["root"]
    # Extra motion priors: match velocity and keep trajectories smooth.
    vel_pred = root_pred[:, 1:] - root_pred[:, :-1]
    vel_true = root_true[:, 1:] - root_true[:, :-1]
    smooth = root_pred[:, 2:] - 2 * root_pred[:, 1:-1] + root_pred[:, :-2]
    losses["motion"] = (
        _l1(motion["pose"], batch["pose"])
        + _l1(motion["joints"], batch["joints"])
        + _l1(root_pred, root_true)
        + _l1(vel_pred, vel_true)
        + smooth.abs().mean()
        + _ce(motion["behavior"], batch["behavior"])
        + F.gaussian_nll_loss(motion["root"], batch["root"], torch.ones_like(batch["root"]) * 0.5)
        + _l1(motion["uncertainty"], batch["uncertainty"])
    )

    # F/G. Scene constraint proxy losses.
    constraints = outputs["constraints"]
    losses["constraints"] = _l1(constraints["free_space"], batch["free_space"]) + _l1(
        constraints["support_height"], batch["support_height"]
    ) + _l1(constraints["free_space"].mean(dim=-1, keepdim=True), batch["free_space"].mean(dim=-1, keepdim=True))

    # H/I/J. Scene geometry, semantics, and topology.
    scene = outputs["scene"]
    losses["scene"] = (
        _l1(scene["occupancy"], batch["occupancy"])
        + _l1(scene["floor_plane"], batch["floor_plane"])
        + _l1(scene["wall"], batch["wall"])
        + _l1(scene["support_plane"], batch["support_plane"])
        + _l1(scene["object_box"], batch["object_box"])
        + _ce(scene["object_class"], batch["object_class"])
        + _ce(scene["semantic"], batch["semantic"])
        + _l1(scene["interaction_zone"], batch["interaction_zone"])
        + _l1(scene["room_graph"], batch["room_graph"])
        + _ce(scene["door"], batch["door"])
        + _ce(scene["stairs"], batch["stairs"])
        + _ce(scene["topology_transition"], batch["topology_transition"])
        + _l1(scene["floor_plane"][:, -1:], batch["root"][:, :, 2].amin(dim=1, keepdim=True))
        + _l1(scene["support_plane"][:, 2:3], batch["support_height"])
    )

    # K. Multi-hypothesis training via best-of-K and diversity.
    hypotheses = outputs["hypotheses"]
    target_scene = batch["best_scene"].unsqueeze(1)
    # best_k finds the closest scene hypothesis to the ground-truth scene signature.
    best_k = torch.abs(hypotheses - target_scene).mean(dim=-1)
    best_k_values, best_idx = best_k.min(dim=1)
    diversity = torch.pdist(hypotheses.reshape(hypotheses.shape[0], -1), p=2).mean() if hypotheses.shape[0] > 1 else 0.0
    losses["multi_hypothesis"] = best_k_values.mean() - 0.01 * diversity + _ce(outputs["hypothesis_scores"], best_idx)

    # L/M/N. Refinement, memory, retrieval, and physics-style consistency.
    refined_scene = outputs["refined_scene"]
    refined_box = outputs["retrieval"] + outputs["mesh_refine"]
    # The refiner is trained to improve both the compact scene signature and the object box.
    losses["refine"] = (
        _l1(refined_scene, batch["best_scene"])
        + _l1(refined_scene, batch["memory_target"])
        + _l1(refined_box, batch["retrieval_target"])
    )
    losses["memory"] = (
        _l1(outputs["memory"], batch["memory_target"])
        + _l1(outputs["memory_embedding"], batch["paired_memory_target"])
        + _batch_memory_alignment(outputs["memory_embedding"], batch["scene_id"])
    )
    losses["retrieval"] = _l1(outputs["retrieval"], batch["retrieval_target"]) + _l1(
        outputs["mesh_refine"], batch["retrieval_embedding"]
    )

    support_energy = outputs["physics"]["support_energy"]
    collision_energy = outputs["physics"]["collision_energy"]
    stability_energy = outputs["physics"]["stability_energy"]
    # These targets are still synthetic proxies, but they let the model learn that:
    # - support should align with plausible support heights
    # - collision energy should rise when objects intrude above supports
    # - stability should correlate with free-space quality
    support_target = batch["support_height"]
    collision_target = torch.clamp(batch["object_box"][:, 2:3] - batch["support_height"], min=0.0)
    stability_target = batch["free_space"].mean(dim=-1, keepdim=True)
    positive_extent_penalty = F.relu(0.2 - refined_box[:, :3].abs()).mean()
    losses["physics"] = (
        _l1(support_energy, support_target)
        + _l1(collision_energy, collision_target)
        + _l1(stability_energy, stability_target)
        + positive_extent_penalty
        + F.relu(refined_scene.abs().mean() - 2.0)
    )

    # Final weighted sum used by the trainer.
    total = sum(weights[name] * loss for name, loss in losses.items())
    # metrics mirrors the grouped loss structure so training logs stay interpretable.
    metrics = {name: float(loss.detach().cpu()) for name, loss in losses.items()}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics
