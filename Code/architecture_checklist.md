# Architecture Checklist

This file tracks implementation status against [arcitecture.md](/home/lala/Documents/GitHub/B2S/Code/arcitecture.md).

## Blocks

- `Done`: input handling for watch IMU, phone IMU, and barometer exists in the synthetic pipeline.
- `Done`: preprocess helpers exist for sync, calibration, denoise, gravity alignment, normalization, and windowing in [preprocess.py](/home/lala/Documents/GitHub/B2S/Code/b2s/preprocess.py).
- `Done`: activity-aware motion parser exists in [model.py](/home/lala/Documents/GitHub/B2S/Code/b2s/model.py).
- `Done`: locomotion, transition, and contact branches exist in [model.py](/home/lala/Documents/GitHub/B2S/Code/b2s/model.py).
- `Partial`: human motion state estimation exists, but full SMPL-X integration is not end-to-end trained in this prototype.
- `Done`: scene-constraint builder exists with voxel-grid targets in [scene_constraints.py](/home/lala/Documents/GitHub/B2S/Code/b2s/scene_constraints.py).
- `Done`: multi-hypothesis scene inference exists in [model.py](/home/lala/Documents/GitHub/B2S/Code/b2s/model.py).
- `Partial`: geometry/semantic/topology decoders exist, but are compact learned heads rather than full paper-grade decoders.
- `Partial`: physics and consistency refinement exists, but as a lightweight iterative/object-level refinement loop rather than a full simulator.
- `Partial`: persistent scene memory exists as a memory bank and loss path, but not full cross-user deployed fusion.
- `Partial`: object retrieval and fitting exists with local/manifest providers and iterative refinement, but not verified internet-scale retrieval.
- `Done`: output export exists for fitted meshes and JSON summaries.

## Losses

- `Done`: regime, locomotion, transition, contact, motion-state, scene, hypothesis, memory, retrieval, and physics groups are implemented in [losses.py](/home/lala/Documents/GitHub/B2S/Code/b2s/losses.py).
- `Partial`: many losses are faithful structural proxies, but not every formula from `arcitecture.md` is implemented exactly.

## Remaining blockers

- Full tested SMPL-X / 4D reconstruction pipeline in the active environment.
- Full real-data training pipeline with synchronized real sensors and aligned scene meshes.
- Verified remote internet-scale retrieval under network-enabled conditions.
- Full occupancy-grid / topology decoders trained on real scene supervision.
