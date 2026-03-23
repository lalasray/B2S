# B2S Prototype

This folder contains a runnable prototype built from `arcitecture.md`.

## Big Picture

The prototype follows this high-level flow:

1. `generate_dummy_data.py` creates short synthetic recordings that imitate:
   - watch IMU
   - phone IMU
   - phone barometer
   - motion labels
   - scene-level targets

2. `train_b2s.py` trains a multi-branch model that:
   - parses activity regime
   - predicts locomotion, transitions, and contacts
   - fuses those predictions into a motion-state estimate
   - predicts scene geometry, semantics, topology, retrieval, refinement, and memory outputs

3. `infer_and_fit_scene.py` runs the trained model on one sample, then:
   - predicts an object category and object box
   - optionally refines that box
   - retrieves a matching asset
   - fits the mesh into the predicted scene
   - exports OBJ files and a JSON summary

## ASCII Map

```text
Sensors
  watch IMU + phone IMU + barometer
        |
        v
Synthetic Dataset
  frame labels + scene targets + memory targets
        |
        v
Shared Temporal Encoder
        |
        +--> Regime Head
        +--> Locomotion Head
        +--> Transition Head
        +--> Contact Head
        |
        v
Motion Fusion
        |
        +--> Motion-State Heads
        +--> Constraint Heads
        +--> Scene Heads
        +--> Multi-Hypothesis Heads
        +--> Retrieval / Refinement / Memory / Physics Heads
        |
        v
Inference Outputs
  object class + refined box + support plane + scene summary
        |
        v
Retrieval Provider
  local assets or manifest-backed assets
        |
        v
PyTorch3D Mesh Fitting
  scale + orient + place + support snap + consistency scoring
        |
        v
Export
  fitted OBJ + environment OBJ + summary JSON
```

## What is included

- `b2s/dataset.py`: synthetic watch IMU, phone IMU, and barometer sequences plus training targets.
- `b2s/model.py`: a multi-branch PyTorch model that mirrors the architecture stages.
- `b2s/losses.py`: grouped training losses for motion parsing, motion state, scene inference, refinement, memory, and retrieval.
- `b2s/mesh_utils.py`: PyTorch3D-backed OBJ IO, mesh transforms, surface sampling, and consistency metrics.
- `b2s/assets.py`: local asset library, retrieval, fitting, support snapping, and scene export.
- `b2s/retrieval.py`: retrieval provider abstraction with local and manifest-backed asset sources.
- `b2s/preprocess.py`: real-sensor preprocessing helpers for sync, calibration, gravity alignment, normalization, and windowing.
- `b2s/scene_constraints.py`: scene-constraint builder for free-space, support, reachability, and topology cues.
- `b2s/memory_bank.py`: persistent scene-memory fusion utilities across repeated visits.
- `b2s/refinement.py`: iterative refinement loop for fitted objects and lightweight deformation.
- `b2s/deformation.py`: vertex-level deformation field for mesh refinement.
- `b2s/smpl_pipeline.py`: WHAM-based adapter for 4D human reconstruction when external dependencies are available.
- `generate_dummy_data.py`: writes a dummy dataset to `Code/data/`.
- `train_b2s.py`: trains the prototype and saves `Code/checkpoints/best.pt`.
- `infer_and_fit_scene.py`: runs inference and exports a fitted object mesh plus an environment mesh.

## Architecture Walkthrough

### 1. Synthetic data

`b2s/dataset.py` creates one sample as a short sequence of regime blocks such as:

- locomotion
- transition
- stationary use
- resting

For each frame it generates:

- fused sensor input
- step / heading / stride targets
- transition targets
- contact and reach targets
- pose, joints, root, and behavior targets

Then it collapses the whole sequence into scene-level targets such as:

- occupancy descriptor
- floor/support/wall descriptors
- object box and object class
- topology hints
- retrieval and memory targets

### 2. Model

`b2s/model.py` has one shared temporal encoder and several branches:

- regime head
- locomotion head
- transition head
- contact head
- motion-state heads
- scene heads
- multi-hypothesis scene heads
- retrieval / refinement / memory / physics heads

This is a compact implementation of the block diagram in `arcitecture.md`.

### 3. Losses

`b2s/losses.py` groups the objective into the same broad stages as the architecture:

- activity parsing
- locomotion
- transitions
- contacts
- motion-state estimation
- scene constraints
- geometry / semantics / topology
- multi-hypothesis scene learning
- refinement
- memory
- retrieval
- physics-style consistency

These are simplified prototype losses, not a full paper-faithful implementation of every formula.

### 4. Retrieval and mesh fitting

`b2s/retrieval.py` chooses where assets come from:

- local library
- manifest-backed source

`b2s/assets.py` and `b2s/mesh_utils.py` then:

- load meshes
- scale/orient/place them from predicted object boxes
- snap them to support planes
- score fit quality and collision heuristics
- export the fitted object and environment mesh

## Run

```bash
python Code/generate_dummy_data.py
python Code/train_b2s.py --epochs 8 --batch-size 32
python Code/infer_and_fit_scene.py --sample-index 0
python Code/infer_and_fit_scene.py --sample-index 0 --asset-source manifest --asset-manifest path/to/assets.json
python Code/infer_and_fit_scene.py --sample-index 0 --iterative-refine
```

## Typical workflow

If you are new to the repo, the easiest order is:

1. Read `arcitecture.md` for the intended system design.
2. Read `architecture_checklist.md` to see what is done, partial, or still blocked.
3. Read `Code/README.md` for the practical implementation map.
4. Open `b2s/dataset.py` to understand what the model is trained on.
5. Open `b2s/model.py` to see how the branches are organized.
6. Open `b2s/losses.py` to see what supervision each branch receives.
7. Run training and then inference to inspect exported results.

## Notes

- The dataset is synthetic and designed to exercise the pipeline structure rather than represent real biomechanics.
- Retrieval supports both a local library and a manifest-based provider. A manifest can point to local files or remote OBJ URLs; remote fetching was added but not network-tested in this sandbox.
- Mesh fitting and scene consistency use PyTorch3D primitives, plus learned box refinement, cross-visit memory supervision, and stronger physics-style consistency losses.
- The code is meant as a strong starting point for replacing synthetic targets with real labels, better encoders, stronger scene decoders, and real asset repositories later.
