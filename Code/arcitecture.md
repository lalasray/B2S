+==================================================================================================+
| INPUT                                                                                            |
|--------------------------------------------------------------------------------------------------|
| watch IMU | phone IMU | phone barometer                                                          |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| PREPROCESS                                                                                       |
|--------------------------------------------------------------------------------------------------|
| time sync | calibration | denoise | gravity alignment | device-frame normalization | windowing   |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| ACTIVITY-AWARE MOTION PARSER                                                                     |
|--------------------------------------------------------------------------------------------------|
| locomotion | transition | stationary use | resting                                               |
+==================================================================================================+
                  |                               |                                |
                  v                               v                                v
      +--------------------------+   +---------------------------+   +-----------------------------+
      | LOCOMOTION BRANCH        |   | TRANSITION BRANCH         |   | STATIONARY / CONTACT BRANCH|
      |--------------------------|   |---------------------------|   |-----------------------------|
      | PDR                      |   | sit/stand/lie dynamics    |   | support/contact inference   |
      | step timing              |   | pelvis/root change        |   | hand/foot/pelvis/back       |
      | heading                  |   | torso orientation change  |   | local anchors               |
      | stride/displacement      |   | short root displacement   |   | reach/support zones         |
      | stair cues from baro     |   | baro-assisted elevation   |   | use-conditioned affordances |
      +--------------------------+   +---------------------------+   +-----------------------------+
                  \                               |                                /
                   \                              |                               /
                    \                             |                              /
                     v                            v                             v
+==================================================================================================+
| HUMAN MOTION STATE ESTIMATION                                                                    |
|--------------------------------------------------------------------------------------------------|
| 4D pose / SMPL-X | root trajectory | contact timeline | behavior labels | uncertainty            |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| SCENE CONSTRAINT BUILDER                                                                         |
|--------------------------------------------------------------------------------------------------|
| swept-body free space | no-occupancy carving | support constraints | reachability volumes        |
| vertical topology from barometer | room-transition / corridor cues                                |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| MULTI-HYPOTHESIS SCENE INFERENCE                                                                 |
|--------------------------------------------------------------------------------------------------|
| z1, z2, ..., zK                                                                                  |
| each latent = one plausible explanation of the surrounding environment                           |
+==================================================================================================+
             |                              |                               |
             v                              v                               v
   +----------------------+      +----------------------+      +----------------------+
   | GEOMETRY DECODER     |      | SEMANTIC DECODER     |      | TOPOLOGY DECODER     |
   |----------------------|      |----------------------|      |----------------------|
   | occupancy grid       |      | object classes       |      | room graph           |
   | floor/support planes |      | affordances          |      | doors/corridors      |
   | walls                |      | interaction zones    |      | floors/stairs        |
   | oriented 3D boxes    |      | confidence           |      | room/floor transitions|
   +----------------------+      +----------------------+      +----------------------+
             \                              |                               /
              \                             |                              /
               \                            |                             /
                v                           v                            v
+==================================================================================================+
| PHYSICS AND CONSISTENCY REFINEMENT                                                               |
|--------------------------------------------------------------------------------------------------|
| collision check | contact support | barometer consistency | behavior plausibility | sparsity     |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| PERSISTENT SCENE MEMORY                                                                          |
|--------------------------------------------------------------------------------------------------|
| fuse repeated visits | align trajectories | refine topology | update object confidence            |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| OBJECT RETRIEVAL AND FITTING                                                                     |
|--------------------------------------------------------------------------------------------------|
| retrieve CAD / STL / OBJ / glTF assets | scale | orient | place | support check | collision fix |
| optional template deformation / mesh refinement                                                  |
+==================================================================================================+
                                                |
                                                v
+==================================================================================================+
| OUTPUT                                                                                           |
|--------------------------------------------------------------------------------------------------|
| functional 3D structure                                                                          |
| fused persistent scene graph                                                                     |
| occupancy / free-space map                                                                       |
| floor / wall / support planes                                                                    |
| object boxes and optional retrieved/fitted meshes                                                |
| 4D interaction state timeline                                                                    |
+==================================================================================================+



## End-to-end blocks and losses

Below is a **paper-ready breakdown** of what each block does, followed by the **losses** grouped by stage.

---

# 1. What each block should do

## 1) Input

**Inputs**

* watch IMU
* phone IMU
* phone barometer

**Goal**

* provide sparse motion and vertical-change cues from everyday devices

**Output**

* raw synchronized sensor streams

---

## 2) Preprocess

**Does**

* timestamp sync across devices
* denoising / smoothing
* gravity alignment
* coordinate normalization
* calibration for device placement/orientation
* segmentation into temporal windows

**Goal**

* convert noisy wearable signals into stable model inputs

**Output**

* normalized sensor windows
* gravity-aligned inertial streams
* cleaned barometric trend

---

## 3) Activity-Aware Motion Parser

**Does**

* classify each time window into:

  * locomotion
  * transition
  * stationary use
  * resting

**Goal**

* choose the right motion model for each regime

**Why**

* PDR is useful for walking/stairs
* contact/support reasoning is more useful for sitting/lying/reaching

**Output**

* regime label per frame/window
* regime confidence

---

## 4) Locomotion Branch

**Does**

* pedestrian dead reckoning during locomotion
* step timing / gait phase
* heading estimation
* stride/displacement estimation
* stair / vertical-motion cue extraction from barometer

**Goal**

* estimate translational movement through space

**Output**

* locomotion root displacement prior
* heading trajectory
* step events
* stair/floor-change likelihood

---

## 5) Transition Branch

**Does**

* model sit-down, stand-up, lie-down, get-up, bend-rise
* estimate short root change and torso/pelvis state changes
* use barometer as weak vertical cue

**Goal**

* explain short but meaningful body-support transitions

**Output**

* transition state
* pelvis/root delta
* support-type cues

---

## 6) Stationary / Contact Branch

**Does**

* infer foot/hand/pelvis/back support events
* detect stationary interaction zones
* infer reach/support/use-conditioned affordances

**Goal**

* estimate how the person is using surrounding structure

**Output**

* contact timeline
* local anchor/support candidates
* reach zones
* affordance cues

---

## 7) Human Motion State Estimation

**Does**

* fuse outputs from locomotion, transition, and contact branches
* infer full 4D body state:

  * SMPL/SMPL-X pose
  * root trajectory
  * contact states
  * behavior labels
  * uncertainty

**Goal**

* produce a consistent body-in-space estimate

**Output**

* 4D pose
* root path
* contact labels
* behavior labels
* uncertainty map

---

## 8) Scene Constraint Builder

**Does**

* construct scene evidence from body motion:

  * swept-body free space
  * no-occupancy carving
  * support surface constraints
  * reachability volumes
  * vertical topology from barometer
  * room/corridor transition evidence

**Goal**

* translate human behavior into scene constraints

**Output**

* occupancy constraints
* support constraints
* topology cues
* interaction/use constraints

---

## 9) Multi-Hypothesis Scene Inference

**Does**

* infer several possible scene explanations
* represent ambiguity explicitly

**Goal**

* avoid overcommitting to one scene from sparse signals

**Output**

* latent scene hypotheses (z_1 \dots z_K)

---

## 10) Geometry Decoder

**Does**

* decode geometric structure:

  * occupancy grid
  * floor plane
  * support planes
  * walls
  * oriented object boxes

**Goal**

* produce functional geometry

**Output**

* geometric scene representation

---

## 11) Semantic Decoder

**Does**

* assign object classes and affordances
* estimate interaction zones and confidence

**Goal**

* convert geometry into functional scene understanding

**Output**

* object labels
* affordances
* confidence scores

---

## 12) Topology Decoder

**Does**

* infer room graph, doors, corridors, stairs, floor transitions

**Goal**

* recover connectivity and vertical organization

**Output**

* room/floor topology graph

---

## 13) Physics and Consistency Refinement

**Does**

* iteratively update scene using:

  * collision checks
  * contact support consistency
  * behavior plausibility
  * barometric consistency
  * sparsity priors

**Goal**

* make the inferred scene more plausible

**Output**

* refined scene hypotheses

---

## 14) Persistent Scene Memory

**Does**

* fuse repeated visits
* align trajectories across visits/users
* update object confidence
* refine topology over time

**Goal**

* accumulate evidence and improve stability

**Output**

* fused persistent scene representation

---

## 15) Object Retrieval and Fitting

**Does**

* retrieve CAD/mesh assets for predicted objects
* scale, orient, and place them
* optionally deform/refine templates
* reject collisions / unsupported placements

**Goal**

* instantiate the functional scene as a concrete 3D scene

**Output**

* retrieved/fitted object meshes

---

## 16) Output

**Two valid outputs**

### A. Functional 3D structure

* free space
* occupancy
* floor/wall/support planes
* object boxes
* topology graph
* interaction timeline

### B. Instantiated 3D scene

* all of the above
* plus fitted object meshes / templates

---

# 2. Losses by block

---

## A. Activity / regime losses

### 1. Regime classification loss

Classify locomotion / transition / stationary use / resting.

[
L_{\text{regime}} = \text{CE}(r_t,\hat r_t)
]

---

## B. Locomotion branch losses

### 2. Step event loss

[
L_{\text{step}} = \text{BCE}(s_t,\hat s_t)
]

### 3. Heading loss

[
L_{\text{heading}} = \sum_t d(\theta_t,\hat \theta_t)
]

### 4. Stride / displacement loss

[
L_{\text{stride}} = \sum_t |\Delta p_t^{xy} - \widehat{\Delta p}_t^{xy}|_1
]

### 5. Stair / floor-change loss

[
L_{\text{floor-event}} = \text{CE}(f_t,\hat f_t)
]

### 6. Barometric height-change loss

[
L_{\text{baro}} = \sum_t |\Delta h_t - \widehat{\Delta h}_t|_1
]

---

## C. Transition branch losses

### 7. Transition classification loss

[
L_{\text{trans-cls}} = \text{CE}(u_t,\hat u_t)
]

### 8. Pelvis/root transition loss

[
L_{\text{trans-root}} = \sum_t | \Delta p_t^{root} - \widehat{\Delta p}_t^{root} |_1
]

### 9. Torso orientation transition loss

[
L_{\text{trans-ori}} = \sum_t d(R_t^{torso}, \hat R_t^{torso})
]

---

## D. Stationary / contact branch losses

### 10. Foot contact loss

[
L_{\text{foot-contact}} = \text{BCE}(c_t^{foot},\hat c_t^{foot})
]

### 11. Hand contact loss

[
L_{\text{hand-contact}} = \text{BCE}(c_t^{hand},\hat c_t^{hand})
]

### 12. Pelvis contact loss

[
L_{\text{pelvis-contact}} = \text{BCE}(c_t^{pelvis},\hat c_t^{pelvis})
]

### 13. Back contact loss

[
L_{\text{back-contact}} = \text{BCE}(c_t^{back},\hat c_t^{back})
]

### 14. Reach/use zone loss

[
L_{\text{reach}} = d(\mathcal{Z}^{reach}, \widehat{\mathcal{Z}}^{reach})
]

### 15. Affordance prediction loss

[
L_{\text{aff}} = \text{CE}(a_t^{aff},\hat a_t^{aff})
]

---

## E. Human motion state estimation losses

### 16. SMPL pose loss

[
L_{\text{pose}} = \sum_{t,j} d(R_{t,j},\hat R_{t,j})
]

### 17. Joint/vertex reconstruction loss

[
L_{\text{joint}} = \sum_t |J_t - \hat J_t|_1
]

### 18. Root position loss

[
L_{\text{root}} = \sum_t |p_t - \hat p_t|_1
]

### 19. Root velocity loss

[
L_{\text{vel}} = \sum_t |(p_t-p_{t-1})-(\hat p_t-\hat p_{t-1})|_1
]

### 20. Temporal smoothness loss

[
L_{\text{smooth}} = \sum_t |p_{t+1}-2p_t+p_{t-1}|_1
]

### 21. Behavior classification loss

[
L_{\text{beh}} = \text{CE}(b_t,\hat b_t)
]

### 22. Uncertainty calibration loss

[
L_{\text{unc}} = \text{NLL or calibration loss}
]

---

## F. Motion-consistency losses

### 23. Foot sliding loss

For stance feet, velocity should be near zero.

[
L_{\text{foot-slide}} = \sum_{t,f}\hat c_{t,f}^{foot}|x_{t,f}-x_{t-1,f}|_1
]

### 24. Contact velocity loss

Active contact should remain attached to the support.

[
L_{\text{contact-vel}} = \sum_{t,k}\hat c_{t,k}|v_{t,k}^{body}-v_{t,k}^{support}|_1
]

### 25. Gravity/upright consistency loss

[
L_{\text{grav}} = \sum_t |g_t-\hat g_t|
]

---

## G. Scene constraint builder losses

### 26. Free-space carving loss

Voxels traversed by the body should prefer free space.

[
L_{\text{free}} = \sum_{v\in \text{swept body}} \text{BCE}(0,\hat O_v)
]

### 27. Reachability consistency loss

Scene surfaces should appear in reachable regions when required.

[
L_{\text{reach-cons}} = \sum_t d(\mathcal{S}_{interact},\mathcal{Z}_t^{reach})
]

### 28. Support-height consistency loss

Seat/bed/table heights should match behavior/contact evidence.

[
L_{\text{support-h}} = \sum_t |\hat h_t^{support} - h_t^{behavior}|_1
]

---

## H. Scene geometry decoder losses

### 29. Occupancy loss

[
L_{\text{occ}} = \text{BCE}(O,\hat O)
]

### 30. Floor-plane loss

[
L_{\text{floorplane}} = \sum_{t,f}\hat c_{t,f}^{foot}, d(x_{t,f},\Pi_{floor})
]

### 31. Wall loss

[
L_{\text{wall}} = d(\mathcal{W}, \hat{\mathcal{W}})
]

### 32. Support-plane loss

[
L_{\text{support}} = d(\mathcal{P}*{support}, \hat{\mathcal{P}}*{support})
]

### 33. Object box regression loss

For center, size, yaw:
[
L_{\text{box}} = L_{\text{center}} + L_{\text{size}} + L_{\text{yaw}}
]

### 34. Object class loss

[
L_{\text{obj-cls}} = \text{CE}(y^{obj},\hat y^{obj})
]

---

## I. Semantic decoder losses

### 35. Semantic label loss

[
L_{\text{sem}} = \text{CE}(y^{sem},\hat y^{sem})
]

### 36. Affordance loss

[
L_{\text{aff-sem}} = \text{CE}(y^{aff},\hat y^{aff})
]

### 37. Interaction-zone loss

[
L_{\text{int-zone}} = d(\mathcal{Z}^{int}, \hat{\mathcal{Z}}^{int})
]

---

## J. Topology decoder losses

### 38. Room graph loss

[
L_{\text{roomgraph}} = \text{graph matching loss}
]

### 39. Door / corridor loss

[
L_{\text{door}} = \text{CE}(y^{door},\hat y^{door})
]

### 40. Floor/stair topology loss

[
L_{\text{stairs}} = \text{CE}(y^{stairs},\hat y^{stairs}) + |\Delta h_t - \widehat{\Delta h}_t^{stairs}|_1
]

### 41. Transition-topology consistency loss

[
L_{\text{topo-trans}} = d(\text{predicted transitions},\text{scene graph transitions})
]

---

## K. Multi-hypothesis losses

### 42. Best-of-K reconstruction loss

[
L_{\text{bestK}} = \min_k \ell(S^{gt}, \hat S_k)
]

### 43. Diversity loss

[
L_{\text{div}} = - \sum_{i<j} d(\hat S_i,\hat S_j)
]

### 44. Confidence/ranking loss

[
L_{\text{rank}} = \text{CE}(k^*, \hat \pi)
]

---

## L. Physics and consistency refinement losses

### 45. Scene penetration loss

Body should not intersect occupied scene.

[
L_{\text{pen}} = \sum_t \sum_{x\in \mathcal{B}_t}\phi(\hat S,x)
]

### 46. Contact support consistency loss

[
L_{\text{contact-sup}} = \sum_{t,k}\hat c_{t,k} d(x_{t,k}^{body},\mathcal{S}_{support})
]

### 47. Behavior plausibility loss

Sitting should imply seat-like support, lying should imply extended support, etc.

[
L_{\text{beh-plaus}} = \text{compatibility loss between behavior and scene}
]

### 48. Sparsity / scene prior loss

[
L_{\text{sparse}} = |\hat O|_1 \quad \text{or learned prior}
]

### 49. Human-scene consistency loss

Umbrella loss over support, reach, penetration, topology.

[
L_{\text{HSC}} = L_{\text{contact-sup}} + L_{\text{reach-cons}} + L_{\text{pen}} + L_{\text{beh-plaus}}
]

---

## M. Persistent scene memory losses

### 50. Cross-visit alignment loss

[
L_{\text{align}} = |T_{a\to b}(\hat S_a)-\hat S_b|_1
]

### 51. Fusion consistency loss

[
L_{\text{fuse}} = |\hat S^{fused} - \text{Agg}(\hat S^{(1)},...,\hat S^{(n)})|_1
]

### 52. Retrieval / memory matching loss

[
L_{\text{mem}} = \text{contrastive retrieval loss}
]

### 53. Temporal memory stability loss

[
L_{\text{stable}} = d(\hat S_t^{mem}, \hat S_{t+1}^{mem})
]

---

## N. Object retrieval and fitting losses

### 54. Asset retrieval loss

Match predicted object embedding to correct asset.

[
L_{\text{retrieval}} = \text{contrastive or CE loss}
]

### 55. Box-to-mesh fitting loss

Retrieved mesh should fit predicted box.

[
L_{\text{fit-box}} = d(B^{pred}, B^{mesh})
]

### 56. Support attachment loss

Object should rest on predicted support.

[
L_{\text{fit-sup}} = d(\mathcal{M}^{obj}*{support}, \mathcal{P}*{support})
]

### 57. Mesh collision loss

[
L_{\text{mesh-coll}} = \text{collision penalty with scene/other objects}
]

### 58. Interaction compatibility loss

Chair should support sitting, bed should support lying, etc.

[
L_{\text{fit-int}} = \text{affordance compatibility loss}
]

### 59. Optional mesh deformation loss

[
L_{\text{deform}} = \text{regularized deformation loss}
]

---

# 3. Total loss

A clean grouped formulation:

[
L_{\text{total}} =
\lambda_r L_{\text{regime}}
+\lambda_l L_{\text{locomotion}}
+\lambda_t L_{\text{transition}}
+\lambda_c L_{\text{contact}}
+\lambda_m L_{\text{motion}}
+\lambda_s L_{\text{scene}}
+\lambda_h L_{\text{hyp}}
+\lambda_p L_{\text{physics}}
+\lambda_{mem} L_{\text{memory}}
+\lambda_o L_{\text{object}}
]

where:

[
L_{\text{locomotion}} = L_{\text{step}}+L_{\text{heading}}+L_{\text{stride}}+L_{\text{floor-event}}+L_{\text{baro}}
]

[
L_{\text{transition}} = L_{\text{trans-cls}}+L_{\text{trans-root}}+L_{\text{trans-ori}}
]

[
L_{\text{contact}} = L_{\text{foot-contact}}+L_{\text{hand-contact}}+L_{\text{pelvis-contact}}+L_{\text{back-contact}}+L_{\text{foot-slide}}+L_{\text{contact-vel}}
]

[
L_{\text{motion}} = L_{\text{pose}}+L_{\text{joint}}+L_{\text{root}}+L_{\text{vel}}+L_{\text{smooth}}+L_{\text{beh}}+L_{\text{unc}}
]

[
L_{\text{scene}} = L_{\text{free}}+L_{\text{support-h}}+L_{\text{occ}}+L_{\text{floorplane}}+L_{\text{wall}}+L_{\text{support}}+L_{\text{box}}+L_{\text{obj-cls}}+L_{\text{sem}}+L_{\text{aff-sem}}+L_{\text{roomgraph}}+L_{\text{door}}+L_{\text{stairs}}
]

[
L_{\text{hyp}} = L_{\text{bestK}}+L_{\text{div}}+L_{\text{rank}}
]

[
L_{\text{physics}} = L_{\text{pen}}+L_{\text{contact-sup}}+L_{\text{beh-plaus}}+L_{\text{sparse}}+L_{\text{HSC}}
]

[
L_{\text{memory}} = L_{\text{align}}+L_{\text{fuse}}+L_{\text{mem}}+L_{\text{stable}}
]

[
L_{\text{object}} = L_{\text{retrieval}}+L_{\text{fit-box}}+L_{\text{fit-sup}}+L_{\text{mesh-coll}}+L_{\text{fit-int}}+L_{\text{deform}}
]

---

# 4. Minimal practical subset

If you want the **smallest set that still works**, keep:

* `L_regime`
* `L_step`
* `L_heading`
* `L_baro`
* `L_foot-contact`
* `L_foot-slide`
* `L_pose`
* `L_root`
* `L_beh`
* `L_free`
* `L_occ`
* `L_floorplane`
* `L_box`
* `L_obj-cls`
* `L_pen`
* `L_contact-sup`
* `L_bestK`
* `L_align`
* `L_retrieval`
* `L_fit-box`

That is a credible first implementation.

## Strongest one-line description

**The system first estimates activity-aware human motion from sparse wearables, converts pose, contact, and barometric signals into scene constraints, infers multiple plausible functional scenes, refines them using human-scene consistency and physical feasibility, and optionally instantiates the result through object retrieval and fitting.**

If helpful, I can turn this into a compact table with columns: **Block | Inputs | Outputs | Losses | Purpose**.
