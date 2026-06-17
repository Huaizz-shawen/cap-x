# RoboCasa GR1 Skill Library Plan

This document defines the intended interface for a reusable RoboCasa GR1
Fourier-hand skill library. The goal is to solve multiple tabletop tasks through
the same observation and execution abstractions, instead of writing a separate
privileged helper for each task.

Current target tasks:

- `PnPPouring`: grasp a source cup and pour the ball into a target bowl.
- `TwoArmLift`: coordinate both arms and hands to lift/support an object.
- Drawer task: make contact with a handle and pull along the drawer opening
  axis.

The library should start with privileged simulator inputs for fast iteration,
but its interface should be designed so those inputs can later be replaced by
RGBD or detector outputs.

## Design Goal

Use one object-centric interface across the three tasks:

```python
scene = observe_scene()
calibration = probe_eef_directions(scene)
grasp = select_grasp_pose(scene, object_name, hand_pose_family)
result = execute_eef_skill(grasp.eef_waypoints, grasp.hand_pose_schedule)
status = verify_success_or_contact(scene, result)
```

Task-specific code should mainly choose objects, targets, contact mode, and
high-level waypoints. It should not define a completely new low-level controller
for every task.

## Skill Layers

### 1. Observation Layer

Initial privileged baseline:

```python
observe_scene() -> SceneState
```

`SceneState` should include:

- RGB observations.
- Depth observations when available.
- Camera metadata and image resolution.
- Robot EEF poses.
- Hand joint state or current hand-pose family.
- Object centers and approximate poses.
- Candidate affordances such as cup rim, bowl center, handle center, support
  points, or lift points.

Early implementation can read exact simulator pose. The stable interface should
not require the caller to know whether the source was simulator truth, noisy
pose, RGBD, or a detector.

### 2. Initialization / Calibration Skill

```python
probe_eef_directions(
    scene: SceneState,
    arm: str = "right",
    step: float = 0.04,
    axes: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z"),
) -> WorkspaceCalibration
```

Purpose:

- Establish how world-frame directions appear in the current camera view.
- Estimate safe local workspace boundaries.
- Verify that IK can reach representative nearby poses.
- Give the model a compact prior for interpreting feedback such as "left",
  "right", "forward", "back", "up", and "down".

Minimal behavior:

1. Record the current EEF pose and image.
2. Move the selected EEF by small deltas along safe axes.
3. Record before/after RGBD, EEF pose, and object pose summaries.
4. Return a direction map and IK reachability flags.
5. Restore or return near the original pose when possible.

Constraints:

- Probe motions must be small and non-contact.
- The skill should run once per episode or reset state, then cache results.
- It should not be run before every action, because that would reduce collection
  efficiency.

Output sketch:

```python
@dataclass
class WorkspaceCalibration:
    arm: str
    origin_eef_pose: Pose
    direction_image_map: dict[str, str]
    reachable_axes: dict[str, bool]
    safe_workspace_hint: dict[str, tuple[float, float]]
    observations: list[ProbeObservation]
```

### 3. Grasp / Contact Selection Layer

```python
select_grasp_pose(
    scene: SceneState,
    object_name: str,
    hand_pose_family: str,
    contact_mode: str = "grasp",
) -> GraspSpec
```

`hand_pose_family` should begin with a small engineered set:

- `open`: no-contact staging.
- `pinch`: small object or handle pinch.
- `cylindrical`: cup, bottle, and round container grasp.
- `support`: palm/support grasp or scoop-like stabilizing contact.
- `hook`: handle pulling or drawer opening.
- `press`: pushing, pressing, or maintaining broad contact.

The long-term goal can be 20+ hand shapes, but the first version should keep the
set small and attach clear preconditions to each family.

`GraspSpec` should include:

- pregrasp EEF pose,
- contact EEF pose,
- retreat/lift pose,
- hand-pose schedule,
- approach axis,
- expected contact surface,
- task-specific assumptions.

### 4. EEF Execution Layer

```python
execute_eef_skill(
    eef_waypoints: list[Pose],
    hand_pose_schedule: list[HandPoseCommand],
    arm: str,
    constraints: SkillConstraints | None = None,
) -> SkillResult
```

This layer should own the IK execution details:

- interpolate EEF waypoints,
- call the existing IK controller,
- apply discrete hand poses at scheduled phases,
- maintain contact when required,
- record transition data and diagnostic frames.

The task-level agent should not hand-write raw joint actions unless debugging a
controller failure.

### 5. Verification Layer

```python
verify_success_or_contact(scene: SceneState, result: SkillResult) -> SkillStatus
```

Verification should include:

- simulator task reward / completion flag when available,
- object-target geometric metrics,
- contact maintained / lost,
- collision or self-collision flags when available,
- first failure phase.

This output is what the code agent should use for iterative feedback.

## Three-Task Acceptance Matrix

| Task | Required shared skills | Task-specific choices | Pass criterion |
| --- | --- | --- | --- |
| PnPPouring | observe, probe, cylindrical grasp, EEF lift/move, final hand/arm posture, verify | source cup, target bowl, pour height, wrist/shoulder posture | ball enters target bowl |
| TwoArmLift | observe, probe both arms, support/cylindrical grasp, synchronized bimanual EEF lift, verify | left/right support points, lift height, hand pose family | object lifted/stabilized |
| Drawer | observe, probe, hook/pinch/press contact, constrained EEF pull, contact verification | handle affordance, opening axis, pull distance | drawer opens or handle moves along axis |

The interface is considered useful only if these tasks share the same function
shapes and differ primarily in parameters and affordances.

## Privilege-Reduction Roadmap

### Stage 0: Privileged Scripted Baselines

Use simulator object poses and exact affordances to make each task work. This is
for fast debugging and transition collection, not for claiming generalization.

### Stage 1: Pose-Noise Tolerance

Inject noise into object centers, target centers, and handle/cup affordances.
Measure success as noise increases.

Suggested levels:

- `0.5cm`
- `1cm`
- `2cm`
- `4cm`

### Stage 2: Weak-Privileged Geometry

Keep object identity from the simulator, but quantize or approximate geometry as
if it came from RGBD or a detector.

### Stage 3: RGBD / Detector Inputs

Replace exact object pose with estimates from RGBD, segmentation masks, or
object detectors. Keep the same skill interface.

### Stage 4: Cross-Task Transfer

Run the same interfaces on tasks outside the initial three. Promote a skill to a
reusable library component only after it survives this stage.

## Implementation Order

1. Define lightweight data structures: `SceneState`, `Pose`, `GraspSpec`,
   `HandPoseCommand`, `SkillResult`, and `WorkspaceCalibration`.
2. Implement `probe_eef_directions()` with safe small EEF deltas.
3. Wrap existing GR1 cup-pouring default candidate behind the common interface.
4. Implement TwoArmLift through the same bimanual EEF execution layer.
5. Implement Drawer through `hook` / `pinch` / `press` contact and constrained
   pull motion.
6. Add pose-noise tests before starting vision-conditioned perception.

## Non-Goals For The First Version

- Full in-hand manipulation.
- Tactile-feedback policies.
- Deformable object manipulation.
- A large 20+ hand-shape taxonomy before the small engineered set is validated.
- Claiming the library is non-privileged before RGBD or detector inputs replace
  simulator pose.
