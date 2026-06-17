# RoboCasa GR1 PnPPouring Baseline

This note records the current GR1 Fourier-hand scripted baseline for RoboCasa
`PnPPouring`. It is intentionally documented as a privileged baseline harness,
not as a reusable general skill.

## Status

The current default candidate succeeds on the 4090 notebook smoke run for the
known `PnPPouring` setup:

- Remote validation run:
  `outputs/robocasa-gr1/diagnostics/gr1_pnp_case7_default_candidate_20260608_124212`
- Result: `task_completed=True`, `reward=1.0`
- `ball_xy_distance_to_target=0.03256m`
- `ball_vertical_offset_to_target=-0.01908m`
- `ball_3d_distance=0.03774m`
- `cup_target_collision_risk=false`
- `transition_count=180`

The candidate was selected from
`gr1_pnp_case7_yaw_roll_refine2_20260608_121829`, where 5 of 6 nearby variants
completed the task. The best case was refine2 case 3.

## Default Candidate

The canonical parameters are:

```python
{
    "mode": "cup",
    "arm": "right",
    "preset": "cylindrical",
    "cup_grasp_profile": "cylindrical_lower",
    "release_steps": 0,
    "move_steps": 95,
    "pour_steps": 30,
    "scale": 0.055,
    "cup_grasp_z_offset": -0.045,
    "cup_grasp_z_floor": -0.075,
    "cup_grasp_side_offset": 0.035,
    "target_z_offset": 0.47,
    "target_xy_offset": (0.08, 0.0),
    "use_pivot_pour": True,
    "use_object_pose_place": True,
    "object_place_axis_angle_delta": (0.0, 0.40, 0.0),
    "pivot_forward": 0.06,
    "pivot_drop": 0.08,
    "pivot_lift": 0.33,
    "final_pour_joint_deltas": {
        "robot0_r_wrist_yaw": 1.57,
        "robot0_r_wrist_roll": 1.50,
        "robot0_r_shoulder_yaw": 0.25,
    },
    "final_pour_joint_steps": 55,
}
```

These parameters are now fixed in:

- `scripts/run_robocasa_gr1_pnp_physical_smoke.py`
  - `cup_case7_default_v1`
  - default `--sweep-preset=cup_case7_default_v1`
- `capx/integrations/robocasa/gr1.py`
  - `solve_pnp_pouring_cup_physical()` default arguments
- `capx/envs/tasks/gr1/robocasa.py`
  - prompt guidance for code agents to prefer `solve_pnp_pouring_cup_physical()`

## Why This Is Not A General Skill

This harness currently uses privileged simulator information:

- It reads exact cup, ball, and bowl poses through scene/object pose helpers.
- The candidate is tuned to the current `PnPPouring` geometry and GR1 hand
  configuration.
- The final pour depends on fixed joint deltas and fixed height/pivot offsets.

Therefore, treat this as:

- a scripted oracle for collecting successful transitions,
- a debugging baseline for simulator, hand, and camera issues,
- a teacher trajectory source for downstream models.

Do not treat it as a transferable policy skill across RoboCasa tasks.

## Running The Smoke

On the 4090 notebook:

```bash
cd /inspire/hdd/global_user/huaizezheng-p-huaizezheng/cap-x

PYTHONPATH=.:third_party/robocasa_gr1/robocasa-gr1-tabletop-tasks:capx/third_party/robosuite \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
.venvs/robocasa-gr1-cpu-py310/bin/python \
  scripts/run_robocasa_gr1_pnp_physical_smoke.py \
  --output-dir outputs/robocasa-gr1/diagnostics/gr1_pnp_case7_default_candidate_$(date +%Y%m%d_%H%M%S) \
  --save-case-videos
```

Expected summary for the default seed is task success with reward `1.0`.

## Robustness Roadmap

The next useful work is not further single-case tuning. The next work should
measure whether the baseline survives progressively less privileged inputs.

### Stage 1: Pose-Noise Tolerance

Goal: quantify how much exact pose dependence exists.

Suggested tests:

- Add Gaussian noise to source cup position.
- Add Gaussian noise to target bowl position.
- Add yaw/orientation noise to the source cup.
- Sweep noise at small scales first, for example `0.5cm`, `1cm`, `2cm`, `4cm`.

Report:

- success rate,
- final ball XY distance,
- vertical offset,
- cup collision risk,
- video for first failure mode.

### Stage 2: Weak-Privileged Geometry

Goal: replace exact object poses with approximate scene estimates.

Suggested tests:

- Use only bowl center and cup center, not full exact object state.
- Quantize positions to image-like resolution.
- Keep object identity from the simulator, but degrade metric precision.

This stage answers whether the controller needs exact simulator state or only a
rough affordance target.

### Stage 3: Vision-Conditioned Inputs

Goal: use perception outputs instead of simulator truth.

Suggested tests:

- Estimate cup and bowl centers from rendered observations or a detector.
- Use camera-space or image-space center estimates projected into a coarse
  workspace coordinate.
- Keep the same cup-pouring primitive, but replace exact `get_object_pose()`
  calls with perception-derived estimates.

### Stage 4: Cross-Task Transfer

Goal: test whether the primitive transfers beyond this one task.

Suggested tasks:

- Other pouring-like RoboCasa tasks.
- Two-arm pouring variants where the source/target relation is similar.
- Tasks with different source-container shapes.

Criteria:

- If Stage 1 fails, the current method is too brittle for transfer.
- If Stage 1 passes but Stage 2 fails, it is a precise-pose controller.
- If Stage 2 passes, it is worth investing in Stage 3 perception wiring.
- If Stage 3 passes on more than one task, then consider extracting a reusable
  GR1 pouring skill.

## Current Recommendation

Keep this implementation as a documented baseline and use it to collect
successful transitions. For generalization, prioritize the staged robustness
tests above before adding more hand-authored task-specific parameters.
