from capx.envs.tasks.base import CodeExecutionEnvBase


PROMPT = """
You are controlling a Fourier GR1 robot in a RoboCasa kitchen task.
Goal: Complete the task described by the environment.
Write only executable Python code, without markdown fences.
The API functions below are already imported into the execution namespace.

Use this workflow:
1. Call get_task_state() first. It returns model-friendly object poses, robot state,
   success flags, and the action dimension.
2. Prefer get_object_pose(), get_named_positions(), move_towards_object(),
   get_scene_poses(), move_towards_position(), get_fourier_hand_spec(),
   get_rgbd_observation(), estimate_depth_grasp_pose(),
   validate_hand_presets(), plan_fourier_grasp(), open_hand(), set_hand_preset(), close_hand(),
   grasp_object(), lift_hand(), step_delta_action(), place_object(),
   solve_pnp_pouring_cup_physical(), solve_pnp_pouring_physical(),
   solve_transport_privileged(), hold_still(), and step_action() over manually
   parsing raw RoboCasa observations.
3. Some RoboCasa tasks do not expose object poses in observations. If
   state["objects"] is empty, use get_scene_poses() and look for MuJoCo names
   such as container_main, ball_obj_main, obj_container_main, pot handles, and
   relevant fixture handles.
   For PnPPouring, get_named_positions() and get_object_pose() expose
   ball_obj, container, and obj_container from those MuJoCo scene names.
   Treat ball_obj as the object being transferred, obj_container as the
   source/start cup, and container as the target/destination bowl. Prefer
   solve_pnp_pouring_cup_physical() for this task; its default parameters are
   the current validated GR1 cup-pouring candidate.
4. Raw observation key "object-state" is a flat numpy vector, not a dictionary.
   Do not call .keys() or .items() on it.
5. Use get_action_layout() before hand-writing low-level actions. The current
   validated layout is right arm IK 0:6, left arm IK 6:12, right Fourier hand
   12:18, and left Fourier hand 18:24.
6. Never treat empty success flags as success. Use state["task_completed"] as
   the primary completion signal; success flags are auxiliary diagnostics.
7. For the default TwoArmTransport smoke task, prefer solve_transport_privileged()
   when the goal is to validate collection/export plumbing and produce successful
   transition shards. It directly edits object state and is not a physically
   realistic robot policy primitive. For physical-control experiments, use the
   hand and motion helpers conservatively. Do not call solve_transport_privileged()
   for PnPPouring, TwoArmLift, or other non-Transport tasks.
8. For Fourier hand experiments, call get_fourier_hand_spec() first and use
   presets instead of raw 6D hand commands. Use grasp_object() for the first
   physical-control baseline; it plans pregrasp, grasp, close and lift stages
   without teleporting the object.
9. When geometric scene poses are not enough, use estimate_depth_grasp_pose()
   to connect visual depth observations to action parameters. It renders RGBD,
   builds a world-frame point cloud, crops around a named object/affordance, and
   returns compact grasp_pos/pregrasp_pos/approach_axis diagnostics. Treat it as
   a GraspNet-compatible perception bridge; do not dump raw RGBD arrays unless
   explicitly debugging camera calibration.

Minimal safe example:

state = get_task_state()
print(state["task_description"], state["objects"], state["success_flags"])
summary = solve_transport_privileged(settle_steps=12)
print(summary["success_flags"], summary["task_completed"])
hold_still(steps=2)

Minimal physical-control example:

state = get_task_state()
print(get_fourier_hand_spec())
plan = plan_fourier_grasp("payload", arm="auto", preset="power")
print(plan)
summary = grasp_object("payload", arm=plan["arm"], preset=plan["preset"])
print(summary["task_completed"], summary.get("grasp_plan"))

Minimal depth-to-grasp example:

grasp = estimate_depth_grasp_pose(
    camera_name="robot0_frontview",
    name_hints=["drawer_tabletop", "door_handle"],
    arm="right",
)
print(grasp["target"], grasp["crop_point_count"], grasp["grasp_pos"], grasp["pregrasp_pos"])

PnPPouring physical-control example:

state = get_task_state()
summary = solve_pnp_pouring_cup_physical(
    cup_name="obj_container",
    ball_name="ball_obj",
    target_name="container",
    arm="right",
)
print(summary["distance_to_target"], summary["task_completed"])
"""


class GR1RobocasaCodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for RoboCasa tasks with a GR1 robot."""

    prompt = PROMPT
    oracle_code = None


__all__ = ["GR1RobocasaCodeEnv"]
