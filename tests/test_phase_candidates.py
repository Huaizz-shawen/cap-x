from capx.envs.phase_candidates import infer_phase_candidates, phase_tags_from_candidates


def test_infer_phase_candidates_from_common_pick_place_code() -> None:
    code = """
open_gripper()
grasp_pos, grasp_quat = sample_grasp_pose("bowl")
goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
close_gripper()
goto_pose([0.1, 0.2, 0.3], grasp_quat)
open_gripper()
goto_home_joint_position()
"""

    candidates = infer_phase_candidates(code)
    tags = phase_tags_from_candidates(candidates)

    assert tags == ["prepare_grasp", "plan_grasp", "approach", "grasp", "transport", "release", "home"]
    assert candidates[0]["evidence_calls"] == ["open_gripper"]
    assert candidates[-1]["evidence_calls"] == ["goto_home_joint_position"]
