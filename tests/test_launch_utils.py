import json

from capx.utils.launch_utils import (
    _count_nonempty_code_blocks,
    _extract_code,
    _is_effectively_empty_code,
    _save_trial_artifacts,
)


def test_extract_code_handles_none_response() -> None:
    assert _extract_code(None) == [""]


def test_empty_code_helpers_ignore_annotation_only_blocks() -> None:
    assert _is_effectively_empty_code("# Code block 0\n") is True
    assert _count_nonempty_code_blocks(["# Code block 0\n", ""]) == 0
    assert _count_nonempty_code_blocks(["print('ok')"]) == 1


def test_save_trial_artifacts_writes_initial_scene_files(tmp_path) -> None:
    trial_dir = tmp_path / "out"
    config = {"output_dir": str(trial_dir)}
    all_responses = [
        {
            "initial_scene_prompt": [{"role": "user", "content": [{"type": "text", "text": "describe"}]}],
            "initial_scene_raw_response": "A black bowl is between the plate and the ramekin.",
            "initial_scene_reasoning": "Picked the most objective scene description.",
            "initial_scene_task_description": "Pick the black bowl between the plate and the ramekin.",
            "initial_model_finish_reason": "stop",
            "initial_model_raw_response": {"choices": [{"message": {"content": "print('ok')"}}]},
        }
    ]

    _save_trial_artifacts(
        config=config,
        trial=1,
        attempt_idx=1,
        sandbox_rc=0,
        reward=1.0,
        task_completed=True,
        final_code="print('ok')",
        raw_code="print('ok')",
        all_responses=all_responses,
        log_lines=["done"],
        visual_feedback_imgs=[],
    )

    prompt_dir = trial_dir / "trial_01" / "attempt_01" / "prompts_and_responses"
    assert json.loads((prompt_dir / "initial_scene_prompt.json").read_text()) == all_responses[0]["initial_scene_prompt"]
    assert (prompt_dir / "initial_scene_raw_response.txt").read_text() == all_responses[0]["initial_scene_raw_response"]
    assert (prompt_dir / "initial_scene_reasoning.txt").read_text() == all_responses[0]["initial_scene_reasoning"]
    assert (prompt_dir / "initial_scene_task_description.txt").read_text() == all_responses[0]["initial_scene_task_description"]
    assert (prompt_dir / "initial_model_finish_reason.txt").read_text() == all_responses[0]["initial_model_finish_reason"]
    assert json.loads((prompt_dir / "initial_model_raw_response.json").read_text()) == all_responses[0]["initial_model_raw_response"]
