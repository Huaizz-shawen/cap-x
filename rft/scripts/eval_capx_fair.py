#!/usr/bin/env python3
"""Fair CaP-X code-policy evaluation for base and RFT models.

This script intentionally keeps the evaluator model-agnostic:
- the same prompt construction is used for every model;
- generated text is passed to the same reward/environment function;
- success is measured by environment task completion, not by parser leniency.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")


def _patch_robosuite_body_name_compat() -> None:
    """Handle robosuite variants that expose mount/eef body names differently."""
    try:
        import cv2

        cv2.destroyAllWindows = lambda: None
    except Exception:
        pass

    try:
        import numpy as np
        import viser.transforms as vtf
        from capx.envs.simulators.robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel
        from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
    except Exception:
        return

    if getattr(RobosuiteBaseEnv, "_capx_body_name_compat_patched", False):
        return

    def _body_id(model: Any, candidates: tuple[str, ...]) -> int:
        last_error: Exception | None = None
        for name in candidates:
            try:
                return int(model.body_name2id(name))
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise ValueError("No body name candidates provided")

    def _joint_pos_from_sim(env: Any) -> Any:
        qpos = np.asarray(env.robosuite_env.sim.data.qpos, dtype=np.float64).copy()
        if qpos.shape[0] < 7:
            raise KeyError("robot0_joint_pos")
        return qpos[:7]

    def _gripper_qpos_from_state(env: Any) -> Any:
        return np.asarray(
            [float(getattr(env, "_gripper_fraction", 1.0)) * env.gripper_metric_length],
            dtype=np.float64,
        )

    def _ensure_robot_obs(env: Any, robosuite_obs: dict[str, Any]) -> None:
        if "robot0_joint_pos" not in robosuite_obs:
            robosuite_obs["robot0_joint_pos"] = _joint_pos_from_sim(env)
        if "robot0_gripper_qpos" not in robosuite_obs:
            robosuite_obs["robot0_gripper_qpos"] = _gripper_qpos_from_state(env)

    def _init_robot_links(self: Any) -> None:
        self.gripper_metric_length = 0.04
        model = self.robosuite_env.sim.model
        self.base_link_idx = _body_id(model, ("fixed_mount0_base", "mount0_base", "robot0_base"))
        self.gripper_link_idx = _body_id(
            model,
            ("gripper0_right_eef", "gripper0_eef", "gripper0_right_hand"),
        )
        self.base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )

    def _compute_gripper_obs(self: Any, robosuite_obs: dict[str, Any]) -> None:
        _ensure_robot_obs(self, robosuite_obs)
        gripper_robot_base = (
            vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz).inverse()
            @ vtf.SE3(wxyz_xyz=self.gripper_link_wxyz_xyz)
            @ vtf.SE3.from_rotation_and_translation(
                rotation=vtf.SO3.from_rpy_radians(0.0, 0.0, np.pi / 2.0),
                translation=np.array([0, 0, -0.107]),
            )
        )
        gripper_fraction = robosuite_obs["robot0_gripper_qpos"][0] / self.gripper_metric_length
        robosuite_obs["robot_joint_pos"] = np.concatenate(
            [robosuite_obs["robot0_joint_pos"], [gripper_fraction]]
        )
        robosuite_obs["robot_cartesian_pos"] = np.concatenate(
            [
                gripper_robot_base.translation(),
                gripper_robot_base.rotation().wxyz,
                [gripper_fraction],
            ]
        )

    def _cube_lift_reset(
        self: Any,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.robosuite_env.reset()
        self.robosuite_env.sim.data.qpos[6] -= np.pi
        self._step_count = 0
        self._sim_step_count = 0

        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        robosuite_obs = self.robosuite_env._get_observations()
        if "robot0_joint_pos" in robosuite_obs:
            self._current_joints = np.array(robosuite_obs["robot0_joint_pos"], dtype=np.float64)
            self._current_joints[6] -= np.pi
        else:
            self._current_joints = _joint_pos_from_sim(self)

        obs = self.get_observation()
        self.gripper_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.gripper_link_idx],
                self.robosuite_env.sim.data.xpos[self.gripper_link_idx],
            ]
        )
        info = {
            "task_prompt": "Place the primary cube on top of the secondary cube. Quaternions are WXYZ."
        }
        return obs, info

    def _move_to_joints_blocking(
        self: Any,
        joints: Any,
        *,
        tolerance: float = 0.02,
        max_steps: int = 100,
    ) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        self._current_joints = target

        steps = 0
        while steps < max_steps:
            robosuite_obs = self.robosuite_env._get_observations()
            current = np.array(
                robosuite_obs.get("robot0_joint_pos", _joint_pos_from_sim(self)),
                dtype=np.float64,
            )
            error = np.linalg.norm(current - target)
            if error < tolerance:
                break

            action = np.concatenate([target, [self._gripper_fraction, self._gripper_fraction]])
            action[-2:] = 1.0 - action[-2:] * 2.0
            self._do_robosuite_step(action)

            if hasattr(self, "viser_server") and self._sim_step_count % self._subsample_rate == 0:
                self._update_viser_server()

            if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
                self._record_frame()

            steps += 1
            self._sim_step_count += 1

    def _do_robosuite_step(self: Any, action: Any) -> None:
        sliced = action[: self._ACTION_SLICE] if self._ACTION_SLICE != 0 else action
        need_render = (
            self._record_frames and self._sim_step_count % self._subsample_rate == 0
        ) or hasattr(self, "viser_server")
        if need_render:
            self.robosuite_env.step(sliced)
            return
        try:
            self.robosuite_env.step(sliced, skip_render_images=True)
        except TypeError:
            self.robosuite_env.step(sliced)

    RobosuiteBaseEnv._init_robot_links = _init_robot_links
    RobosuiteBaseEnv._compute_gripper_obs = _compute_gripper_obs
    RobosuiteBaseEnv._do_robosuite_step = _do_robosuite_step
    RobosuiteBaseEnv.move_to_joints_blocking = _move_to_joints_blocking
    FrankaRobosuiteCubeLiftLowLevel.reset = _cube_lift_reset
    RobosuiteBaseEnv._capx_body_name_compat_patched = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="HF model path for generation.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSONL and summary outputs.")
    parser.add_argument("--label", default="model", help="Label stored in output records.")
    parser.add_argument("--data-source", default="franka_lift_code_env")
    parser.add_argument("--seed-base", type=int, default=20000)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed list. Overrides --seed-base/--num-trials.",
    )
    parser.add_argument("--samples-per-seed", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument(
        "--backend",
        choices=("vllm", "transformers"),
        default="vllm",
        help="Generation backend. transformers avoids vLLM worker subprocess CUDA init issues.",
    )
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=4,
        help="Prompt batch size for the transformers backend.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=1234,
        help="Global generation RNG seed, used by the transformers backend.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-model-len", type=int, default=1280)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager=True to vLLM.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--reward-workers",
        type=int,
        default=1,
        help="Number of parallel reward/environment workers.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalise_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalised: list[dict[str, str]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = "\n".join(parts)
        normalised.append({"role": str(msg.get("role", "user")), "content": str(content)})
    return normalised


def _build_prompt_messages(data_source: str) -> list[dict[str, str]]:
    _patch_robosuite_body_name_compat()
    from capx.envs.tasks import get_config, get_exec_env

    env = get_exec_env(data_source)(get_config(data_source))
    try:
        obs, _ = env.reset(seed=0)
        return _normalise_messages(obs["full_prompt"])
    finally:
        env.close()


def _prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    return [args.seed_base + idx for idx in range(args.num_trials)]


def _generate(args: argparse.Namespace, prompt: str, seeds: list[int]) -> list[dict[str, Any]]:
    if args.backend == "transformers":
        return _generate_transformers(args, prompt, seeds)
    return _generate_vllm(args, prompt, seeds)


def _generate_vllm(args: argparse.Namespace, prompt: str, seeds: list[int]) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    # Validate that the prompt is templated with the model tokenizer. The caller
    # passes prompt text because all trials for this task share the same prompt.
    _ = tokenizer

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(
        n=args.samples_per_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate([prompt] * len(seeds), sampling)

    records: list[dict[str, Any]] = []
    for seed, request_output in zip(seeds, outputs, strict=True):
        for sample_idx, completion in enumerate(request_output.outputs):
            records.append(
                {
                    "label": args.label,
                    "data_source": args.data_source,
                    "seed": seed,
                    "sample_idx": sample_idx,
                    "model_path": args.model_path,
                    "completion": completion.text,
                    "finish_reason": str(completion.finish_reason),
                    "token_ids": list(completion.token_ids),
                }
            )

    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return records


def _torch_dtype(dtype_name: str) -> Any:
    import torch

    aliases = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in aliases:
        raise ValueError(f"Unsupported dtype for transformers backend: {dtype_name}")
    return aliases[dtype_name]


def _generate_transformers(args: argparse.Namespace, prompt: str, seeds: list[int]) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.generation_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=_torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
        device_map=None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    requests: list[dict[str, Any]] = []
    for seed in seeds:
        for sample_idx in range(args.samples_per_seed):
            requests.append({"seed": seed, "sample_idx": sample_idx})

    records: list[dict[str, Any]] = []
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_tokens,
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else None,
        "top_p": args.top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample and args.top_k > 0:
        generation_kwargs["top_k"] = args.top_k
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    for batch_start in range(0, len(requests), args.gen_batch_size):
        batch = requests[batch_start : batch_start + args.gen_batch_size]
        encoded = tokenizer(
            [prompt] * len(batch),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_model_len,
        ).to(device)
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)
        input_len = encoded["input_ids"].shape[1]
        completion_ids = output_ids[:, input_len:]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for request, text, token_ids in zip(batch, completions, completion_ids.tolist(), strict=True):
            records.append(
                {
                    "label": args.label,
                    "data_source": args.data_source,
                    "seed": request["seed"],
                    "sample_idx": request["sample_idx"],
                    "model_path": args.model_path,
                    "completion": text,
                    "finish_reason": "unknown",
                    "token_ids": token_ids,
                }
            )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _score_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for record in records:
        scored.append(_score_one(record))
    return scored


def _score_one(record: dict[str, Any]) -> dict[str, Any]:
    _patch_robosuite_body_name_compat()
    from verl_agent_reward.hyrl_franka_reward import compute_score

    start = time.time()
    reward = compute_score(
        data_source=record["data_source"],
        solution_str=record["completion"],
        ground_truth={},
        extra_info={"seed": record["seed"]},
    )
    elapsed = time.time() - start
    if not isinstance(reward, dict):
        reward = {"score": float(reward)}
    reward = _jsonable(reward)
    # Main success metric: environment termination / task completion.
    task_success = bool(reward.get("terminated", False))
    valid_execution = not bool(reward.get("error"))
    out = dict(record)
    out.update(
        {
            "score": float(reward.get("score", 0.0) or 0.0),
            "task_success": task_success,
            "valid_execution": valid_execution,
            "terminated": bool(reward.get("terminated", False)),
            "truncated": bool(reward.get("truncated", False)),
            "error": str(reward.get("error", "")),
            "reward_raw": reward,
            "eval_seconds": elapsed,
        }
    )
    return out


def _score_batch(indices: list[int], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_score_one(records[idx]) for idx in indices]


def _score_records_parallel(records: list[dict[str, Any]], *, num_workers: int) -> list[dict[str, Any]]:
    if num_workers <= 1 or len(records) <= 1:
        return _score_records(records)
    from capx.utils.parallel_eval import run_parallel_batches

    batch_fn = functools.partial(_score_batch, records=records)
    scored = run_parallel_batches(
        list(range(len(records))),
        num_workers=num_workers,
        batch_fn=batch_fn,
    )
    scored.sort(key=lambda item: (int(item["seed"]), int(item["sample_idx"])))
    return scored


def _error_bucket(error: str) -> str:
    if not error:
        return "none"
    if "TimeoutError" in error:
        return "timeout"
    if "SyntaxError" in error:
        return "syntax"
    if "NameError" in error:
        return "name_error"
    if "TypeError" in error:
        return "type_error"
    if "ValueError" in error or "invalid literal" in error:
        return "value_error"
    return "other"


def _summary(
    args: argparse.Namespace,
    prompt_messages: list[dict[str, str]],
    prompt: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(records)
    successes = sum(1 for r in records if r["task_success"])
    valid = sum(1 for r in records if r["valid_execution"])
    scores = [float(r["score"]) for r in records]
    errors = Counter(_error_bucket(str(r.get("error", ""))) for r in records)
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), []).append(record)
    pass_at_k = {
        str(seed): any(item["task_success"] for item in items)
        for seed, items in sorted(by_seed.items())
    }
    return {
        "label": args.label,
        "model_path": args.model_path,
        "data_source": args.data_source,
        "num_trials": len(by_seed),
        "samples_per_seed": args.samples_per_seed,
        "num_records": total,
        "success_count": successes,
        "success_rate": successes / total if total else 0.0,
        "pass_at_k_success_count": sum(1 for ok in pass_at_k.values() if ok),
        "pass_at_k_success_rate": (
            sum(1 for ok in pass_at_k.values() if ok) / len(pass_at_k) if pass_at_k else 0.0
        ),
        "valid_execution_count": valid,
        "valid_execution_rate": valid / total if total else 0.0,
        "mean_score": statistics.fmean(scores) if scores else 0.0,
        "error_buckets": dict(errors),
        "seeds": sorted(by_seed),
        "generation": {
            "backend": args.backend,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "gen_batch_size": args.gen_batch_size,
            "generation_seed": args.generation_seed,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
        },
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_messages": prompt_messages,
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    prompt_messages = _build_prompt_messages(args.data_source)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    prompt = _prompt_text(tokenizer, prompt_messages)
    seeds = _seed_list(args)

    generated = _generate(args, prompt, seeds)
    scored = _score_records_parallel(generated, num_workers=args.reward_workers)
    summary = _summary(args, prompt_messages, prompt, scored)

    records_path = output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in scored:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
