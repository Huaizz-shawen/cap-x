---
name: capx-eap-data-collection
description: "Orchestrate CaP-X EAP data collection runs and convert the resulting transition datasets into LeRobot video datasets for downstream robot training. Use when asked to launch LIBERO or robosuite collection jobs, batch-export CaP-X outputs to LeRobot dtype=video, or validate a local LeRobot dataset with the official loader."
metadata:
  short-description: "Run CaP-X collection and LeRobot export workflows"
---

# CaP-X EAP Data Collection

## When To Use

Use this skill when the user wants the outer Codex session to act as the workflow controller around `cap-x`, rather than using the `cap-x` agent loop alone.

Typical triggers:
- Launching `cap-x` collection runs for LIBERO or robosuite.
- Running EAP-style collection with rollback or recovery enabled.
- Exporting saved `transition_dataset` outputs into LeRobot `dtype=video`.
- Validating that a LeRobot dataset can be opened by the official `lerobot` loader.

Do not use this skill for editing policy code inside `cap-x` unless the user is explicitly asking for repository changes. This skill is about orchestration and dataset production.

## Preconditions

- The `cap-x` repository is available locally.
- Prefer running from the repo root.
- The repo should have `.venv` or `.venv-libero` available. This workflow prefers `.venv-libero` when present.
- Model-backed collection needs a reachable `--server-url`, `--api-key`, and `--model`.
- If the repo uses local API bash profiles under `.capx_api/`, prefer `--api-bash-profile` over hardcoding secrets in the command line. For visual-feedback runs, the profile should also set `--visual-differencing-model*` so image differencing does not fall back to `127.0.0.1:8110`. A mixed planner/VDM profile such as `codex_gemini_vdm` is a good default when you want Codex for code generation and Gemini for image understanding.
- Non-privileged visual configs may also require SAM3, GraspNet, or PyRoKi service setup. For a quick smoke test, prefer privileged LIBERO configs.

## Core Workflow

1. Choose a config and always pass an explicit `--output-dir`.
2. Run collection through `skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py collect` or `collect-export`.
3. Export the produced trial outputs through `skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py export`.
4. Validate the exported LeRobot dataset through `skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py validate`.
5. Report the collection directory, LeRobot output directory, and dataset size.

## Preferred Defaults

- For EAP development, start with LIBERO privileged configs because simulator rewind and rollback are easier to validate there.
- For non-privileged collection, prefer `--use-img-differencing` so the planner gets visual feedback across turns instead of relying only on stdout/stderr.
- For long-running collection, prefer `--tmux-session <name>` so terminal crashes do not kill the run.
- The orchestration script now cleans up configured API service ports before and after collection, so the next run should not inherit stale `8114/8115/8116` listeners.
- The orchestration script also exports stronger model retry defaults through environment variables, so transient `read timeout` / `503` failures do not immediately kill the workflow.
- Raw artifacts are now organized as `trial_xx/attempt_yy`, so retries and final outputs stay grouped under one trial instead of producing multiple ambiguous flat directories.
- Exporters skip samples marked `exclude_from_training=true` by default, especially `exclusion_reason=sim_limit_reset`. Use `--include-excluded` only when the user explicitly wants those samples.
- For training-oriented export, use LeRobot `dtype=video` with H.264 and a moderate `--crf` such as `30`.
- Keep collection outputs and LeRobot outputs separate:
  - raw collection under `outputs/<model>/<run-id>` or another explicit directory
  - exported LeRobot dataset under `outputs/lerobot/<run-id>`
- Validation uses a writable Hugging Face cache under `/tmp/capx_hf_cache` by default, so it should work even when `~/.cache/huggingface` is read-only.

## Fast Paths

Single collection run:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py collect \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --api-bash-profile codex_gemini_vdm \
  --config-path env_configs/libero/franka_libero_goal_1.yaml \
  --output-dir outputs/codex_key_goal1_10trials_vdm \
  --total-trials 10 \
  --num-workers 1 \
  --enable-eap-rollback \
  --enable-eap-recovery \
  --use-img-differencing \
  --tmux-session libero-goal1-vdm
```

Cleanup only:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py cleanup-services \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --config-path env_configs/libero/franka_libero_goal_1.yaml
```

Export only:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py export \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --input-root outputs/gpt-5.3-codex/my_goal_run \
  --output-root outputs/lerobot/my_goal_run \
  --crf 30
```

If the user explicitly wants to keep simulator-limit resets in the export:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py export \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --input-root outputs/gpt-5.3-codex/my_goal_run \
  --output-root outputs/lerobot/my_goal_run \
  --include-excluded
```

Validate only:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py validate \
  --dataset-root /media/user/B29202FA9202C2B91/cap-x/outputs/lerobot/my_goal_run
```

One-shot collect, export, and validate:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py collect-export \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --api-bash-profile codex_gemini_vdm \
  --config-path env_configs/libero/franka_libero_goal_1.yaml \
  --output-dir outputs/codex_key_goal1_10trials_vdm \
  --lerobot-output-root outputs/lerobot/codex_key_goal1_10trials_vdm \
  --total-trials 10 \
  --num-workers 1 \
  --enable-eap-rollback \
  --enable-eap-recovery \
  --use-img-differencing \
  --validate
```

## What To Read Next

- Read [references/workflow.md](./references/workflow.md) for concrete run patterns, directory conventions, and troubleshooting.
- Prefer using [scripts/capx_eap_pipeline.py](./scripts/capx_eap_pipeline.py) instead of retyping long commands.
