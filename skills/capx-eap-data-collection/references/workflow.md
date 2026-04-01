# Workflow Notes

## Repo Assumptions

- Expected repo root: the directory that contains `capx/`, `env_configs/`, and `outputs/`.
- The orchestration script auto-detects the repo root from `--repo-root`, the current working directory, or the current script location.
- The script prefers:
  1. `<repo-root>/.venv-libero/bin/python`
  2. `<repo-root>/.venv/bin/python`
  3. current interpreter

## Recommended Collection Patterns

### LIBERO EAP Smoke Test

Use a privileged config first:

- `env_configs/libero/franka_libero_goal_1_privileged.yaml`
- `env_configs/libero/franka_libero_spatial_0_privileged.yaml`

Recommended flags:

- `--enable-eap-rollback`
- `--enable-eap-recovery`
- `--total-trials 1` or `5`
- explicit `--output-dir`

### Non-Privileged Collection

Use non-privileged configs only after the visual service stack is ready. This may require:

- SAM3 access
- GraspNet
- PyRoKi
- `--use-img-differencing` for multi-turn visual feedback
- Prefer `--api-bash-profile <name>` if the repo stores provider credentials under `.capx_api/`
- Make sure the profile configures both the main planner and `--visual-differencing-model-server-url` / `--visual-differencing-model-api-key`, otherwise image differencing will still try `127.0.0.1:8110`.
- Use `codex_gemini_vdm` when you want Codex as the planner and Gemini as the visual differencing model.
- Prefer `--tmux-session <name>` for any multi-hour run so the workflow survives terminal disconnects.
- The pipeline script now applies stronger default model retry settings via environment variables and automatically cleans API service ports defined in the YAML before and after the run.

Example:

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

If ports are stuck from a previous run:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py cleanup-services \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --config-path env_configs/libero/franka_libero_goal_1.yaml
```

## Output Conventions

Raw collection output:

- `outputs/<model>/<run-id>` or another explicit directory you control
- per-trial layout under `trial_xx/attempt_yy`
- example:
  - `outputs/gpt-5.3-codex/my_run/trial_03/attempt_02/summary.txt`
  - `outputs/gpt-5.3-codex/my_run/trial_03/attempt_02/trial_metadata.json`
  - `outputs/gpt-5.3-codex/my_run/trial_03/attempt_02/transition_dataset/data.pkl.gz`

LeRobot export output:

- `outputs/lerobot/<run-id>`

LeRobot export structure produced by this repo:

- `data/chunk-000/file-000.parquet`
- `meta/info.json`
- `meta/stats.json`
- `meta/tasks.jsonl`
- `meta/tasks.parquet`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `meta/episodes/chunk-000/file-000.parquet`
- `videos/<video_key>/chunk-000/file-000.mp4`

## Validation Rules

After exporting, validate all of the following:

1. `meta/info.json` exists.
2. `meta/info.json` reports `dtype=video` features for image keys.
3. `LeRobotDatasetMetadata(...)` loads successfully.
4. `LeRobotDataset(...)[0]` returns low-dimensional state and decoded video frames.

By default, exporters skip attempts tagged with:

- `exclude_from_training = true`
- `exclusion_reason = "sim_limit_reset"`

This is intended to keep obviously invalid simulator-reset artifacts out of downstream training. If the user explicitly wants to keep those episodes, pass `--include-excluded`.

## Troubleshooting

- If `collect` fails immediately, check that the chosen environment is compatible with the selected venv.
- If `export` fails, verify that the collection directory contains `transition_dataset/data.pkl.gz`.
- If `validate` fails because Hugging Face cache paths are read-only, use the orchestration script's default `/tmp/capx_hf_cache` behavior or pass an explicit writable `--hf-cache-root`.
- If LeRobot validation fails, the most common causes are:
  - parquet schema mismatch
  - missing `videos/.../chunk_index` episode metadata
  - missing `videos/.../from_timestamp` episode metadata
  - absent `lerobot`, `datasets`, or `av` packages in the active environment
