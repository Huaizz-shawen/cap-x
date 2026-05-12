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
- Prefer `--detached` for any multi-hour run so the workflow survives IDE or terminal crashes. Use `--tmux-session <name>` only when you need interactive inspection.
- `collect-manifest` now launches one detached child job per task and, on rerun, resumes from the first missing trial for that task instead of starting the whole suite over.
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
  --detached
```

### Multi-Task Manifest Collection

Keep `launch.py` single-task and let the outer workflow expand a manifest into many single-task jobs.

Generate the standard four-suite LIBERO manifest:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py write-manifest \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --preset libero_standard_4 \
  --output-path outputs/manifests/libero_standard_4_2trials.yaml \
  --trials-per-task 2
```

Run only `libero_spatial` from that manifest:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py collect-manifest \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --api-bash-profile codex_gemini_vdm \
  --manifest-path outputs/manifests/libero_standard_4_2trials.yaml \
  --output-root outputs/libero_standard_4_spatial_10x2 \
  --suite-filter libero_spatial \
  --trials-per-task 2 \
  --num-workers 1 \
  --enable-eap-rollback \
  --enable-eap-recovery \
  --use-img-differencing \
  --detached
```

The repository also includes a checked-in preset template at `skills/capx-eap-data-collection/manifests/libero_standard_4.yaml`.

If ports are stuck from a previous run:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py cleanup-services \
  --repo-root /media/user/B29202FA9202C2B91/cap-x \
  --config-path env_configs/libero/franka_libero_goal_1.yaml
```

Detached runs write a log and pid file under:

- `outputs/detached_logs/<job>.log`
- `outputs/detached_pids/<job>.pid`

You can override those paths with `--detached-log-file` and `--detached-pid-file`, and replace a stale detached job with `--detached-replace-existing`.

### Inspire Offline Notebook Export (Shared `.venv` Pattern)

When the target collection notebook cannot reach the internet, use a paired online notebook with the same shared storage mount:

1. On the online notebook, install missing export dependencies into the shared venv:

```bash
cd /inspire/hdd/project/exploration-topic/public/zzhuai/cap-x
uv pip install --python ./.venv/bin/python pyarrow
```

2. Run collection on the target notebook as usual.
3. Export each attempt as a separate LeRobot shard:

```bash
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py export-shards \
  --repo-root /inspire/hdd/project/exploration-topic/public/zzhuai/cap-x \
  --input-root outputs/<model>/<run_id> \
  --output-root outputs/lerobot_shards/<run_id> \
  --fps 20 \
  --crf 30 \
  --cleanup-transition-dataset
```

4. Verify each shard has `manifest.json` and then keep only shard outputs.

This keeps storage pressure low by deleting `transition_dataset` immediately after successful export.

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
- per-attempt shards can also be stored under `outputs/lerobot_shards/<run-id>/<trial_xx__attempt_yy>`

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

## Storage Planning Notes

- Transition datasets are intermediate and can be much larger than final LeRobot shards.
- In one validated Inspire run:
  - raw run dir: `11,918,087 KB`
  - `transition_dataset` subtotal: `11,803,137 KB`
  - exported LeRobot shard root: `16,912 KB`
  - reclaimable after `--cleanup-transition-dataset`: about `11.26 GiB`
- For planning, treat long-term storage as shard size plus logs/metadata, not raw transition payload.

## Troubleshooting

- If `collect` fails immediately, check that the chosen environment is compatible with the selected venv.
- If `export` fails, verify that the collection directory contains `transition_dataset/data.pkl.gz`.
- If `validate` fails because Hugging Face cache paths are read-only, use the orchestration script's default `/tmp/capx_hf_cache` behavior or pass an explicit writable `--hf-cache-root`.
- If LeRobot validation fails, the most common causes are:
  - parquet schema mismatch
  - missing `videos/.../chunk_index` episode metadata
  - missing `videos/.../from_timestamp` episode metadata
  - absent `lerobot`, `datasets`, or `av` packages in the active environment
