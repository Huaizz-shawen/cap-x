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

## Output Conventions

Raw collection output:

- `outputs/<model>/<run-id>` or another explicit directory you control

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

## Troubleshooting

- If `collect` fails immediately, check that the chosen environment is compatible with the selected venv.
- If `export` fails, verify that the collection directory contains `transition_dataset/data.pkl.gz`.
- If LeRobot validation fails, the most common causes are:
  - parquet schema mismatch
  - missing `videos/.../chunk_index` episode metadata
  - missing `videos/.../from_timestamp` episode metadata
  - absent `lerobot`, `datasets`, or `av` packages in the active environment
