# LIBERO Tasks

CaP-X supports both the standard LIBERO suites and the extended LIBERO-PRO perturbation suites.

## Setup

LIBERO requires a **separate virtual environment** because its Robosuite fork conflicts with the standard Robosuite used by other tasks.

```bash
# 1. Create a dedicated LIBERO venv
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate

# 2. Install LIBERO dependencies (--active targets the activated venv)
uv sync --active --extra libero

# 3. Set up an LLM proxy (needed for code generation)
echo "sk-or-v1-your-key" > .openrouterkey
python capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

> **Note:** The PyRoKi IK server (port 8116) is auto-started by the YAML config. No manual setup needed.

## Running a Task

### Web UI (recommended for exploration)

```bash
source .venv-libero/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --web-ui True
```

### Headless evaluation

```bash
source .venv-libero/bin/activate
python capx/envs/launch.py \
    --config-path env_configs/libero/franka_libero_spatial_0.yaml \
    --total-trials 10
```

## Running the Batch Evaluator

The batch runner can now target either the standard four LIBERO suites or the LIBERO-PRO perturbation suites from the same entrypoint.

```bash
source .venv-libero/bin/activate
python capx/envs/scripts/run_libero_batch.py --suite-preset standard_4
python capx/envs/scripts/run_libero_batch.py --suite-preset pro_table2
```

You can also override the preset completely:

```bash
python capx/envs/scripts/run_libero_batch.py --suites libero_object libero_goal
```

## Reproducing the Standard Experiments

To reproduce the LIBERO experiments used during repository validation, use the fixed script below. It runs:

- `privileged` `libero_spatial` with the primary model
- `privileged` `libero_goal` with the primary model
- `privileged` `libero_spatial` with the comparison model
- `privileged` `libero_goal` with the comparison model
- `non-privileged` `libero_spatial` with the primary model
- `non-privileged` `libero_goal` with the primary model

```bash
source .venv-libero/bin/activate
bash scripts/reproduce_libero_standard_experiments.sh \
  --server-url http://10.11.18.197:8317/v1/chat/completions \
  --api-key YOUR_KEY
```

Defaults match the experiments we already ran:

- primary model: `gpt-5.3-codex`
- comparison model: `gpt-5.4`
- privileged trials: `5`
- non-privileged trials: `3`

You can override them:

```bash
bash scripts/reproduce_libero_standard_experiments.sh \
  --server-url http://10.11.18.197:8317/v1/chat/completions \
  --api-key YOUR_KEY \
  --primary-model gpt-5.3-codex \
  --compare-model gpt-5.4 \
  --privileged-trials 5 \
  --non-privileged-trials 3 \
  --output-root ./outputs/libero_repro
```

The script writes one log per run to `outputs/libero_repro/logs/` and a tabular summary to `outputs/libero_repro/summary.tsv`.

## Choosing a Task

Each LIBERO task is specified by a **suite name** and **task index**. The YAML config's `low_level` field follows the pattern:

```
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: <suite_name>
  task_id: <task_id>
```

For example, to run `libero_goal` task 2 ("put the wine bottle on top of the cabinet"), set:

```yaml
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 2
```

### Available Suites, LIBERO

| Suite | Tasks | Description |
|-------|-------|-------------|
| `libero_10` | 10 | Core benchmark tasks (multi-object, multi-step) |
| `libero_90` | 90 | Extended benchmark (diverse kitchen/living room scenes) |
| `libero_object` | 10 | Object generalization (same scene, different objects) |
| `libero_spatial` | 10 | Spatial generalization (same objects, different positions) |
| `libero_goal` | 10 | Goal generalization (same scene, different goals) |

### Available Suites, LIBERO-PRO
| `libero_10_swap` | 10 | Position perturbation on `libero_10` |
| `libero_10_task` | 10 | Task perturbation on `libero_10` |
| `libero_object_swap` | 10 | Position perturbation on `libero_object` |
| `libero_object_task` | 10 | Task perturbation on `libero_object` |
| `libero_spatial_swap` | 10 | Position perturbation on `libero_spatial` |
| `libero_spatial_task` | 10 | Task perturbation on `libero_spatial` |
| `libero_goal_swap` | 10 | Position perturbation on `libero_goal` |
| `libero_goal_task` | 10 | Task perturbation on `libero_goal` |

All other LIBERO-PRO perturbations (i.e. Obj, Sem, Env) are also supported. To view the full list with the exact suite names please run the following in your terminal.

```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark  # type: ignore[import-not-found]
benchmark_dict = benchmark.get_benchmark_dict(help=True)"
```

### Example Tasks

**libero_10:**
| Index | Task |
|-------|------|
| 0 | Put both the alphabet soup and the tomato sauce in the basket |
| 1 | Put both the cream cheese box and the butter in the basket |
| 2 | Turn on the stove and put the moka pot on it |

**libero_spatial:**
| Index | Task |
|-------|------|
| 0 | Pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | Pick up the black bowl next to the ramekin and place it on the plate |
| 2 | Pick up the black bowl from table center and place it on the plate |

**libero_goal:**
| Index | Task |
|-------|------|
| 0 | Open the middle drawer of the cabinet |
| 1 | Put the bowl on the stove |
| 2 | Put the wine bottle on top of the cabinet |

To list all tasks in a suite:

```bash
source .venv-libero/bin/activate
python -c "
from libero import benchmark
suite = benchmark.get_benchmark_dict()['libero_spatial']()
for i in range(suite.n_tasks):
    print(f'  [{i}] {suite.get_task(i).language}')
"
```

## Creating a Config for Any Task

Copy an existing config and change `suite_name` and `task_id`:

```bash
cp env_configs/libero/franka_libero_spatial_0.yaml env_configs/libero/franka_libero_goal_5.yaml
```

Example configs for the standard suites:

- `env_configs/libero/franka_libero.yaml` for `libero_10` non-privileged
- `env_configs/libero/franka_libero_10_0_privileged.yaml` for `libero_10` privileged
- `env_configs/libero/franka_libero_object_0.yaml` for `libero_object` non-privileged
- `env_configs/libero/franka_libero_object_0_privileged.yaml` for `libero_object` privileged
- `env_configs/libero/franka_libero_spatial_0.yaml` for `libero_spatial` non-privileged
- `env_configs/libero/franka_libero_spatial_0_privileged.yaml` for `libero_spatial` privileged
- `env_configs/libero/franka_libero_goal_1.yaml` for `libero_goal` non-privileged
- `env_configs/libero/franka_libero_goal_1_privileged.yaml` for `libero_goal` privileged

Then edit `suite_name` and `task_id` in the new file:

```yaml
low_level:
  _target_: capx.envs.simulators.libero.FrankaLiberoEnv
  suite_name: libero_goal
  task_id: 5
```

The task prompt is automatically populated from LIBERO's task language description via the `{libero_environment_goal}` placeholder.

Privileged configs only need `PyRoKi`. Non-privileged configs also start `SAM3` and `GraspNet`.

## Config Reference

```yaml
env:
  _target_: capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv
  cfg:
    _target_: capx.envs.tasks.base.CodeExecEnvConfig
    low_level:
      _target_: capx.envs.simulators.libero.FrankaLiberoEnv
      suite_name: libero_goal
      task_id: 5
    privileged: true                                       # true = ground-truth state
    apis:
      - FrankaLiberoPrivilegedApi                          # privileged API
    prompt: |
      You are controlling a Franka Emika robot with API described below.
      Goal: {libero_environment_goal}
      ...
```

## Available APIs

| API | Description |
|-----|-------------|
| `FrankaLiberoPrivilegedApi` | Ground-truth object poses, IK-based control (privileged) |
| `FrankaLiberoApi` | Perception-based control with SAM3 + GraspNet (requires servers) |
| `FrankaLiberoApiReduced` | Low-level abstractions for perception and control functions (requires servers) |
| `FrankaLiberoApiReducedSkillLibrary` | Low-level abstractions for perception and control functions + extra utility functions from automatically synthesized skill library (requires servers) |

## Using CuRobo
To use CuRobo uncomment the following functions in the API you want to use (i.e. `capx/integrations/franka/libero.py`, `capx/integrations/franka/libero_reduced.py`).

```python
def functions(self) -> dict[str, Any]:
  fns =  {
      ...
      # # CuRobo, uncomment these for the coding agent to use them!
      # "plan_grasp_trajectory": self.plan_grasp_trajectory,
      # "plan_with_grasped_object": self.plan_with_grasped_object,
      # "execute_joint_trajectory": self.execute_joint_trajectory,
  }
```
