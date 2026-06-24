# CaP-X Fork

### A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

[Project Page](https://capgym.github.io/) | [Paper](https://arxiv.org/abs/2603.22435) | [Fork](https://github.com/Huaizz-shawen/cap-x)

This repository is a personal fork of **CaP-X**. The `main` branch keeps the
upstream-style CaP-X framework and quick-start workflow, while the other
branches are experiment branches for data collection, humanoid robot extension,
Inspire platform deployment, and RFT / GRPO training research.

**Max Fu<sup>&#42;,1,2</sup>, Justin Yu<sup>&#42;,2</sup>, Karim El-Refai<sup>&#42;,2</sup>, Ethan Kou<sup>&#42;,2</sup>, Haoru Xue<sup>&#42;,1,2</sup>,
Huang Huang<sup>3</sup>, Wenli Xiao<sup>4</sup>, Guanzhi Wang<sup>1</sup>, Fei-Fei Li<sup>3</sup>, Guanya Shi<sup>4</sup>, Jiajun Wu<sup>3</sup>,
Shankar Sastry<sup>2</sup>, Yuke Zhu<sup>1</sup>, Ken Goldberg<sup>&dagger;,2</sup>, Jim Fan<sup>&dagger;,1</sup>**

<sup>1</sup>NVIDIA &ensp; <sup>2</sup>UC Berkeley &ensp; <sup>3</sup>Stanford University &ensp; <sup>4</sup>Carnegie Mellon University

<sup>&#42;</sup>Equal contribution &ensp; <sup>&dagger;</sup>Equal advising

---

## What CaP-X Provides

**CaP-X** is an open-access framework for systematically studying Code-as-Policy
agents in robot manipulation. It consists of four components:

| Component | What it does |
| --- | --- |
| **CaP-Gym** | Interactive Gymnasium environments where agents control robots by generating Python code that composes perception and control primitives. |
| **CaP-Bench** | Systematic benchmark evaluating coding agents across abstraction levels, interaction modes, and visual grounding modalities. |
| **CaP-Agent0** | Training-free agentic framework with multi-turn visual differencing, auto-synthesized skill libraries, and parallel ensembled reasoning. |
| **CaP-RL** | Reinforcement learning on the coding agent via GRPO, using environment rewards to post-train language models. |

---

## Fork Branches

Use `main` if you want the original CaP-X-style entry point. Use the experiment
branches when you want the corresponding research workflow.

| Branch | Purpose | Main additions |
| --- | --- | --- |
| [`main`](https://github.com/Huaizz-shawen/cap-x/tree/main) | Baseline CaP-X fork | Core CaP-X benchmark, simulator setup, evaluation launchers, upstream documentation and quick start. |
| [`data-collection`](https://github.com/Huaizz-shawen/cap-x/tree/data-collection) | Automated simulation data collection | LIBERO transition capture, EAP-style rollback / recovery, manifest-based collection, LeRobot video dataset export, service cleanup, and workflow tests. |
| [`humanoid-robot`](https://github.com/Huaizz-shawen/cap-x/tree/humanoid-robot) | Humanoid / GR1 manipulation experiments | RoboCasa GR1 simulator integration, GR1 skill adapter interface, PnPPouring / TwoArmLift / Drawer smoke scripts, Fourier hand validation, and GR1 skill-library documentation. |
| [`inspire-version`](https://github.com/Huaizz-shawen/cap-x/tree/inspire-version) | Inspire platform data-collection deployment | Inspire-friendly LIBERO collection scripts, persistent SAM3 / Molmo recovery helpers, notebook launch scripts, tunnel recovery utilities, randomized initial arm-joint configs, and transition-format compatibility fixes. |
| [`rft`](https://github.com/Huaizz-shawen/cap-x/tree/rft) | RFT / GRPO and trajectory-quality experiments | GRPO training and evaluation scripts, reward-filtered SFT from teacher rollouts, strict local evaluator, experiment notes, project narrative, and archived summaries for the lift task. |

### Branch Relationships

- `main` is the clean baseline branch for general CaP-X usage.
- `data-collection` adds the reusable automated data-production layer.
- `humanoid-robot` builds on the data-collection direction and focuses on
  RoboCasa / GR1 humanoid manipulation.
- `inspire-version` is an infrastructure branch for running the collection
  workflow on Inspire/HPC-style environments.
- `rft` is the training-research branch. It contains the data-collection and
  robot-workflow context needed for the RFT experiments, plus the additional
  training, scoring, and analysis artifacts.

To switch branches:

```bash
git fetch origin
git checkout data-collection      # automated collection workflow
git checkout humanoid-robot       # RoboCasa GR1 / humanoid workflow
git checkout inspire-version      # Inspire deployment workflow
git checkout rft                  # RFT / GRPO experiments
```

---

## Experiment Directions

### Automated Data Collection

The `data-collection` branch turns CaP-X evaluation into a dataset-production
workflow. It records low-level transitions, stores trial and retry metadata,
supports LIBERO state rewind and recovery, and exports collected trajectories
into LeRobot-compatible `dtype=video` datasets.

Key files:

| Path | Role |
| --- | --- |
| `skills/capx-eap-data-collection/` | Repository-local Codex skill for orchestrating collection, export, and validation. |
| `skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py` | Main collect / export / manifest / cleanup CLI. |
| `capx/data/export_lerobot_video_dataset.py` | Converts transition datasets into LeRobot video format. |
| `capx/envs/transition_dataset.py` | Transition dataset data model and serialization. |
| `tests/test_capx_eap_pipeline.py` | End-to-end workflow coverage for the orchestration script. |

Example collection command:

```bash
source .venv-libero/bin/activate
python skills/capx-eap-data-collection/scripts/capx_eap_pipeline.py collect-export \
  --repo-root "$PWD" \
  --api-bash-profile codex_gemini_vdm \
  --config-path env_configs/libero/franka_libero_goal_1.yaml \
  --output-dir outputs/codex_key_goal1_10trials_vdm \
  --lerobot-output-root outputs/lerobot/codex_key_goal1_10trials_vdm \
  --total-trials 10 \
  --num-workers 1 \
  --enable-eap-rollback \
  --enable-eap-recovery \
  --use-img-differencing \
  --detached \
  --validate
```

### Humanoid Robot / GR1

The `humanoid-robot` branch explores CaP-X-style code policies on RoboCasa GR1
tasks. It adds a GR1 simulator wrapper, robot API integration, reusable skill
adapters, and smoke tests for manipulation tasks such as pouring, lifting, and
drawer interaction.

Useful entry points:

| Path | Role |
| --- | --- |
| `capx/envs/simulators/robocasa_gr1.py` | RoboCasa GR1 simulator binding. |
| `capx/integrations/robocasa/gr1.py` | GR1 robot integration and control API. |
| `capx/skills/adapter_registry.py` | Skill-adapter registration layer. |
| `docs/robocasa-gr1-pnp-pouring.md` | PnPPouring baseline notes and roadmap. |
| `docs/robocasa-gr1-skill-library.md` | Reusable GR1 skill-library plan. |
| `scripts/run_robocasa_gr1_skill_interface_smoke.py` | GR1 skill-interface smoke run. |
| `scripts/validate_robocasa_gr1_skill_adapter.py` | Skill-adapter validation utility. |

### Inspire Platform Deployment

The `inspire-version` branch adapts the data-collection workflow for an Inspire
or HPC-like environment where services, notebooks, tunnels, and long-running
detached jobs need more explicit control.

Representative tools:

| Path | Role |
| --- | --- |
| `scripts/start_collect_notebook_18447.sh` | Starts a collection notebook workflow. |
| `scripts/start_collect_notebook_parallel_18447.sh` | Parallel notebook collection launcher. |
| `scripts/check_collect_run_status.sh` | Detached collection status helper. |
| `scripts/recover_sam3_8114.sh` | SAM3 service recovery helper. |
| `scripts/recover_molmo_8122.sh` | Molmo service recovery helper. |
| `scripts/recover_c8_tunnel.sh` | Tunnel recovery helper. |
| `env_configs/libero/*persistent_sam3*.yaml` | LIBERO configs for persistent service usage. |
| `env_configs/libero/*randinit*.yaml` | LIBERO randomized initial arm-joint experiments. |

### RFT / GRPO Experiments

The `rft` branch studies whether reinforcement fine-tuning improves
code-generating robot policies, and why direct self-improvement can fail when
the rollout policy is weak.

Current conclusion from the branch notes:

- Direct base-policy GRPO did not improve strict held-out lift success.
- Strict success should be measured with `terminated=True`, not only shaped reward.
- A remote training workspace produced false-negative strict-success results for
  otherwise successful local trajectories, so local strict evaluation is the
  trusted path for the documented lift results.
- Reward-filtered SFT from stronger teacher rollouts improved the same
  open-source student from `49 / 100` to `100 / 100` strict local held-out lift
  success in the recorded experiments.

Key files:

| Path | Role |
| --- | --- |
| `rft/docs/project_narrative.md` | Main project story, hypotheses, measured results, and reporting guidance. |
| `rft/docs/experiments.md` | Chronological experiment log. |
| `rft/scripts/train_lora_sft_warmup.py` | LoRA SFT training utility. |
| `rft/scripts/eval_capx_fair.py` | Fair model evaluation script. |
| `rft/scripts/collect_api_completions.py` | Teacher API candidate collection. |
| `rft/scripts/score_external_completions.py` | Offline environment scoring for generated completions. |
| `rft/verl_agent_reward/hyrl_franka_strict_reward.py` | Strict reward entry point for RFT experiments. |

---

## Installation

CaP-X uses [uv](https://docs.astral.sh/uv/) for dependency management. Requires
**Python 3.10** and a **CUDA-capable GPU**.

```bash
git clone --recurse-submodules https://github.com/Huaizz-shawen/cap-x.git
cd cap-x

# Or if already cloned without --recurse-submodules:
git submodule update --init --recursive

# Install uv (if not present)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.10
uv venv -p 3.10

# Base install
uv sync
```

### Simulator-Specific Setup

Pick **one** simulator family to install. Robosuite and LIBERO require different
environment setups because LIBERO uses its own Robosuite fork.

#### Robosuite

```bash
uv sync --extra robosuite
```

#### LIBERO-PRO

LIBERO requires a separate virtual environment:

```bash
uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --active --extra libero --extra contactgraspnet
```

See [docs/libero-tasks.md](docs/libero-tasks.md) for running LIBERO tasks.

#### BEHAVIOR (Isaac Sim)

BEHAVIOR tasks run on NVIDIA Isaac Sim via OmniGibson. Requires Python 3.10 and
CUDA 12.x.

```bash
cd capx/third_party/b1k
./uv_install.sh --dataset
cd ../../..

# Post-install fixes: copy cuRobo JIT headers to site-packages
cp capx/third_party/curobo/src/curobo/curobolib/cpp/*.h \
   $(python -c "import sysconfig; print(sysconfig.get_path('purelib'))")/curobo/curobolib/cpp/
```

The `--dataset` flag downloads robot assets, BEHAVIOR-1K scene/object assets,
and 2025 challenge task instances. You will be prompted to accept the NVIDIA
Isaac Sim EULA and BEHAVIOR dataset license. To auto-accept, add
`--accept-dataset-tos`.

For headless servers:

```bash
sudo apt-get update
sudo apt-get install -y libegl1 libgl1

# Remove duplicate Vulkan ICD if present. It can cause segfaults on multi-GPU systems.
sudo rm -f /usr/share/vulkan/icd.d/nvidia_icd.json
```

See [docs/behavior-tasks.md](docs/behavior-tasks.md) for task details and
expected baselines.

### Optional Extras

```bash
uv sync --extra verl             # RL training with VeRL / GRPO
uv sync --extra contactgraspnet  # Contact-GraspNet grasp planning
uv sync --extra curobo           # cuRobo GPU-accelerated IK and motion planning
```

---

## Quick Start

This section keeps the original CaP-X quick-start flow: start perception
services if needed, expose an OpenAI-compatible LLM endpoint, then launch an
evaluation.

### 1. Perception Servers

Perception servers such as SAM3, ContactGraspNet, and PyRoKi are auto-launched
by the YAML config when you run an evaluation. No manual setup is required for
most configs.

To pre-launch servers for sharing across multiple eval runs:

```bash
# Start SAM3 + GraspNet + PyRoKi with automatic GPU allocation
uv run --no-sync --active capx/serving/launch_servers.py --profile default
```

Use `--dry-run` to preview the allocation. Other profiles:

```bash
--profile full      # All perception servers: SAM3, GraspNet, PyRoKi, OWL-ViT, SAM2
--profile minimal   # PyRoKi only, for oracle / privileged evals
```

### 2. Set Up an LLM Proxy

The evaluation harness queries an LLM through a local proxy that exposes an
OpenAI-compatible API.

```bash
# OpenRouter: get a key at openrouter.ai/keys
echo "sk-or-v1-your-key-here" > .openrouterkey
uv run --no-sync --active capx/serving/openrouter_server.py --key-file .openrouterkey --port 8110
```

`.openrouterkey` is git-ignored. The default server URL in configs is
`http://127.0.0.1:8110/chat/completions`.

See [docs/configuration.md](docs/configuration.md) for all provider options,
including OpenRouter, NVIDIA, vLLM, and custom OpenAI-compatible providers.

### 3. Run Evaluation

```bash
# Robosuite: single-turn benchmark
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --model "google/gemini-3.1-pro-preview"

# Robosuite: multi-turn with visual differencing
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm.yaml \
  --model "google/gemini-3.1-pro-preview"

# LIBERO-PRO: spatial task, requires .venv-libero
source .venv-libero/bin/activate
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/libero/franka_libero_spatial_0.yaml \
  --model "google/gemini-3.1-pro-preview"

# BEHAVIOR: R1Pro radio pickup
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 \
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/r1pro/r1pro_pick_up_radio.yaml \
  --model "google/gemini-3.1-pro-preview"

# Interactive Web UI
uv run --no-sync --active capx/envs/launch.py \
  --config-path env_configs/cube_stack/franka_robosuite_cube_stack.yaml \
  --web-ui True
# Open http://localhost:8200

# Regression tests
./scripts/regression_test.sh quick
./scripts/regression_test.sh test1
```

For BEHAVIOR, Isaac Sim uses `OMNIGIBSON_GPU_ID` rather than
`CUDA_VISIBLE_DEVICES` to select the GPU. On multi-GPU systems, run perception
servers on a separate GPU when possible.

---

## Documentation

| Guide | Contents |
| --- | --- |
| [Adding Environments](docs/adding-environments.md) | Creating simulators, task environments, YAML configs. |
| [Adding APIs](docs/adding-apis.md) | Implementing and registering new robot control APIs. |
| [Configuration](docs/configuration.md) | YAML format, CLI flags, and LLM provider setup. |
| [LIBERO-PRO Tasks](docs/libero-tasks.md) | LIBERO setup, task running, and suite reference. |
| [BEHAVIOR Tasks](docs/behavior-tasks.md) | BEHAVIOR setup, R1Pro tasks, baselines, and environment variables. |
| [Development](docs/development.md) | Testing, linting, LIBERO / GraspNet setup, checkpoints, and known issues. |
| [Real-World Franka Panda Bringup](docs/real-franka.md) | Bringup with robots_realtime and real-robot QuickStart. |
| [RL Training](docs/rl-training.md) | CaP-RL with GRPO / VeRL and sim-to-real transfer. |
| [RFT Project Narrative](rft/docs/project_narrative.md) | Trajectory-quality hypothesis and RFT experiment summary. |
| [RFT Experiment Notes](rft/docs/experiments.md) | Chronological training, scoring, and evaluation notes. |
| [RoboCasa GR1 PnPPouring](docs/robocasa-gr1-pnp-pouring.md) | GR1 PnPPouring baseline and roadmap. |
| [RoboCasa GR1 Skill Library](docs/robocasa-gr1-skill-library.md) | Reusable GR1 skill-library plan. |

Some docs exist only on the branches that implement the corresponding workflow.
If a link is missing on `main`, switch to the relevant experiment branch.

---

## Citation

```bibtex
@inproceedings{fu2025capx,
  title     = {{CaP-X}: A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation},
  author    = {Fu, Max and Yu, Justin and El-Refai, Karim and Kou, Ethan and Xue, Haoru and Huang, Huang and Xiao, Wenli and Wang, Guanzhi and Li, Fei-Fei and Shi, Guanya and Wu, Jiajun and Sastry, Shankar and Zhu, Yuke and Goldberg, Ken and Fan, Jim},
  year      = {2025}
}
```

## License

This project is released under the [MIT License](LICENSE).
