# RL Homework Experiment Notes

## 2026-06-18: Lift GRPO Smoke Checkpoint Fair Evaluation

Question: does the very short GRPO smoke run already improve task success over the base model?

Training checkpoint:
- Job: `job-b8db7bc7-07e8-4f3b-b6ca-8e561e02c89e`
- Setup: `franka_lift_code_env`, Qwen2.5-Coder-7B-Instruct, GRPO, 4xH100, group size 2, 2 optimizer steps
- Merged HF model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/grpo_lift_smoke_4h100_global_step2_hf`

Fair evaluation:
- Job: `job-2759b4b1-875c-4dd9-8c3b-ff0ca427cf49`
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_transformers_pilot_2h100_pyroki_0618`
- Same prompt, same seeds, same Transformers generation backend, same reward/environment evaluator.
- Main metric: `task_success == terminated`; shaped reward score is secondary only.
- Seeds: `30000..30019`

Result:

| Model | Success | Success rate | Valid execution | Mean shaped score |
| --- | ---: | ---: | ---: | ---: |
| Base | 4 / 20 | 0.20 | 1.00 | 0.6155 |
| GRPO step 2 | 4 / 20 | 0.20 | 1.00 | 0.6165 |

Successful seeds:
- Base: `30003`, `30004`, `30011`, `30017`
- GRPO step 2: `30003`, `30004`, `30011`, `30019`

Interpretation:
- This run validates the end-to-end training, merging, and fair evaluation pipeline.
- It does not provide evidence of improved success rate. A 2-step smoke checkpoint is too short to solidify the behavior.
- The next experiment should increase rollout diversity and optimizer steps while keeping this evaluator unchanged.

## 2026-06-18: Planned Medium GRPO Run

Goal: collect more positive/negative rollout signal and train long enough for the lift behavior to stabilize.

Planned setup:
- Task: `franka_lift_code_env`
- Base model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/Qwen2.5-Coder-7B-Instruct`
- Train prompts: `1024`
- Validation prompts: `64`
- Train batch size: `64`
- Rollouts per prompt: `GROUP_SIZE=4`
- Epochs: `2`, approximately `32` optimizer steps
- Save frequency: `16`, so expected checkpoints around steps `16` and `32`
- Storage root: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/runs/capx_grpo_lift_medium_4h100_global_0618`

Run status:
- Initial job `job-102e46d5-0036-402f-ad0a-b878228a093b` failed immediately after dataset preparation with a truncated Python logging error, but it successfully wrote `train.parquet`, `test.parquet`, and `manifest.json`.
- Dataset validation after that failure: `train.parquet` has `1024` rows; `test.parquet` has `64` rows.
- Retry job: `job-48029253-84ff-4ce7-a0c1-221a89068e01`
- Retry log: `/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng/.inspire/training_master_20260619_062625.log`
- Retry result: succeeded in `30m30s`, completed `32/32` optimizer steps.
- Checkpoints:
  - `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/runs/capx_grpo_lift_medium_4h100_global_0618/checkpoints/hyrl/grpo_qwen25coder7b_franka_lift_code_env_0618_medium_temperature_1.0_group_size_4_train1024_epochs2/global_step_16`
  - `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/runs/capx_grpo_lift_medium_4h100_global_0618/checkpoints/hyrl/grpo_qwen25coder7b_franka_lift_code_env_0618_medium_temperature_1.0_group_size_4_train1024_epochs2/global_step_32`

Evaluation plan after training:
- Merge the final actor checkpoint to HF format.
- Run the same fair evaluator with held-out seeds, preferably `100+` trials.
- Primary claim should be based only on `task_success` / environment termination.

Fair evaluation result:
- Job: `job-479b49e3-398e-46d1-b939-f81d03626d65`
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_medium_step32_transformers_100seeds_2h100_0619`
- Merged model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/grpo_lift_medium_4h100_global_step32_hf`
- Seeds: `40000..40099`
- Same prompt, same Transformers generation backend, same evaluator, same success metric.

| Model | Success | Success rate | Valid execution | Mean shaped score |
| --- | ---: | ---: | ---: | ---: |
| Base | 39 / 100 | 0.39 | 1.00 | 0.7097 |
| GRPO step 32 | 38 / 100 | 0.38 | 1.00 | 0.7056 |

Success overlap:
- Common successes: `38`
- Base-only successes: `40003`
- GRPO-only successes: none

Interpretation:
- The medium GRPO run did not improve held-out task success. It nearly reproduces the base model's successful cases but loses one successful seed.
- This suggests the current reward/training setup is not yet turning rollout feedback into a more reliable code policy.
- Possible next changes: reduce generation format failures by adding an explicit no-code-fence supervised warm start, train with a parser-format reward component, or use successful rollouts as a separate SFT/RFT replay stage before another GRPO pass.

## 2026-06-20: Format-Stabilized Warmup Plan

Goal: improve the executable-code prior before another RL pass, without contaminating previous held-out evaluation seeds.

Plan:
1. Collect train-only candidates from the base model with stochastic sampling.
2. Score them with the same environment evaluator.
3. Keep only `task_success=True` completions and normalize obvious output-format issues such as markdown code fences.
4. Train a small LoRA SFT warmup model.
5. Merge LoRA into a normal HF model and evaluate on a fresh held-out seed range.

Candidate collection:
- Job: `job-d7b4364c-f7fe-4e92-9e9b-835bbd2fc4e1`
- Status: succeeded
- Model: base Qwen2.5-Coder-7B-Instruct
- Seeds: `50000..50127`
- Samples per seed: `4`
- Total candidates: `512`
- Success candidates: `174`
- Candidate success rate: `0.3398`
- Valid execution rate: `1.00`
- Candidate output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/data/warmup_lift_base_candidates_0619/base`
- SFT data: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/data/sft_lift_success_warmup_0619/train.jsonl`
- SFT examples after dedup/normalization: `174`

LoRA SFT warmup:
- Job: `job-f81192e6-275a-4e7a-b34d-8ec740a27168`
- Status: succeeded
- Training script: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_lora_sft_warmup_2h100_0619.sh`
- Adapter: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/runs/lora_sft_lift_success_warmup_0619/adapter_final`
- Merged model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_success_warmup_0619_merged`
- Setup: LoRA rank `16`, alpha `32`, dropout `0.05`, `3` epochs over `174` successful examples.
- Trainable parameters: `40,370,176 / 7,655,986,688` (`0.5273%`)
- Completed `33` optimizer steps in `87.6s`; final logged `train_loss=0.2137`.

Fresh held-out fair evaluation plan:
- Job: `job-d6943f93-2f50-4c99-9f41-b5ef75b81450`
- Initial status: queued at `2026-06-21 16:05:36`
- Log: `/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng/.inspire/training_master_20260621_080535.log`
- Evaluation script: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_fair_eval_lora_sft_warmup_2h100_0621.sh`
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_lora_sft_warmup_100seeds_2h100_0621`
- Seeds: `60000..60099`
- Compare base Qwen2.5-Coder-7B-Instruct vs merged LoRA SFT warmup model with the same Transformers backend and environment success metric.

## 2026-06-21: Local Teacher API and Scoring Smoke

Goal: validate that teacher API generation and local CaP-X environment scoring can be used for faster offline trajectory construction.

Teacher API:
- API args script: `/media/user/B29202FA9202C2B91/cap-x/.capx_api/codex_gemini_vdm.sh`
- Main endpoint: `http://10.11.18.197:8317/v1/chat/completions`
- Model: `gpt-5.5`
- Smoke response: returned `API_OK` in about `2.4s`

Candidate generation smoke:
- Script: `scripts/collect_api_completions.py`
- Prompt source: `archives/baseline_rft_no_improvement_20260619/medium_eval/base_summary.json`
- Output: `data/api_smoke_0621/raw.jsonl`
- Seeds: `79999`, `1` sample
- Result: generated one executable-looking lift program.

Local scoring smoke:
- Local CaP-X root: `/media/user/B29202FA9202C2B91/cap-x`
- Validated venv: `/media/user/B29202FA9202C2B91/cap-x/.venv`
- Script: `scripts/run_score_external_completions_local.sh`
- Output: `data/api_smoke_0621/scored_dotvenv`
- Result: `1 / 1` success, valid execution `1.0`, mean score `1.0`.

Interpretation:
- The local `.venv` is suitable for teacher candidate scoring after the PyRoKi server is started through the wrapper.
- For final evidence, teacher-collected datasets should still be re-scored or spot-checked on Inspire to guard against local/remote evaluator drift.

Teacher pilot collection:
- Script: `scripts/collect_api_completions.py`
- Output: `data/teacher_lift_codex_gemini_vdm_pilot_0621/raw.jsonl`
- Model: `gpt-5.5`
- Seeds: `70000..70031`
- Samples per seed: `4`
- Total candidates: `128`
- Empty completions: `0`
- API request concurrency: `4`

Teacher pilot local scoring:
- Script: `scripts/run_score_external_completions_local.sh`
- Output: `data/teacher_lift_codex_gemini_vdm_pilot_0621/scored_local`
- Local venv: `/media/user/B29202FA9202C2B91/cap-x/.venv`
- Success candidates: `128 / 128`
- Success rate: `1.0`
- Valid execution rate: `1.0`
- Mean score: `1.0`

Teacher SFT data:
- Deduplicated analysis data: `data/sft_lift_teacher_codex_gemini_vdm_pilot_0621/train.jsonl`
- Weighted training data: `data/sft_lift_teacher_codex_gemini_vdm_pilot_0621/train_all_success.jsonl`
- Training examples: `128`
- Unique normalized completions: `27`
- Interpretation: the teacher policy is extremely reliable for lift but produces a narrow family of concise successful programs. Training should use the weighted successful dataset, while reporting should mention the low unique-program count.

Teacher LoRA SFT training:
- Job: `job-5d4254a2-7b6a-4f24-b92d-fcad1c8a3e50`
- Initial status: queued at `2026-06-21 18:27:22`
- Resource: `1xH100`
- Log: `/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng/.inspire/training_master_20260621_102722.log`
- Training script: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_lora_sft_teacher_codex_gemini_vdm_2h100_0621.sh`
- Expected adapter: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/runs/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621/adapter_final`
- Expected merged model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged`

High-priority workspace copy attempt:
- Copied job: `job-d4163a89-acad-4aaa-8665-75b1d2baf7d3`
- Status: failed after `6s`
- Likely cause: copied job retained the old command wrapper/log path under `/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng`, which may not be mounted in the higher-priority workspace.
- Fix: use the global-only script `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_lora_sft_teacher_codex_gemini_vdm_global_only_0621.sh` and set the job working/log target to a visible global_user path.

High-priority global-only resubmissions:
- Teacher SFT job: `job-7c2f01ed-2c93-40f3-a8c5-dc1348e67f2a`
- Teacher SFT log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-lora-sft-teacher-codex-gemini-vdm-1h100-globalonly-0621_20260621_104650.log`
- Self-rollout warmup fair eval job: `job-c2923823-89a9-48f2-aebc-0ed66758fd98`
- Self-rollout warmup fair eval log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-fair-eval-self-lora-warmup-1h100-globalonly-0621_20260621_104650.log`
- Shared settings: project `project-f277dd4c-fa6b-4908-b75c-4416a0407a3d`, priority `10`, platform display priority `35`, workspace `ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6`, `1xH100`.

Teacher SFT high-priority dependency retry:
- Failed job: `job-7c2f01ed-2c93-40f3-a8c5-dc1348e67f2a`
- Failure mode: global-only job reached the high-priority workspace and visible global symlink, but failed at Python import with `ModuleNotFoundError: No module named 'peft'`.
- Fix: copied pure-Python `peft` and `accelerate` packages from the course `.venv-rl` into `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/vendor_site`, and updated the global-only training script to prepend this directory to `PYTHONPATH` when present.
- Retry job: `job-289aff5b-a2da-45a9-ada0-ca861335e8ed`
- Retry log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-lora-sft-teacher-codex-gemini-vdm-1h100-globalonly-vendor-0621_20260621_185519.log`
- Status check at `2026-06-21 18:56`: teacher retry `job_running`; self-rollout warmup fair eval `job-c2923823-89a9-48f2-aebc-0ed66758fd98` also `job_running`.

High-priority environment fixes:
- Teacher SFT retry `job-289aff5b-a2da-45a9-ada0-ca861335e8ed` then failed because the high-priority symlinked venv did not include `flash_attn` while the trainer forced `attn_implementation="flash_attention_2"`.
- Fix: made `scripts/train_lora_sft_warmup.py` accept `--attn-implementation`; the global-only teacher SFT launcher now passes `--attn-implementation sdpa`.
- Successful teacher SFT retry: `job-a3f89ef3-9d82-4687-aa06-6baf33d8d859`
- Successful teacher SFT log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-lora-sft-teacher-codex-gemini-vdm-1h100-globalonly-sdpa-0621_20260621_185822.log`
- Result: completed `32` optimizer steps, final train loss `0.1210`, merged model written to `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged`.
- Fair eval retry `job-c2923823-89a9-48f2-aebc-0ed66758fd98` failed because PyRoKi tried to clone `example-robot-data` inside the no-internet training workspace.
- Fix: uploaded local robot description cache to `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/cache/robot_descriptions/example-robot-data`; verified `panda_description` is present.
- Fair eval retry `job-d5118f66-351d-4dcb-8005-beae1ef2def0` passed PyRoKi startup but failed on robosuite body names (`fixed_mount0_base` absent; `mount0_base` present).
- Fix: added a fair-eval-only compatibility patch for robosuite body-name variants.
- Fair eval retry `job-be10388a-d1e4-45ef-b001-19e4d33942dc` passed body-name initialization but failed when prompt construction used `_get_observation()` before reset and `robot0_joint_pos` was absent.
- Fix: prompt construction now uses `env.reset(seed=0)` and then reads `full_prompt`, matching the public environment lifecycle.
- Fair eval retry `job-2b009f37-9829-404e-9187-c317ec8f7a63` passed prompt reset further than before, but failed because headless OpenCV did not implement `cv2.destroyAllWindows()`.
- Fix: the fair evaluator now patches `cv2.destroyAllWindows` to a no-op before constructing or scoring robosuite environments.
- Fair eval retry `job-882868d5-65bb-49de-86ee-168b4cb79dce` failed after PyRoKi/headless fixes because this high-priority robosuite environment did not include `robot0_joint_pos` in `_get_observations()` after reset.
- Fix: the fair evaluator now patches CaP-X robosuite lift reset and gripper observation construction to fall back to MuJoCo `sim.data.qpos` and the tracked gripper state when `robot0_joint_pos` or `robot0_gripper_qpos` is absent.
- Current fair eval retry: `job-fdbd6282-79ec-4137-9bfe-5984e59f0ca0`
- Current fair eval retry log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-fair-eval-self-lora-warmup-1h100-globalonly-jointfallback-0621_20260621_194357.log`
- Status check at `2026-06-21 19:45`: teacher SFT `job_succeeded`; fair eval joint-fallback retry `job_running` and already loading model checkpoint shards; both in high-priority project `project-f277dd4c-fa6b-4908-b75c-4416a0407a3d` with display priority `35`.
- Final status check with fixed `inspire job status`: fair eval joint-fallback retry `job_succeeded`, running time `10m 9s`.
- Final comparison: base success `0 / 100` (`0.0`), self-rollout warmup success `0 / 100` (`0.0`), delta `0.0`; both valid execution rates `1.0`.
- Mean score: base `0.012634777864726525`, self-rollout warmup `0.012769927609783785`.

Teacher SFT pilot fair evaluation:
- Job: `job-2f4a05c3-8207-47fb-98e7-681980871de9`
- Status: `job_succeeded`
- Running time: `10m 9s`
- Log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-fair-eval-teacher-sft-pilot-1h100-globalonly-0621_20260621_200558.log`
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_teacher_sft_pilot_100seeds_1h100_globalonly_0621`
- Model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged`
- Final comparison: base success `0 / 100` (`0.0`), teacher-SFT pilot success `0 / 100` (`0.0`), delta `0.0`; both valid execution rates `1.0`.
- Mean score: base `0.012430317903811857`, teacher-SFT pilot `0.01199444415530172`.
- Sample inspection: teacher-SFT generates concise executable lift-style code without parser errors, but the generated trajectories still do not terminate successfully under the fair environment success metric.
- Follow-up rerun with partial motion compatibility: `job-667e228d-06ab-41e4-bcf9-7b9e8f4c22d5`, output `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_teacher_sft_pilot_100seeds_1h100_globalonly_motionfix_0621`.
- Rerun summary file reported base success `0 / 100`, teacher-SFT pilot success `0 / 100`, base mean score `0.012227861491580474`, teacher-SFT mean score `0.012544214939971038`.
- However, the rerun log still contains repeated `TypeError: MujocoEnv.step() got an unexpected keyword argument 'skip_render_images'` during `home_pose()` execution. Treat this rerun as an environment-compatibility diagnostic, not a final fair metric. A later evaluator patch adds a fallback for robosuite versions without `skip_render_images`.
- Clean rerun after `skip_render_images` fallback: `job-6f048c18-7d79-4b32-8507-6358547cc774`.
- Clean rerun status: `job_succeeded`, running time `28m 49s`, created `2026-06-21 22:32:57`, finished `2026-06-21 23:02:10`.
- Clean rerun log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-fair-eval-teacher-sft-pilot-clean-1h100-0621_20260621_223256.log`.
- Clean rerun output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_teacher_sft_pilot_100seeds_1h100_globalonly_clean_0621`.
- Clean rerun final metrics: base success `0 / 100` (`0.0`), teacher-SFT pilot success `0 / 100` (`0.0`), delta `0.0`; both valid execution rates `1.0`.
- Clean rerun mean score: base `0.09502881440435315`, teacher-SFT pilot `0.1`.
- Clean rerun log check: no `KeyError`, no `TypeError`, no `skip_render_images`, and no `robot0_joint_pos` compatibility errors. There are `5` `NameError` tracebacks from generated code using missing `np`, which should be treated as model execution failures rather than evaluator/environment failures. This run is the first clean proof that the fair evaluation chain is usable for the small teacher-SFT model.

Large teacher dataset and overnight SFT:
- Raw teacher collection: `data/teacher_lift_codex_gemini_vdm_large_0621/raw.jsonl`
- Prompt source: `archives/baseline_rft_no_improvement_20260619/medium_eval/base_summary.json`
- Seeds: `71000..71511`
- Samples per seed: `4`
- Total completions: `2048`
- Empty completions: `0`
- Raw unique completions: `97`
- SFT data: `data/sft_lift_teacher_codex_gemini_vdm_large_0621/train_all_raw.jsonl`
- SFT examples: `2048`
- Normalized unique completions: `97`
- Note: full environment scoring of all `2048` completions was attempted locally, but PyRoKi startup and sequential scoring were too slow for the night schedule. This training set is therefore built from raw teacher completions, not full scored-success filtering. The earlier pilot had `128 / 128` local scorer success, so this is a pragmatic overnight training run rather than final filtered evidence.
- Training job: `job-9acfd9ea-3fdb-48ec-aab3-7cc726b2c77f`
- Training log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/inspire/capx-lora-sft-teacher-codex-gemini-vdm-large-1h100-8ep-0621_20260621_222532.log`
- Training script: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_lora_sft_teacher_codex_gemini_vdm_large_global_only_0621.sh`
- Expected merged model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_large_0621_merged`
- Setup: LoRA rank `16`, alpha `32`, dropout `0.05`, learning rate `5e-5`, effective batch size `16`, `8` epochs over `2048` examples, expected `1024` optimizer steps.
- Final status: `job_succeeded`, running time `35m 58s`.
- Result: completed `1024` optimizer steps, final train loss `0.04412402070011012`, adapter written to `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/runs/lora_sft_lift_teacher_codex_gemini_vdm_large_0621/adapter_final`, merged model written to `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_large_0621_merged`.

Large teacher SFT fair evaluation:
- First submission: `job-beb6758a-6beb-4b6b-8cde-cefbe1a4babd`.
- First submission issue: the `inspire job create` wrapper still used the project-level target `/inspire/hdd/project/machine-behavior/huaizezheng-p-huaizezheng`, so stdout/logging and working directory were on a path that is unreliable from the high-priority workspace. The job stayed `job_running` without producing eval outputs and was stopped after confirming the wrapper command.
- Successful submission: `job-30a23ce9-989c-4289-8d17-5d5d3b402fb2`.
- Successful submission target: `INSPIRE_TARGET_DIR=/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework`.
- Status: `job_succeeded`, running time `24m 13s`, created `2026-06-22 16:37:16`, finished `2026-06-22 17:01:47`.
- Log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/.inspire/training_master_20260622_083716.log`.
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_teacher_sft_large_100seeds_1h100_globalonly_clean_0622`.
- Model: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_large_0621_merged`.
- Final metrics: base success `0 / 100` (`0.0`), large teacher-SFT success `0 / 100` (`0.0`), delta `0.0`; both valid execution rates `1.0`.
- Mean score: base `0.09500612654240935`, large teacher-SFT `0.1`.
- Log check: no `KeyError`, no `TypeError`, no `skip_render_images`, no `robot0_joint_pos`, and no CUDA OOM. There are `5` `NameError` tracebacks from generated code, matching the clean small-model eval pattern and not indicating environment failure.
- Interpretation: scaling the raw teacher SFT data from `128` pilot examples to `2048` raw examples and training for `1024` optimizer steps did not improve the strict task success metric under the fair evaluator, although it preserves the higher shaped mean score around `0.1`. This suggests the next improvement should target trajectory quality/control details rather than simply increasing SFT quantity on the current raw teacher programs.

Oracle-code fair evaluation sanity check:
- Purpose: test whether the repository's own `FrankaLiftCodeEnv.oracle_code` can trigger the strict `terminated` success metric under the same patched fair evaluator. This separates model/RFT failure from control primitive or success-condition mismatch.
- Script: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/eval_capx_oracle_fair.py`.
- Launcher: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_fair_eval_oracle_global_only_0622.sh`.
- Job: `job-983b7a8c-93f2-4b05-bff1-c01c5f79ff9a`.
- Status: `job_succeeded`, running time `13m 24s`, created `2026-06-22 19:52:46`, finished `2026-06-22 20:06:57`.
- Log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/.inspire/training_master_20260622_115246.log`.
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/fair_lift_oracle_100seeds_1h100_globalonly_clean_0622`.
- Final metrics: oracle success `0 / 100` (`0.0`), valid execution rate `1.0`, mean score `0.1`, error buckets `{"none": 100}`.
- Log check: no `KeyError`, no `TypeError`, no `skip_render_images`, no `robot0_joint_pos`, no `Traceback`, no `NameError`, and no CUDA OOM.
- Interpretation: the canonical oracle program executes cleanly but still never reaches the strict `terminated` success condition. This strongly suggests the current `franka_lift_code_env` fair-eval failure is not primarily caused by model/RFT quality or missing VDM/SAM/GraspNet perception. The bottleneck is more likely in the privileged lift primitive/control path, grasp pose definition, low-level robosuite compatibility, or success-condition alignment.

Strict-success reward alignment and template search:
- Motivation: the original VeRL reward function returned `score = max(raw_reward, 0.1)` after sandbox-successful execution, while the `prime` reward manager only consumed the scalar `score`. This allowed non-terminating programs with tiny shaped rewards to look positive during RL/RFT data selection.
- Fix prepared: added `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/verl_agent_reward/hyrl_franka_strict_reward.py`, which returns `score = 1.0` only when `terminated=True`; otherwise it returns only a very small shaped tie-breaker `0.05 * max(raw_reward, 0.0)` for sandbox-clean non-success cases.
- Search script: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/search_strict_lift_success.py`.
- Launcher: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/scripts/run_search_strict_lift_success_global_only_0622.sh`.
- Job: `job-811b6e60-5505-4345-9ed1-43428e10a97c`.
- Status: `job_succeeded`, running time `17m 32s`, created `2026-06-22 20:37:17`, finished `2026-06-22 20:55:25`.
- Output: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/strict_lift_success_search_67templates_3seeds_0622`.
- Note: the output directory and job name say `67templates`, but the actual script contained `51` templates: `48` API parameter sweeps plus `api_oracle`, `api_hold_close_then_lift_high`, and `env_joint_hold_after_oracle`.
- Final metrics: `0 / 153` strict successes, success rate `0.0`, over `51` templates and `3` seeds.
- Record integrity: all `153` records had `sandbox_rc=0`, `terminated=False`, `task_completed=False`, `truncated=False`, and empty stderr. The two env-access templates also executed: `api_hold_close_then_lift_high` and `env_joint_hold_after_oracle`.
- Best observed shaped reward: `5.41407697271327e-05`.
- Best observed cube z delta: `-0.010107755420216802`; no tested template produced positive cube lifting.
- Interpretation: with the current privileged control API and robosuite Lift environment, neither the repository oracle nor a small hand-written sweep produces any strict `terminated=True` sample. This means the next experimental bottleneck is not just model data quality or more RFT steps. We first need at least one reliable strict-success trajectory, or we need to fix/realign the low-level lift primitive and success condition before treating generated rollouts as successful training data.

Local lift primitive diagnosis:
- Purpose: reproduce and debug the strict-success issue on the local machine, where iteration is faster than on the distributed training workspace.
- Added diagnostic script: `scripts/diagnose_lift_primitive_local.py`.
- Added local launcher: `scripts/run_search_strict_lift_success_local.sh`.
- Key local finding: the local CaP-X/robosuite environment can produce strict `terminated=True` trajectories for `franka_lift_code_env`. The bottom primitive and robosuite `_check_success()` are therefore not intrinsically unusable.
- `env.step` strict check, seed `60000`: `terminated=True`, `task_completed=True`, `sandbox_rc=0`, reward `1.0`, cube z delta about `0.402m`.
- Direct API home/no-home comparison over the first six diagnostic cases:
  - `home_q_0010_tcp_neg_dz+0.000`: success, reward `1.0`, cube z delta about `0.406m`.
  - `nohome_q_0010_tcp_neg_dz+0.000`: success, reward `1.0`, cube z delta about `0.404m`.
  - `home_q_0010_tcp_neg_dz-0.025`: success, reward `1.0`, cube z delta about `0.379m`.
  - `nohome_q_0010_tcp_neg_dz-0.025`: success, reward `1.0`, cube z delta about `0.368m`.
  - `dz+0.025` variants failed with shaped reward around `0.50`, indicating that targeting too high above the cube misses the grasp.
- Program ablation in local `env.step`: the old sweep-style program, the success template, and hold/no-hold variants all produced `terminated=True` on seed `60000`. This indicates the earlier remote `0 / 153` result is not explained only by bad program structure.
- Updated `scripts/search_strict_lift_success.py` to prepend two reliable strict templates:
  - `strict_home_single_hold_approach0.08_lift0.45`
  - `strict_nohome_single_hold_approach0.08_lift0.45`
- Local search over those two templates and seeds `60000..60004`: `10 / 10` strict successes, success rate `1.0`, reward `1.0` for every record, cube z delta about `0.408m..0.416m`.
- Version check: local and remote `scripts/eval_capx_fair.py` and `capx/integrations/franka/privileged.py` matched by hash, but remote `scripts/search_strict_lift_success.py` was stale before resync and remote `capx/envs/simulators/robosuite_cube_lift.py` differed from local. The remote file uses `load_controller_config_compat` while local uses robosuite's `load_composite_controller_config`. This controller/environment version difference is the leading explanation for the earlier remote failure and should be verified with a short updated remote strict-template search before using platform rollouts as training data.
- Remote strict-template check after resync: `job-da17892d-a507-45a7-9150-c879eb62ec39`, output `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/evals/strict_lift_success_remote_2tpl_5seed_0622`, completed `2026-06-22 21:26:05`, success `0 / 10`, max cube z delta `-0.010107755420216802`. The exact templates that were `10 / 10` locally still failed remotely, confirming an environment/controller mismatch in the training workspace.

Existing local trajectory label audit:
- Small teacher pilot raw data: `data/teacher_lift_codex_gemini_vdm_pilot_0621/raw.jsonl`, `128` raw completions, `27` unique completions.
- Small teacher pilot local scored data: `data/teacher_lift_codex_gemini_vdm_pilot_0621/scored_local/records.jsonl`, `128` records, all have `task_success=True`, `terminated=True`, `valid_execution=True`, score `1.0`.
- Small teacher pilot SFT data: `data/sft_lift_teacher_codex_gemini_vdm_pilot_0621/train_all_success.jsonl`, `128` examples. Source metadata keeps `score=1.0` and `task_success=True`, so this dataset is label-backed.
- Large teacher raw data: `data/teacher_lift_codex_gemini_vdm_large_0621/raw.jsonl`, `2048` raw completions, `97` unique completions.
- Large teacher scored data: `data/teacher_lift_codex_gemini_vdm_large_0621/scored_local/records.jsonl` does not exist; `scored_local/` only contains a PyRoKi log. The previous large SFT data was built from raw completions, not full environment-scored labels.
- Large teacher SFT data: `data/sft_lift_teacher_codex_gemini_vdm_large_0621/train_all_raw.jsonl`, `2048` examples. Source metadata has `raw_teacher_unscored=True` for all examples, not `task_success`.
- Local strict sample check for large raw: `diagnostics/large_teacher_sample_32_scored/summary.json`, first `32` large raw records over seeds `71000..71008`, success `32 / 32`, pass@k `9 / 9`, valid execution `32 / 32`, mean score `1.0`.
- Interpretation: we likely do not need to regenerate large teacher rollouts. Before using the large data as RFT-positive or final evidence, we should full-rescore the existing `2048` raw large completions locally and rebuild a `train_all_success` dataset from scored records. For plain SFT, the existing large raw SFT data is probably usable, but it should be described as unscored unless we regenerate the label-backed version.

Large teacher full local strict rescore:
- Purpose: determine whether the existing large teacher rollout data already has real local strict-success labels, or whether we need to regenerate rollouts before retraining.
- Input: `data/teacher_lift_codex_gemini_vdm_large_0621/raw.jsonl`, `2048` raw completions over `512` seeds with `4` samples per seed.
- Command: `REWARD_WORKERS=1 PYROKI_STARTUP_TIMEOUT=300 bash scripts/run_score_external_completions_local.sh data/teacher_lift_codex_gemini_vdm_large_0621/raw.jsonl data/teacher_lift_codex_gemini_vdm_large_0621/scored_local_strict_0622 large_teacher_strict_local_0622`.
- Output: `data/teacher_lift_codex_gemini_vdm_large_0621/scored_local_strict_0622/summary.json` and `records.jsonl`.
- Runtime: about `30` minutes on the local workstation with one scorer worker after PyRoKi startup.
- Final metrics: success `2048 / 2048`, success rate `1.0`; pass@k success `512 / 512`, pass@k success rate `1.0`; valid execution `2048 / 2048`; mean score `1.0`; error buckets `{"none": 2048}`.
- Field audit over `records.jsonl`: `task_success=True` for all `2048`, `terminated=True` for all `2048`, `valid_execution=True` for all `2048`, `score=1.0` for all `2048`.
- Rebuilt label-backed SFT data: `data/sft_lift_teacher_codex_gemini_vdm_large_0621/train_all_success_strict_0622.jsonl`.
- Rebuilt data summary: `data/sft_lift_teacher_codex_gemini_vdm_large_0621/train_all_success_strict_0622.summary.json`.
- Dataset build settings: kept duplicates with `--no-dedup`, yielding `2048` examples from `97` unique normalized completions; skipped `0` not-success, `0` empty, and `0` duplicate examples.
- Interpretation: the large teacher data does not need to be recollected for the lift task. The previous large SFT run should be described as trained on raw/unscored examples, but we now have a strict local label-backed version for the next training run. The main remaining risk is not teacher trajectory execution quality in the local environment; it is training/evaluation alignment and the remote training workspace's controller/environment mismatch.

Small teacher pilot model local deployment and fair evaluation:
- Purpose: verify that the previously trained pilot SFT model can be downloaded, loaded locally, and evaluated against the local strict simulator rather than the mismatched remote training workspace simulator.
- Downloaded archive: `models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged.tar.gz`, about `12G`.
- Extracted model: `models/lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged`, about `15G`, with `4` safetensors shards plus tokenizer/config files.
- Added local launcher: `scripts/run_fair_eval_teacher_sft_pilot_local.sh`.
- Smoke test output: `evals/local_fair_lift_teacher_sft_pilot_10seeds_0623/teacher_sft`, success `10 / 10`, valid execution `10 / 10`, mean score `1.0`.
- Full local fair-eval output: `evals/local_fair_lift_teacher_sft_pilot_100seeds_0623/teacher_sft`.
- Full local fair-eval command: `NUM_TRIALS=100 GEN_BATCH_SIZE=1 OUT_ROOT=/media/user/B29202FA9202C2B91/rl-homework/evals/local_fair_lift_teacher_sft_pilot_100seeds_0623 bash scripts/run_fair_eval_teacher_sft_pilot_local.sh`.
- Full local fair-eval metrics: success `100 / 100`, success rate `1.0`; pass@k success `100 / 100`; valid execution `100 / 100`; mean score `1.0`; error buckets `{"none": 100}`.
- Field audit over `records.jsonl`: `task_success=True`, `terminated=True`, `valid_execution=True`, and `score=1.0` for all `100` records.
- Interpretation: the pilot SFT model actually deploys and executes successfully in the local environment. Its earlier remote fair-eval failure should be attributed to the remote controller/environment mismatch rather than to the pilot SFT model being unable to produce executable lift code. This also validates the local-workstation evaluation path for comparing future checkpoints.

Large strict teacher model local deployment and fair evaluation:
- Purpose: evaluate the larger label-backed teacher-SFT checkpoint in the local strict simulator, using the same fair evaluation path as the pilot checkpoint.
- Downloaded archive: `models/lora_sft_lift_teacher_codex_gemini_vdm_large_strict_0622_merged.tar.gz`, about `12G`.
- Extracted model: `models/lora_sft_lift_teacher_codex_gemini_vdm_large_strict_0622_merged`, about `15G`.
- Full local fair-eval output: `evals/local_fair_lift_teacher_sft_large_strict_100seeds_0623/teacher_sft`.
- Full local fair-eval command: `MODEL_PATH=/media/user/B29202FA9202C2B91/rl-homework/models/lora_sft_lift_teacher_codex_gemini_vdm_large_strict_0622_merged MODEL_LABEL=teacher_sft_large_strict_local NUM_TRIALS=100 GEN_BATCH_SIZE=1 OUT_ROOT=/media/user/B29202FA9202C2B91/rl-homework/evals/local_fair_lift_teacher_sft_large_strict_100seeds_0623 bash scripts/run_fair_eval_teacher_sft_pilot_local.sh`.
- Full local fair-eval metrics: success `100 / 100`, success rate `1.0`; pass@k success `100 / 100`; valid execution `100 / 100`; mean score `1.0`; error buckets `{"none": 100}`.
- Field audit over `records.jsonl`: `task_success=True`, `terminated=True`, `valid_execution=True`, and `score=1.0` for all `100` records.
- Interpretation: the large strict teacher-SFT model also solves the local lift task under the strict `terminated=True` metric. Together with the pilot result, this supports the local-workstation evaluation path and suggests that the earlier remote `0 / 100` results are dominated by remote simulator/controller mismatch rather than by failure of the teacher-SFT checkpoints.

Base model local deployment and fair evaluation:
- Purpose: establish the local strict baseline under the same evaluator, seeds, prompt, parser, and decoding settings used for the teacher-SFT checkpoints.
- Downloaded model: `models/Qwen2.5-Coder-7B-Instruct`.
- Source: Hugging Face `Qwen/Qwen2.5-Coder-7B-Instruct`, downloaded directly into the project `models/` directory with local HF cache under `models/.hf_home`.
- Downloaded model size: about `15G`, with `4` safetensors shards and tokenizer/config files.
- Full local fair-eval output: `evals/local_fair_lift_base_100seeds_0623/teacher_sft`.
- Full local fair-eval command: `MODEL_PATH=/media/user/B29202FA9202C2B91/rl-homework/models/Qwen2.5-Coder-7B-Instruct MODEL_LABEL=base_local NUM_TRIALS=100 GEN_BATCH_SIZE=1 OUT_ROOT=/media/user/B29202FA9202C2B91/rl-homework/evals/local_fair_lift_base_100seeds_0623 bash scripts/run_fair_eval_teacher_sft_pilot_local.sh`.
- Full local fair-eval metrics: success `49 / 100`, success rate `0.49`; pass@k success `49 / 100`; valid execution `100 / 100`; mean score `0.761359735809887`; error buckets `{"none": 100}`.
- Field audit over `records.jsonl`: `task_success=True` and `terminated=True` for `49` records, `task_success=False` and `terminated=False` for `51` records, `valid_execution=True` for all `100` records.
- Interpretation: under the validated local strict evaluator, the base model is already competent on lift but is not saturated. Both teacher-SFT checkpoints improve from the base model's `49 / 100` strict success to `100 / 100` on the same held-out seed range and generation settings.

GRPO medium model local deployment and fair evaluation:
- Purpose: evaluate the downloaded medium GRPO checkpoint under the same local strict evaluator, seeds, prompt, parser, and decoding settings as the base and teacher-SFT checkpoints.
- Downloaded archive: `models/grpo_lift_medium_4h100_global_step32_hf.tar.gz`, about `12G`.
- Extracted model: `models/grpo_lift_medium_4h100_global_step32_hf`, about `15G`, with `4` safetensors shards plus tokenizer/config files.
- Full local fair-eval output: `evals/local_fair_lift_grpo_medium_100seeds_0623/teacher_sft`.
- Full local fair-eval command: `MODEL_PATH=/media/user/B29202FA9202C2B91/rl-homework/models/grpo_lift_medium_4h100_global_step32_hf MODEL_LABEL=grpo_medium_local NUM_TRIALS=100 GEN_BATCH_SIZE=1 OUT_ROOT=/media/user/B29202FA9202C2B91/rl-homework/evals/local_fair_lift_grpo_medium_100seeds_0623 bash scripts/run_fair_eval_teacher_sft_pilot_local.sh`.
- Full local fair-eval metrics: success `48 / 100`, success rate `0.48`; pass@k success `48 / 100`; valid execution `100 / 100`; mean score `0.7534934545103751`; error buckets `{"none": 100}`.
- Field audit over `records.jsonl`: `task_success=True` and `terminated=True` for `48` records, `task_success=False` and `terminated=False` for `52` records, `valid_execution=True` for all `100` records.
- Seed overlap with local base model: common successes `48`, base-only success seed `[60005]`, GRPO-only successes `[]`.
- Interpretation: the medium GRPO checkpoint does not improve the local strict success rate. It reproduces a subset of the base model's successful cases and loses one base success, matching the earlier diagnosis from the original held-out GRPO comparison.

GRPO medium checkpoint compression:
- Purpose: prepare the GRPO checkpoint for local download and evaluation.
- Remote checkpoint directory: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/grpo_lift_medium_4h100_global_step32_hf`.
- Compression output path: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/grpo_lift_medium_4h100_global_step32_hf.tar.gz`.
- Compression log: `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/logs/compress_grpo_lift_medium_4h100_global_step32_hf.log`.
- Compression status check on 2026-06-23: background process `750270` was still running; temporary archive `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/grpo_lift_medium_4h100_global_step32_hf.tar.gz.tmp` had reached about `4.1G` after roughly `10` minutes. The final `.tar.gz` should appear only after `tar` finishes and the temporary file is moved atomically.

Multi-task VeRL environment expansion: Franka restack smoke:
- Purpose: move beyond the saturated single-task lift setting by adding a second Franka code environment for multi-task GRPO/RFT experiments.
- Candidate environment: `franka_restack_code_env`.
- Added generic local launchers:
  - `rft/scripts/run_oracle_eval_local.sh`
  - `rft/scripts/run_fair_eval_local.sh`
- Oracle smoke command: `DATA_SOURCE=franka_restack_code_env NUM_TRIALS=3 SEED_BASE=62000 OUT_ROOT=/media/user/B29202FA9202C2B91/cap-x/rft/evals/local_oracle_franka_restack_3seeds_0624 bash rft/scripts/run_oracle_eval_local.sh`.
- Oracle smoke result: strict success `3 / 3`, valid execution `3 / 3`, mean score `1.0`.
- Base smoke command: `DATA_SOURCE=franka_restack_code_env MODEL_PATH=/media/user/B29202FA9202C2B91/rl-homework/models/Qwen2.5-Coder-7B-Instruct MODEL_LABEL=base_restack_local NUM_TRIALS=10 SEED_BASE=62000 GEN_BATCH_SIZE=1 OUT_ROOT=/media/user/B29202FA9202C2B91/cap-x/rft/evals/local_fair_franka_restack_base_10seeds_0624 bash rft/scripts/run_fair_eval_local.sh`.
- Base smoke result: strict success `0 / 10`, valid execution `10 / 10`, mean score `0.09012812600343614`.
- Observed base failure modes: empty completions, undefined variables such as `green_cube_extent`, wrong object-name variants such as `red_cube`, and incorrect placement target positions.
- Interpretation: `franka_restack_code_env` is a good next environment. The oracle can complete it under the local strict evaluator, while the base model is far from saturated. This makes it more useful than lift for testing multi-task generalization and subsequent GRPO/RFT.

Multi-source VeRL dataset builder smoke:
- Added script: `rft/scripts/prepare_verl_multisource_dataset.py`.
- Added training launcher: `rft/scripts/train_franka_grpo_multisource_global.sh`.
- Smoke command: `PYTHONPATH=/media/user/B29202FA9202C2B91/cap-x /media/user/B29202FA9202C2B91/cap-x/.venv/bin/python /media/user/B29202FA9202C2B91/cap-x/rft/scripts/prepare_verl_multisource_dataset.py --output-dir /tmp/capx_multisource_smoke --data-sources franka_lift_code_env,franka_restack_code_env --train-size-per-source 2 --val-size-per-source 1 --seed 63000 --jsonl-only`.
- Smoke result: train rows `4` total, with `2` from `franka_lift_code_env` and `2` from `franka_restack_code_env`; val rows `2` total, with `1` per environment.
- Local note: the CaP-X `.venv` used for simulation does not include `pyarrow`, so the local smoke used `--jsonl-only`. The Inspire training environment should use parquet output for VeRL; `pyarrow` is still required there.
- Interpretation: the mixed dataset format is compatible with VeRL's per-row `data_source` reward routing. The strict reward function already initializes environments by `data_source`, so multi-source GRPO should be feasible once the parquet dataset is built in the training environment.
