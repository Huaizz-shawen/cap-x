# Project Narrative: Trajectory Quality as the Bottleneck in RFT for Robot Code Policies

## Core Question

This project studies whether reinforcement fine-tuning can improve an open-source code-policy/VLM for robot manipulation tasks, and more specifically what limits the effectiveness of RFT/GRPO in this setting.

Our central hypothesis is:

> For robot code-policy RFT, the quality of the rollout/data-collection policy is a major bottleneck. Direct self-improvement from a weak base model may fail because its rollout distribution contains sparse, noisy, and weakly structured successful trajectories. A stronger data collection policy can provide higher-quality trajectories, allowing the same RFT pipeline to produce better student-model performance.

The project should therefore be framed not as simply tuning GRPO hyperparameters, but as an investigation of how trajectory quality affects reinforcement fine-tuning.

## Current Evidence Snapshot

The latest experiments support a refined version of the story:

1. Direct GRPO from the base policy did not improve held-out success in the original baseline setting.
2. Base-model self-rollout produced some successful candidates, but the available self-rollout warmup result has not yet shown a reliable deployed-model gain under the strict local evaluation path.
3. Stronger teacher-policy rollouts are high quality under the local strict simulator: both the small pilot dataset and the large dataset are fully successful after environment scoring.
4. Student models trained by reward-filtered SFT / behavior cloning on teacher rollouts can be deployed locally and solve the task under the strict `terminated=True` metric, improving from the base model's local `49 / 100` success to `100 / 100`.
5. Inspire is currently reliable for training and checkpoint production, but the remote training workspace simulator/controller is not aligned with the local evaluator for this task. Therefore remote `0 / 100` strict-eval results for otherwise successful checkpoints should be treated as environment-mismatch diagnostics, not as final model-quality evidence.

Current measured results:

| Model / artifact | Data source | Evaluation setting | Strict success | Notes |
| --- | --- | --- | ---: | --- |
| GRPO smoke checkpoint | Base-policy online GRPO | Original held-out eval, seeds `30000..30019` | `4 / 20` | Same as base; no improvement |
| GRPO medium checkpoint | Base-policy online GRPO | Original held-out eval, seeds `40000..40099` | `38 / 100` | Base was `39 / 100`; no improvement |
| Self-rollout warmup data | Base-policy sampled candidates | Train-seed collection `50000..50127` | `174 / 512` | Successful candidate rate `0.3398`; deployed local eval still needs completion |
| Base model | None | Local strict held-out eval, seeds `60000..60099` | `49 / 100` | Valid execution `100 / 100`, mean score `0.7614` |
| GRPO medium checkpoint | Base-policy online GRPO | Local strict held-out eval, seeds `60000..60099` | `48 / 100` | Common successes with base: `48`; GRPO-only successes: `0` |
| Teacher pilot data | Stronger API policy | Local strict scoring | `128 / 128` | `27` unique completions |
| Teacher large data | Stronger API policy | Local strict scoring | `2048 / 2048` | `97` unique completions; label-backed SFT data rebuilt |
| Teacher-SFT pilot model | `128` teacher successes | Local strict held-out eval, seeds `60000..60099` | `100 / 100` | Valid execution `100 / 100`, mean score `1.0`; +51 points over base |
| Teacher-SFT large strict model | `2048` teacher successes | Local strict held-out eval, seeds `60000..60099` | `100 / 100` | Valid execution `100 / 100`, mean score `1.0`; +51 points over base |
| Remote strict oracle/template checks | Hand-written/oracle lift programs | Inspire training workspace simulator | `0 / 100` oracle, `0 / 10` strict templates | Same templates succeed locally; indicates simulator/controller mismatch |
| Local strict template check | Hand-written lift templates | Local strict simulator | `10 / 10` | Confirms the local environment can produce `terminated=True` |

## Evaluation Principle

All model comparisons should use a fair and model-agnostic evaluation protocol.

The final claims should be based on:

- Same task environment: `franka_lift_code_env`
- Same prompt template
- Same held-out seed range
- Same decoding parameters
- Same parser / code extraction logic
- Same environment evaluator
- Primary metric: `task_success == terminated`

Shaped reward scores can be reported as secondary diagnostics, but success rate should be the main metric.

This is important because the goal is to show a real task-success improvement, not an advantage caused by a more favorable parser or evaluation rule.

Important correction after the environment audit:

- The original VeRL reward returned a positive shaped score, often around `0.1`, even when `terminated=False`. This is useful as a diagnostic but should not be treated as success.
- The strict reward/evaluation rule should count success only when the environment reports `terminated=True`.
- The local workstation evaluator is currently the trusted strict evaluator for this project, because it can reproduce successful lift trajectories and matches the downloaded student checkpoints.
- The Inspire training workspace remains useful for GPU training, checkpoint merging, and artifact storage, but its simulator/controller version has produced false-negative strict-success evaluations for this lift task.

Method naming boundary:

- `Base-policy GRPO` means the candidate responses are sampled from the same base/student policy that is being optimized. This is the on-policy GRPO setting tested by the smoke and medium GRPO checkpoints.
- `Teacher-guided reward-filtered SFT` means a stronger external policy generates trajectories, the environment labels/filter them with the strict success rule, and the open-source student is trained to imitate the successful teacher programs.
- The teacher-guided runs in this project should not be described as "GRPO on the base policy", because their trajectories are not sampled from the base/student policy. They are better described as reward-filtered supervised fine-tuning, behavior cloning from successful teacher rollouts, or an offline trajectory-quality intervention.
- A true RL follow-up would first use teacher-guided SFT as a warm start, then let the resulting student perform its own rollouts, and finally run GRPO/RFT with the strict `terminated=True` reward on those student-generated trajectories.

## Stage 1: Base Self-Rollout Does Not Reliably Improve the Policy

The first part of the project uses the base model itself as the rollout/data-collection policy. This tests whether the repository baseline RFT/GRPO setup can directly improve the model through self-generated trajectories.

Existing results:

| Experiment | Evaluation seeds | Base success | Trained success | Conclusion |
| --- | ---: | ---: | ---: | --- |
| 2-step GRPO smoke | `30000..30019` | `4 / 20` | `4 / 20` | No measurable improvement |
| 32-step GRPO medium | `40000..40099` | `39 / 100` | `38 / 100` | No improvement; slightly worse |
| 32-step GRPO medium, local strict re-eval | `60000..60099` | `49 / 100` | `48 / 100` | No improvement; all GRPO successes are base successes |

Interpretation:

- The baseline training and evaluation pipeline works end-to-end.
- However, direct GRPO from the base policy does not improve held-out task success.
- The local strict re-evaluation strengthens this conclusion: the medium GRPO checkpoint has `48` successes, all of which overlap with the base model's successful seeds, while losing base success seed `60005`.
- This suggests that the problem is not merely running more infrastructure, but that the rollout distribution may not contain sufficiently useful learning signal.

We then add a self-rollout warmup experiment:

1. Sample multiple trajectories from the base model on train-only seeds.
2. Score each trajectory with the environment evaluator.
3. Keep only successful trajectories.
4. Normalize obvious formatting issues without changing the evaluator.
5. Train a LoRA SFT warmup model on these successful base rollouts.
6. Evaluate the merged student model on fresh held-out seeds.

Current self-rollout collection result:

| Source policy | Train seeds | Candidates | Successful candidates | Success rate |
| --- | ---: | ---: | ---: | ---: |
| Base model | `50000..50127` | `512` | `174` | `0.3398` |

The self-rollout warmup evaluation should be reported after completion. If it still does not improve held-out success, it strengthens the diagnosis that base-model self-generated successful trajectories are not enough to drive robust policy improvement.

Current status:

- The self-rollout warmup checkpoint exists as `/inspire/hdd/global_user/huaizezheng-p-huaizezheng/rl-homework/models/lora_sft_lift_success_warmup_0619_merged`.
- The available remote fair-eval result reported base success `0 / 100` and self-rollout warmup success `0 / 100`, but this run was affected by the later-discovered remote simulator/controller mismatch.
- A local strict evaluation of the self-rollout warmup checkpoint is still needed before making a final claim about this model.

## Stage 2: Stronger Data Collection Policy

The second part changes the data collection policy while keeping the student model and evaluator fixed.

Instead of using the base model to collect trajectories, we use a stronger model API as a teacher / behavior policy during training-time data collection.

The teacher model is only used to produce candidate trajectories on train-only seeds. Each candidate is still filtered by the same environment evaluator. The final evaluated model remains the open-source student model, not the teacher API.

This stage is not on-policy GRPO for the base model. It is a controlled test of whether high-quality successful trajectories can be transferred into the same open-source student through reward-filtered SFT / behavior cloning. That distinction matters: standard GRPO needs rollouts from the current training policy, while teacher rollouts are off-policy demonstrations unless we add a separate offline RL objective or importance-correction scheme.

Planned procedure:

1. Use a stronger API model to generate code trajectories for `franka_lift_code_env`.
2. Use the same environment evaluator to label success or failure.
3. Keep trajectories with `task_success=True`.
4. Construct a format-stabilized reward-filtered SFT dataset.
5. Fine-tune the same student base model.
6. Evaluate on fresh held-out seeds with the same fair protocol.

Practical offline workflow:

1. Run API-based teacher generation on a machine with external network access.
2. Save the generated candidates as raw JSONL. This step should not require CaP-X or GPUs.
3. Prefer local full-dataset scoring when the local CaP-X evaluator is available. This gives faster iteration and avoids spending platform jobs on CPU-heavy filtering.
4. Upload the raw candidates, scored records, and derived reward-filtered SFT data to Inspire global storage.
5. Run an Inspire-side re-score on the final dataset or a representative subset to check evaluator consistency.
6. Train the student model on the fixed dataset. The training job should not need external network access or online rollout from the teacher.
7. Train and merge the student checkpoint on Inspire, then download and evaluate it with the local strict evaluator until the remote simulator/controller mismatch is resolved.

Supporting scripts:

- `scripts/collect_api_completions.py`: local teacher API collection. It defaults to `capx.llm.client` from the local CaP-X repository, with a minimal OpenAI-compatible fallback available through `--client openai`.
- `scripts/score_external_completions.py`: Inspire-side offline environment scoring for externally generated completions.
- `scripts/run_score_external_completions_local.sh`: local CaP-X/PyRoKi wrapper for full-dataset scoring.

Local scoring dependency note:

- The local CaP-X path is `/media/user/B29202FA9202C2B91/cap-x`.
- The validated local venv is `/media/user/B29202FA9202C2B91/cap-x/.venv`.
- As checked on 2026-06-21, this venv contains `pyroki`, `open3d`, `gymnasium`, `mujoco`, `robot_descriptions`, and the reward entrypoint.
- A local smoke test with `data/api_smoke_0621/raw.jsonl` succeeded: `1 / 1` task success, score `1.0`, valid execution `1.0`.
- For robustness, final teacher datasets can still be spot-checked on Inspire, but the current evidence shows that Inspire strict-success results are not comparable to local strict-success results for this task. The main model-quality evidence should therefore use the local strict evaluator.

Measured teacher-policy data quality:

| Dataset | Raw candidates | Strict successful candidates | Unique normalized completions | Derived training data |
| --- | ---: | ---: | ---: | --- |
| Pilot teacher rollout | `128` | `128 / 128` | `27` | `data/sft_lift_teacher_codex_gemini_vdm_pilot_0621/train_all_success.jsonl` |
| Large teacher rollout | `2048` | `2048 / 2048` | `97` | `data/sft_lift_teacher_codex_gemini_vdm_large_0621/train_all_success_strict_0622.jsonl` |

Measured teacher-student deployment results:

| Student checkpoint | Training data | Optimizer steps / scale | Local strict held-out success | Output |
| --- | --- | ---: | ---: | --- |
| `Qwen2.5-Coder-7B-Instruct` | None | base model | `49 / 100` | `evals/local_fair_lift_base_100seeds_0623/teacher_sft` |
| `grpo_lift_medium_4h100_global_step32_hf` | base-policy GRPO | `32` GRPO steps | `48 / 100` | `evals/local_fair_lift_grpo_medium_100seeds_0623/teacher_sft` |
| `lora_sft_lift_teacher_codex_gemini_vdm_pilot_0621_merged` | `128` pilot teacher successes | `32` steps | `100 / 100` | `evals/local_fair_lift_teacher_sft_pilot_100seeds_0623/teacher_sft` |
| `lora_sft_lift_teacher_codex_gemini_vdm_large_strict_0622_merged` | `2048` strict teacher successes | larger strict SFT run | `100 / 100` | `evals/local_fair_lift_teacher_sft_large_strict_100seeds_0623/teacher_sft` |

These results show that the stronger data-collection policy can produce trajectories that transfer into the open-source student model under the strict local simulator. The result is especially useful because the teacher is not used at test time.

The key comparison should be:

| Model | Training data source | Test-time model | Purpose |
| --- | --- | --- | --- |
| Base | None | Base open-source model | Main baseline |
| Base-policy GRPO | Student/base policy online rollouts | Student model | Tests on-policy self-improvement |
| Self-rollout SFT warmup | Base model successful rollouts | Student model | Tests whether filtered base successes are enough |
| Teacher-guided reward-filtered SFT | Stronger API successful rollouts | Student model | Tests trajectory-quality hypothesis |
| Teacher API upper bound | None | Teacher API | Optional reference, not the main claim |

## Expected Claim

If teacher-collected trajectories produce a stronger student model while base-policy GRPO does not, the main conclusion should be:

> The same student model can improve substantially when trained on a high-quality, strictly verified trajectory distribution. The failure of direct base-policy GRPO is therefore likely caused by weak on-policy trajectory quality and sparse useful signal, rather than by the model architecture or evaluator being fundamentally unusable.

This should be written carefully. We should not claim that the RFT method is fully solved or that trajectory quality is the only bottleneck. A more defensible statement is:

> The experiments indicate that trajectory quality is a major bottleneck for applying RFT/GRPO to code-generating robot policies. Improving the data collection policy gives the student model a better learning distribution and can make the same training pipeline more effective.

Given the current evidence, the report should additionally state:

> We found that reward/evaluator alignment is as important as trajectory quality. Shaped reward values around `0.1` were not reliable indicators of task completion, and one remote simulator/controller configuration produced false-negative strict-success evaluations even for templates that succeeded locally. After switching to a strict `terminated=True` metric and a validated local evaluator, the base model achieved `49 / 100` held-out lift success, base-policy GRPO achieved `48 / 100`, and teacher-collected trajectories produced student checkpoints with `100 / 100` held-out lift success.

This keeps the narrative honest: the project is not only "teacher data is better", but also "the RL signal must be aligned with real task success before RFT results are meaningful."

The final report should avoid saying that the teacher-guided result is "teacher-data GRPO". A precise phrasing is:

> Direct on-policy GRPO from the base policy did not improve strict task success. In contrast, reward-filtered SFT on successful trajectories collected from a stronger teacher policy improved the same open-source student from `49 / 100` to `100 / 100` under the validated local strict evaluator. This suggests that trajectory quality is a key bottleneck for subsequent RFT/GRPO, and that teacher-guided data can provide an effective warm start before a future on-policy RL stage.

## Report Structure

A good final report can follow this structure:

1. **Motivation**
   - RFT/GRPO is promising for robot code policies, but direct self-improvement may fail when the rollout policy is weak.

2. **Fair Evaluation Setup**
   - Explain the task, model, prompt, evaluator, seed protocol, and success metric.

3. **Baseline RFT/GRPO Results**
   - Report the 2-step smoke run and 32-step medium run.
   - Emphasize that direct baseline GRPO does not improve held-out success.

4. **Diagnosis**
   - Discuss sparse success signal, noisy rollouts, and limited useful trajectory coverage from the base policy.

5. **Self-Rollout Warmup**
   - Describe successful-trajectory filtering from base rollouts.
   - Report whether this improves held-out success.

6. **Teacher-Guided Reward-Filtered SFT**
   - Describe the stronger data collection policy.
   - Explain that the teacher is not used at test time.
   - State explicitly that this is off-policy demonstration learning / behavior cloning, not base-policy GRPO.

7. **Results and Analysis**
   - Compare base, base-policy GRPO, self-rollout warmup, and teacher-guided reward-filtered SFT.
   - Analyze success overlap and failure cases when possible.
   - Separate model-quality failures from evaluator/environment mismatch.

8. **Conclusion**
   - Main conclusion: high-quality trajectory construction is crucial for effective RFT in robot code-policy settings.

## Experimental Standard Going Forward

Future experiments should preserve the following standards:

- Use train-only seeds for data collection.
- Use fresh held-out seeds for evaluation.
- Keep the evaluator unchanged across comparisons.
- Track success overlap, not only aggregate success rate.
- Archive negative results, because they support the trajectory-quality narrative.
- Avoid claiming improvements from parser changes or evaluation shortcuts.
- Treat shaped reward as secondary unless it is strongly tied to `terminated=True`.
- Keep remote training and local strict evaluation clearly separated until the simulator/controller mismatch is fixed.
- Use "GRPO" only for rollouts generated by the current training policy, and use "reward-filtered SFT" or "behavior cloning" for teacher-generated successful trajectories.
