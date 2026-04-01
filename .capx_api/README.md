# Local API Bash Profiles

Put non-OpenRouter API launch snippets here.

`capx/envs/launch.py` supports:

```bash
python capx/envs/launch.py \
  --api-bash-profile my_provider \
  --config-path env_configs/libero/franka_libero_goal_1.yaml
```

or:

```bash
python capx/envs/launch.py \
  --api-bash-file ~/.config/capx/my_provider.sh \
  --config-path env_configs/libero/franka_libero_goal_1.yaml
```

Rules for profile scripts:

- Use Bash.
- Print one CLI argument per line to stdout.
- Blank lines and lines starting with `#` are ignored.
- Put secrets in `.sh` files here; they stay gitignored.

Example output contract:

```bash
printf '%s\n' \
  --server-url \
  'https://your-endpoint.example/v1/chat/completions' \
  --api-key \
  'your-secret-key' \
  --model \
  'azure/openai/gpt-5.1'
```

Explicit CLI flags you pass after `--api-bash-profile` still win, because the profile args are injected before the rest of the command line.
