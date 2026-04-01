#!/usr/bin/env bash
set -euo pipefail

# Print one argument per line. launch.py will splice them into argv.
printf '%s\n' \
  --server-url \
  'https://your-endpoint.example/v1/chat/completions' \
  --api-key \
  'replace-with-your-api-key' \
  --model \
  'azure/openai/gpt-5.1'
