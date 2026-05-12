#!/usr/bin/env bash
set -euo pipefail

# Recover notebook-side rtunnel + refresh bridge + reattach reverse forwards.
#
# Default target is the offline H200 notebook used for cap-x collection.
# Override only the variables you need:
#   NOTEBOOK_ID=... BRIDGE_ALIAS=... bash scripts/recover_c8_tunnel.sh
#
# This script:
# 1. Restores /tmp/rtunnel-31337 on the notebook from the shared Linux ELF copy
# 2. Starts the notebook-side rtunnel server on 31337
# 3. Refreshes a fresh bridge token via inspire-cli
# 4. Writes/updates an explicit SSH Host block in ~/.ssh/config
# 5. Reattaches 18086 -> local 8320 and 18447 -> local 8321 in tmux
# 6. Runs notebook-side probes and prints the results

NOTEBOOK_ID="${NOTEBOOK_ID:-c8b86c9c-5923-4c62-b566-26f59528850f}"
BRIDGE_ALIAS="${BRIDGE_ALIAS:-yuki-c8-reboot-manual}"
CODE_TARGET="${CODE_TARGET:-api.deepseek.com:443}"
VLM_LOCAL_PORT="${VLM_LOCAL_PORT:-8321}"
CODE_REMOTE_PORT="${CODE_REMOTE_PORT:-18086}"
VLM_REMOTE_PORT="${VLM_REMOTE_PORT:-18447}"
TMUX_CODE_SESSION="${TMUX_CODE_SESSION:-lisa_c8_code_18086}"
TMUX_VLM_SESSION="${TMUX_VLM_SESSION:-lisa_c8_vlm_18447}"
SHARED_RTUNNEL="${SHARED_RTUNNEL:-/inspire/hdd/project/exploration-topic/public/zzhuai/tmp/rtunnel-linux-amd64}"
SSH_CONFIG_PATH="${SSH_CONFIG_PATH:-$HOME/.ssh/config}"
INSPIRE_REPO="${INSPIRE_REPO:-/Users/kazusa/Documents/inspire-cli}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

need_cmd uv
need_cmd tmux
need_cmd python3
need_cmd ssh

run_inspire() {
  (cd "$INSPIRE_REPO" && uv run inspire "$@")
}

echo "[1/6] Restore notebook-side rtunnel-31337"
run_inspire notebook exec -n "$NOTEBOOK_ID" --timeout 90 -- bash -lc \
  "pids=\$(ss -lntp 2>/dev/null | awk '/:31337/ { while (match(\$0, /pid=[0-9]+/)) { print substr(\$0, RSTART+4, RLENGTH-4); \$0=substr(\$0, RSTART+RLENGTH) } }' | sort -u); \
   for pid in \$pids; do kill \"\$pid\" 2>/dev/null || true; done; \
   cp -f '$SHARED_RTUNNEL' /tmp/rtunnel-31337 \
   && chmod +x /tmp/rtunnel-31337; \
   nohup /tmp/rtunnel-31337 localhost:22222 0.0.0.0:31337 >/tmp/rtunnel-31337.log 2>&1 & \
   sleep 1; \
   ss -lntp | egrep '31337' || true; \
   file /tmp/rtunnel-31337; \
   tail -n 5 /tmp/rtunnel-31337.log 2>/dev/null || true"

echo "[2/6] Refresh bridge token"
bridge_output="$(cd "$INSPIRE_REPO" && uv run inspire notebook ssh "$NOTEBOOK_ID" --save-as "$BRIDGE_ALIAS" --timeout 90 --command 'echo bridge-ok' 2>&1)"
printf '%s\n' "$bridge_output"

remote_url="$(printf '%s\n' "$bridge_output" | python3 -c 'import sys
for line in sys.stdin:
    if line.startswith("Remote URL: "):
        print(line.split("Remote URL: ", 1)[1].strip())
        break')"

if [[ -z "$remote_url" ]]; then
  echo "failed to parse Remote URL from inspire output" >&2
  exit 1
fi

echo "[3/6] Ensure explicit SSH Host block exists in $SSH_CONFIG_PATH"
mkdir -p "$(dirname "$SSH_CONFIG_PATH")"
touch "$SSH_CONFIG_PATH"

python3 - "$SSH_CONFIG_PATH" "$BRIDGE_ALIAS" "$remote_url" <<'PY'
from pathlib import Path
import sys

cfg_path = Path(sys.argv[1])
alias = sys.argv[2]
remote_url = sys.argv[3]
text = cfg_path.read_text() if cfg_path.exists() else ""
lines = text.splitlines()

start = None
end = None
for i, line in enumerate(lines):
    if line.strip() == f"Host {alias}":
        start = i
        end = i + 1
        while end < len(lines) and not lines[end].startswith("Host "):
            end += 1
        break

block = [
    f"Host {alias}",
    "    HostName localhost",
    "    User root",
    "    Port 22222",
    f"    ProxyCommand /Users/kazusa/.local/bin/rtunnel '{remote_url}' stdio://%h:%p",
    "    StrictHostKeyChecking no",
    "    UserKnownHostsFile /dev/null",
    "    LogLevel ERROR",
    "    IdentityFile /Users/kazusa/.ssh/id_ed25519",
]

if start is None:
    if text and not text.endswith("\n"):
        text += "\n"
    text += ("\n" if text else "") + "\n".join(block) + "\n"
else:
    new_lines = lines[:start] + block + lines[end:]
    text = "\n".join(new_lines) + "\n"

cfg_path.write_text(text)
PY

echo "[4/6] Verify SSH alias connectivity"
ssh "$BRIDGE_ALIAS" "echo ssh-ok && ss -lntp | egrep '31337|${CODE_REMOTE_PORT}|${VLM_REMOTE_PORT}' || true"

echo "[4.5/6] Clear stale remote listeners on ${CODE_REMOTE_PORT}/${VLM_REMOTE_PORT}"
ssh "$BRIDGE_ALIAS" "pids=\$(ss -lntp 2>/dev/null | egrep ':(${CODE_REMOTE_PORT}|${VLM_REMOTE_PORT})\b' | awk '{ while (match(\$0, /pid=[0-9]+/)) { print substr(\$0, RSTART+4, RLENGTH-4); \$0=substr(\$0, RSTART+RLENGTH) } }' | sort -u); for pid in \$pids; do kill \"\$pid\" 2>/dev/null || true; done; ss -lntp | egrep '31337|${CODE_REMOTE_PORT}|${VLM_REMOTE_PORT}' || true"

echo "[5/6] Reattach reverse forwards in tmux"
tmux kill-session -t "$TMUX_CODE_SESSION" 2>/dev/null || true
tmux kill-session -t "$TMUX_VLM_SESSION" 2>/dev/null || true

tmux new-session -d -s "$TMUX_CODE_SESSION" \
  "zsh -lc 'while true; do ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R ${CODE_REMOTE_PORT}:${CODE_TARGET} ${BRIDGE_ALIAS}; sleep 2; done'"

tmux new-session -d -s "$TMUX_VLM_SESSION" \
  "zsh -lc 'while true; do ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N -R ${VLM_REMOTE_PORT}:127.0.0.1:${VLM_LOCAL_PORT} ${BRIDGE_ALIAS}; sleep 2; done'"

sleep 3
printf '== %s ==\n' "$TMUX_CODE_SESSION"
tmux capture-pane -pt "${TMUX_CODE_SESSION}:0" -S -12 || true
printf '\n== %s ==\n' "$TMUX_VLM_SESSION"
tmux capture-pane -pt "${TMUX_VLM_SESSION}:0" -S -12 || true

echo "[6/6] Notebook-side business probes"
ssh "$BRIDGE_ALIAS" "curl -sS -m 20 -k --resolve api.deepseek.com:${CODE_REMOTE_PORT}:127.0.0.1 -o /tmp/code_probe.out -w '%{http_code}\n' https://api.deepseek.com:${CODE_REMOTE_PORT}/v1/models || true; sed -n '1,3p' /tmp/code_probe.out 2>/dev/null || true"
ssh "$BRIDGE_ALIAS" "curl -sS -m 20 -k --resolve open.xiaojingai.com:${VLM_REMOTE_PORT}:127.0.0.1 -o /tmp/vlm_probe.out -w '%{http_code}\n' https://open.xiaojingai.com:${VLM_REMOTE_PORT}/v1/models || true; sed -n '1,3p' /tmp/vlm_probe.out 2>/dev/null || true"

echo
echo "Done."
echo "Bridge alias: $BRIDGE_ALIAS"
echo "Code tunnel:  127.0.0.1:${CODE_REMOTE_PORT} -> ${CODE_TARGET}"
echo "VLM tunnel:   127.0.0.1:${VLM_REMOTE_PORT} -> local ${VLM_LOCAL_PORT}"
