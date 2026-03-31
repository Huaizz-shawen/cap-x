#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/sync_repo_skills.sh [--repo-root PATH] [--target-dir PATH] [--dry-run]

Synchronize repository-local Codex skills from ./skills into the target Codex skill directory.

Defaults:
  --repo-root   inferred from this script location
  --target-dir  $HOME/.codex/skills

Examples:
  scripts/sync_repo_skills.sh
  scripts/sync_repo_skills.sh --dry-run
  scripts/sync_repo_skills.sh --target-dir /tmp/codex-skills
EOF
}

REPO_ROOT=""
TARGET_DIR="${HOME}/.codex/skills"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --target-dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${REPO_ROOT}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

SOURCE_DIR="${REPO_ROOT}/skills"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Source skills directory not found: ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

echo "Repo root:   ${REPO_ROOT}"
echo "Source dir:  ${SOURCE_DIR}"
echo "Target dir:  ${TARGET_DIR}"

if command -v rsync >/dev/null 2>&1; then
  RSYNC_CMD=(
    rsync
    -a
    --delete
    --exclude='__pycache__/'
    "${SOURCE_DIR}/"
    "${TARGET_DIR}/"
  )

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    RSYNC_CMD=(
      rsync
      -av
      --dry-run
      --delete
      --exclude='__pycache__/'
      "${SOURCE_DIR}/"
      "${TARGET_DIR}/"
    )
  fi

  "${RSYNC_CMD[@]}"
else
  export SOURCE_DIR TARGET_DIR DRY_RUN
  python3 - <<'PY'
from __future__ import annotations

import os
import shutil
from pathlib import Path

source = Path(os.environ["SOURCE_DIR"]).resolve()
target = Path(os.environ["TARGET_DIR"]).resolve()
dry_run = os.environ["DRY_RUN"] == "1"

source_entries = {p.name for p in source.iterdir() if p.name != "__pycache__"}
target_entries = {p.name for p in target.iterdir()} if target.exists() else set()

for stale_name in sorted(target_entries - source_entries):
    stale_path = target / stale_name
    print(f"DELETE {stale_path}")
    if not dry_run:
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        else:
            stale_path.unlink()

for src_path in sorted(source.iterdir()):
    if src_path.name == "__pycache__":
        continue
    dst_path = target / src_path.name
    print(f"COPY   {src_path} -> {dst_path}")
    if dry_run:
        continue
    if dst_path.exists():
        if dst_path.is_dir():
            shutil.rmtree(dst_path)
        else:
            dst_path.unlink()
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)
PY
fi

echo "Synchronized repository skills into ${TARGET_DIR}"
