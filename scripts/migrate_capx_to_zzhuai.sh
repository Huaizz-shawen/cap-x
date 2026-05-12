#!/usr/bin/env bash
set -euo pipefail

# Migrate cap-x repo from personal workspace to shared zzhuai workspace.
# Designed to be re-runnable and safe against partial previous attempts.
#
# Defaults match the current Inspire notebook paths.
SRC="${SRC:-/inspire/hdd/project/exploration-topic/huaizezheng-p-huaizezheng/cap-x}"
DST="${DST:-/inspire/hdd/project/exploration-topic/public/zzhuai/cap-x}"
KEEP_OLD_SYMLINK="${KEEP_OLD_SYMLINK:-1}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[error] missing required command: $1" >&2
    exit 1
  }
}

need_cmd du
need_cmd df
need_cmd find
need_cmd rsync
need_cmd rg

if [[ "$SRC" == "$DST" ]]; then
  echo "[error] SRC and DST are identical: $SRC" >&2
  exit 1
fi

echo "[info] src=$SRC"
echo "[info] dst=$DST"

if [[ ! -e "$SRC" && ! -e "$DST" ]]; then
  echo "[error] neither SRC nor DST exists; nothing to migrate" >&2
  exit 1
fi

if [[ -L "$SRC" ]]; then
  link_target="$(readlink "$SRC" || true)"
  if [[ "$link_target" == "$DST" ]]; then
    echo "[ok] old path already points to destination symlink: $SRC -> $DST"
    exit 0
  fi
fi

if [[ -e "$SRC" ]]; then
  src_size_kb="$(du -sk "$SRC" | awk '{print $1}')"
else
  src_size_kb=0
fi

mkdir -p "$(dirname "$DST")"
avail_kb="$(df -Pk "$(dirname "$DST")" | awk 'NR==2{print $4}')"
echo "[info] src_size_kb=$src_size_kb dst_avail_kb=$avail_kb"

if [[ "$src_size_kb" -gt 0 && "$src_size_kb" -ge "$avail_kb" ]]; then
  echo "[error] destination free space appears insufficient for full copy" >&2
  echo "        free=$avail_kb KB, needed~$src_size_kb KB" >&2
  exit 1
fi

if [[ ! -e "$SRC" && -d "$DST" ]]; then
  echo "[warn] source missing but destination exists; ensuring compatibility symlink only"
  if [[ "$KEEP_OLD_SYMLINK" == "1" ]]; then
    ln -sfn "$DST" "$SRC"
    echo "[ok] symlink restored: $SRC -> $DST"
  fi
  exit 0
fi

if [[ ! -d "$SRC" ]]; then
  echo "[error] source is not a directory: $SRC" >&2
  exit 1
fi

if [[ ! -d "$DST" ]]; then
  echo "[step] create destination: $DST"
  mkdir -p "$DST"
fi

echo "[step] rsync source -> destination (incremental/resumable)"
rsync -a --delete --info=stats2,progress2 "$SRC"/ "$DST"/

echo "[step] verify rsync delta is empty"
verify_out="$(rsync -ani --delete "$SRC"/ "$DST"/ | sed -n '1,40p')"
if [[ -n "$verify_out" ]]; then
  echo "[error] verification mismatch remains; refusing to cut over" >&2
  echo "$verify_out" >&2
  exit 1
fi

backup="${SRC}.bak.$(date +%Y%m%d_%H%M%S)"
echo "[step] move old source aside: $SRC -> $backup"
mv "$SRC" "$backup"

if [[ "$KEEP_OLD_SYMLINK" == "1" ]]; then
  echo "[step] create compatibility symlink"
  ln -s "$DST" "$SRC"
fi

echo "[step] quick absolute-path audit under scripts/.capx_api"
rg -n --hidden -S "huaizezheng-p-huaizezheng/cap-x|/inspire/hdd/project/exploration-topic/huaizezheng-p-huaizezheng/cap-x" \
  "$DST/scripts" "$DST/.capx_api" "$DST/env_configs" 2>/dev/null || true

echo
echo "[ok] migration completed"
echo "  dst: $DST"
echo "  backup: $backup"
if [[ "$KEEP_OLD_SYMLINK" == "1" ]]; then
  echo "  symlink: $SRC -> $DST"
fi
