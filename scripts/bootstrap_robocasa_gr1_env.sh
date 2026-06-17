#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-/inspire/hdd/global_user/huaizezheng-p-huaizezheng/cap-x}
REF_DIR=${REF_DIR:-/inspire/hdd/project/exploration-topic/public/zzhuai/Isaac-GR00T-robocasa}
VENV=${ROBOCASA_GR1_VENV:-$BASE_DIR/.venvs/robocasa-gr1-cpu-py310}
UV=${UV:-/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin/uv}
PYTHON=${PYTHON:-/usr/bin/python}
CACHE=${UV_CACHE_DIR:-$BASE_DIR/.uv-cache-robocasa-gr1}
LOG=${LOG:-$BASE_DIR/outputs/robocasa_gr1_env_bootstrap.log}
SRC_ROOT=${ROBOCASA_GR1_SRC_ROOT:-$BASE_DIR/third_party/robocasa_gr1}
ROBOCASA_GR1_REPO=${ROBOCASA_GR1_REPO:-https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git}
ROBOSUITE_LOCAL=${ROBOSUITE_LOCAL:-$BASE_DIR/capx/third_party/robosuite}
PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.org/simple}

mkdir -p "$BASE_DIR/outputs" "$BASE_DIR/.venvs" "$CACHE" "$SRC_ROOT"
export UV_CACHE_DIR="$CACHE"

exec > >(tee -a "$LOG") 2>&1

echo "=== robocasa gr1 env bootstrap $(date -Is) ==="
echo "BASE_DIR=$BASE_DIR"
echo "REF_DIR=$REF_DIR"
echo "VENV=$VENV"
echo "SRC_ROOT=$SRC_ROOT"
echo "PIP_INDEX_URL=$PIP_INDEX_URL"
"$UV" --version

if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --python "$PYTHON" "$VENV"
fi

clone_or_update() {
  local repo_url=$1
  local dest=$2
  if [ -d "$dest/.git" ]; then
    echo "Updating $dest"
    timeout 120 git -C "$dest" fetch --depth 1 origin || true
  else
    echo "Cloning $repo_url -> $dest"
    rm -rf "$dest"
    timeout 180 git clone --depth 1 "$repo_url" "$dest"
  fi
}

clone_or_update "$ROBOCASA_GR1_REPO" "$SRC_ROOT/robocasa-gr1-tabletop-tasks"

# Use cap-x's vendored robosuite when available. This avoids a second network clone
# and keeps the offline/GPU notebook path closer to the existing cap-x environment.
ROBOSUITE_SRC="$ROBOSUITE_LOCAL"
if [ ! -d "$ROBOSUITE_SRC/robosuite" ]; then
  echo "ERROR: vendored robosuite not found at $ROBOSUITE_SRC" >&2
  exit 2
fi

"$UV" pip install --python "$VENV/bin/python" \
  numpy==1.26.4 gymnasium==1.0.0 pyyaml requests tqdm imageio h5py \
  opencv-python-headless==4.11.0.86 matplotlib==3.10.0 scipy pillow fastapi uvicorn imageio-ffmpeg \
  --index-url "$PIP_INDEX_URL"

# Runtime deps used by the GR1 harness and nearby cap-x simulator imports.
"$UV" pip install --python "$VENV/bin/python" \
  omegaconf viser rich tyro cloudpickle msgpack-numpy robot_descriptions yourdfpy \
  --index-url "$PIP_INDEX_URL"

"$UV" pip install --python "$VENV/bin/python" -e "$ROBOSUITE_SRC" --index-url "$PIP_INDEX_URL"
"$UV" pip install --python "$VENV/bin/python" -e "$SRC_ROOT/robocasa-gr1-tabletop-tasks" --index-url "$PIP_INDEX_URL"
if [ -d "$REF_DIR" ]; then
  "$UV" pip install --python "$VENV/bin/python" --no-deps -e "$REF_DIR" --index-url "$PIP_INDEX_URL"
else
  echo "WARN: REF_DIR not found, skipping optional reference install: $REF_DIR"
fi

# robocasa currently pulls opencv-python through dependencies; remove it so CPU
# notebooks without libGL can still import cv2 via opencv-python-headless.
"$UV" pip uninstall --python "$VENV/bin/python" opencv-python || true
"$UV" pip install --python "$VENV/bin/python" opencv-python-headless==4.11.0.86 --index-url "$PIP_INDEX_URL"

"$VENV/bin/python" - <<'PY'
import importlib.util
mods = ["gymnasium", "cv2", "omegaconf", "viser", "robot_descriptions", "robosuite", "robocasa"]
for name in mods:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {bool(spec)} {spec.origin if spec else ''}")
PY

echo "BOOTSTRAP_DONE"
