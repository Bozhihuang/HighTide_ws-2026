#!/usr/bin/env bash
#
# deploy.sh — sync this workspace's source to the ZED Box and (optionally) rebuild.
#
# Usage:
#   ./deploy.sh              # sync src/ to the sub
#   ./deploy.sh build        # sync, then colcon build on the sub
#   ./deploy.sh build mission # sync, then build only the hightide_mission package
#
# One-time setup for passwordless sync (recommended — otherwise you type the
# password on every run):
#   ssh-keygen -t ed25519          # if you don't already have ~/.ssh/id_ed25519
#   ssh-copy-id user@192.168.2.1   # enter the password 'admin' once
#
set -euo pipefail

# ---- Config (edit REMOTE_WS if the workspace lives elsewhere on the box) ----
REMOTE_USER="user"
REMOTE_HOST="192.168.2.1"
REMOTE_WS="/home/user/HighTide_ws-2026"      # workspace root on the ZED Box
LOCAL_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

echo ">> Syncing ${LOCAL_WS}/src/  ->  ${REMOTE}:${REMOTE_WS}/src/"

# --delete mirrors the tree (removes files on the box that you deleted locally).
# Generated/vendored dirs are excluded so we only push source. The ZED wrapper
# is excluded because it's large and rarely changes — sync it once by hand, or
# remove the exclude line below if you need to push wrapper changes.
rsync -avz --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  --exclude 'build/' \
  --exclude 'install/' \
  --exclude 'log/' \
  --exclude 'zed-ros2-wrapper/' \
  "${LOCAL_WS}/src/" "${REMOTE}:${REMOTE_WS}/src/"

echo ">> Sync complete."

if [[ "${1:-}" == "build" ]]; then
  PKG="${2:-}"
  if [[ -n "${PKG}" ]]; then
    BUILD_CMD="colcon build --symlink-install --packages-select hightide_${PKG}"
  else
    BUILD_CMD="colcon build --symlink-install"
  fi
  echo ">> Building on the sub:  ${BUILD_CMD}"
  # -t gives a live terminal so build output streams back to you.
  ssh -t "${REMOTE}" "source /opt/ros/humble/setup.bash && cd '${REMOTE_WS}' && ${BUILD_CMD}"
  echo ">> Build finished. On the box, remember to:  source ${REMOTE_WS}/install/setup.bash"
fi
