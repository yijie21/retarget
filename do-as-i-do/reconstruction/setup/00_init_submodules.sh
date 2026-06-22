#!/bin/bash
# [00] Initialize the third-party submodules at their pinned fork-date commits.
#      (SAM3D, Fast-SAM3D, HaWoR + nested lietorch/eigen, TAPNet.)
# Weights are NOT pulled here — run 02_fetch_weights.sh for those.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export GIT_LFS_SKIP_SMUDGE=1   # don't smudge any LFS blobs; weights come from 02_fetch_weights.sh

echo "[00] Cloning + checking out submodules at pinned commits (this can take a while)..."
git submodule update --init --recursive

echo "[00] Submodule pins:"
git submodule status
echo "[00] Done. Next: ./setup/01_create_envs.sh"
