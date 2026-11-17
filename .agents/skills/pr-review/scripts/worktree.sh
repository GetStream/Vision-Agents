#!/usr/bin/env bash
# Prints the directory to review PR <number> in at its head commit <sha>: the current checkout when
# it is already that commit and clean and the PR is not from a fork, otherwise a detached worktree
# next to the main checkout. Never switches branches, stashes, or discards changes.
set -euo pipefail

if [ $# -ne 3 ] || { [ "$3" != true ] && [ "$3" != false ]; }; then
  echo "usage: worktree.sh <number> <head-sha> <is-cross-repository: true|false>" >&2
  exit 2
fi
NUM=$1
SHA=$2
FORK=$3

# A fork PR always gets the worktree, which has no .env, so its code never runs beside credentials.
if [ "$FORK" = false ] && [ "$(git rev-parse HEAD)" = "$SHA" ] && [ -z "$(git status --porcelain)" ]; then
  git rev-parse --show-toplevel
  exit 0
fi

# The main checkout, even when run from inside another worktree.
TOP=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
WT="$(dirname "$TOP")/$(basename "$TOP")-pr-$NUM"

git fetch --quiet origin "pull/$NUM/head"
if [ "$(git rev-parse FETCH_HEAD)" != "$SHA" ]; then
  echo "pull/$NUM/head is $(git rev-parse FETCH_HEAD), not $SHA: the PR moved, resolve it again" >&2
  exit 1
fi

# Check out the SHA, not FETCH_HEAD: FETCH_HEAD is per worktree, so the one fetched here does not
# resolve inside $WT.
if git worktree list --porcelain | grep -Fxq "worktree $WT"; then
  if [ -n "$(git -C "$WT" status --porcelain)" ]; then
    echo "$WT has local changes; not discarding them" >&2
    exit 1
  fi
  git -C "$WT" checkout --quiet --detach "$SHA"
else
  git worktree add --quiet --detach "$WT" "$SHA"
fi

# .env is gitignored, so the worktree has none, and nothing searching parent directories finds the
# main checkout's. A fork PR's code is untrusted, so its worktree gets none.
if [ "$FORK" = false ] && [ -e "$TOP/.env" ]; then
  ln -sfn "$TOP/.env" "$WT/.env"
elif [ "$FORK" = true ] && [ -L "$WT/.env" ]; then
  rm "$WT/.env"
fi

echo "$WT"
