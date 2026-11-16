#!/usr/bin/env bash
# Prints the generated files the PR in <dir> leaves out of step with their source. Drift that is
# already at the merge base with <base-ref> is reported as inherited, not blamed on the PR. Exits 1
# when the PR introduces drift. Runs the code in <dir>: never on a fork PR without asking.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: generated.sh <dir> <base-ref>" >&2
  exit 2
fi
DIR=$(cd "$1" && pwd)
BASE=$2

# Prints each generated file that is out of step with its source in checkout $1.
stale() {
  local co=$1 out
  # cmd/openapi writes api/openapi.yaml in place: compare, then put the committed file back.
  (cd "$co/acceleration" && go run ./cmd/openapi)
  if ! git -C "$co" diff --quiet -- acceleration/api/openapi.yaml; then
    echo acceleration/api/openapi.yaml
  fi
  git -C "$co" restore -- acceleration/api/openapi.yaml
  # The check runs openapi-typescript from node_modules. --ignore-scripts installs it without
  # running any package's install scripts.
  if [ ! -d "$co/sdks/js/node_modules" ]; then
    (cd "$co/sdks/js" && npm ci --ignore-scripts --no-audit --no-fund >/dev/null)
  fi
  if out=$(cd "$co/sdks/js" && npm run --silent types -- --check 2>&1); then
    :
  elif grep -q "out of step" <<<"$out"; then
    echo sdks/js/src/generated/api.ts
  else
    echo "$out" >&2
    return 1
  fi
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

stale "$DIR" >"$WORK/head"
if [ ! -s "$WORK/head" ]; then
  echo "in step: every generated file matches its source"
  exit 0
fi

git -C "$DIR" fetch --quiet origin "$BASE"
MB=$(git -C "$DIR" merge-base "origin/$BASE" HEAD)
trap 'git -C "$DIR" worktree remove --force "$WORK/base" 2>/dev/null || true; rm -rf "$WORK"' EXIT
git -C "$DIR" worktree add --quiet --detach "$WORK/base" "$MB"
stale "$WORK/base" >"$WORK/base.txt"

INTRODUCED=$(grep -vxFf "$WORK/base.txt" "$WORK/head" || true)
INHERITED=$(grep -xFf "$WORK/base.txt" "$WORK/head" || true)
for f in $INHERITED; do echo "inherited from $BASE at ${MB:0:7}: $f"; done
for f in $INTRODUCED; do echo "introduced by the PR: $f"; done
[ -z "$INTRODUCED" ]
