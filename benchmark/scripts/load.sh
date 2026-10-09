#!/usr/bin/env bash
# Runs one pack's scenario set at several concurrency levels against one router built from
# this checkout, and digests the levels side by side: how reply time holds up when calls
# share the router and the Gemma deployment.
#
#   scripts/load.sh                                              restaurant, short set, 1 and 3 at once
#   VOICEBENCH_CONCURRENCY="1 3 5" VOICEBENCH_PACK=healthcare scripts/load.sh
#   VOICEBENCH_LOAD_POST=1 scripts/load.sh                       also post the digest to Slack
#
# A level of N runs N copies of the set at once, each its own voicebench process with its own
# agent and world server, all on the one router. Results go to out/load-<time>/xN-<copy>,
# or under the directory given as the argument, and the digest names each level "accelerated ×N".
set -uo pipefail
cd "$(dirname "$0")/.."

levels="${VOICEBENCH_CONCURRENCY:-1 3}"
pack="${VOICEBENCH_PACK:-restaurant}"
scenario_set="${VOICEBENCH_SET:-short}"
k="${VOICEBENCH_K:-1}"
profile="${VOICEBENCH_NETWORK_PROFILE:-local}"
# 8000 is often taken on a laptop by a local stream-api.
agent_port="${VOICEBENCH_AGENT_PORT_BASE:-8001}"
out="${1:-out/load-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"

# The router reads its settings from the environment, so load the file voicebench reads.
# Like voicebench, a variable already set wins over the file.
for env in .env ../.env; do
  if [[ -f "$env" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]] || continue
      [[ -n "${!BASH_REMATCH[1]+set}" ]] && continue
      eval "export $line"
    done < "$env"
    break
  fi
done
export ROUTER_AUTH_MODE="${ROUTER_AUTH_MODE:-noauth}"
export ROUTER_RATE_LIMIT_MESSAGES_PER_DAY="${ROUTER_RATE_LIMIT_MESSAGES_PER_DAY:-0}"
export ROUTER_RATE_LIMIT_TOKENS_PER_DAY="${ROUTER_RATE_LIMIT_TOKENS_PER_DAY:-0}"
export ROUTER_ADDR=127.0.0.1:8080
export STREAM_ACCELERATION_URL="http://$ROUTER_ADDR"
export CGO_ENABLED=1

router="$(mktemp -d)/router"
(cd ../acceleration && go build -o "$router" ./cmd/router) || exit 1
"$router" > "$out/router.log" 2>&1 &
router_pid=$!
trap 'kill "$router_pid" 2>/dev/null; wait "$router_pid" 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  curl -fsS "$STREAM_ACCELERATION_URL/health" > /dev/null 2>&1 && break
  if ! kill -0 "$router_pid" 2>/dev/null; then
    echo "router exited, see $out/router.log"
    exit 1
  fi
  sleep 1
done

voicebench="$(dirname "$router")/voicebench"
go build -tags webrtc -o "$voicebench" ./cmd/voicebench || exit 1

# Copies running at once would otherwise synthesize the same caller lines into the same cache
# files at the same moment.
"$voicebench" synth --pack "$pack" > "$out/synth.log" 2>&1 || { echo "caller audio failed, see $out/synth.log"; exit 1; }

runs=()
for level in $levels; do
  echo "== $pack, $scenario_set set, $level at once"
  pids=()
  for copy in $(seq 1 "$level"); do
    dir="$out/x$level-$copy"
    "$voicebench" run --pack "$pack" "--$scenario_set" --k "$k" --target accelerated --spawn \
      --target-url "http://127.0.0.1:$((agent_port + copy - 1))" --world-addr "127.0.0.1:$((8090 + copy - 1))" \
      --system "accelerated ×$level" --network-profile "$profile" --out "$dir" > "$dir.log" 2>&1 &
    pids+=("$!")
    runs+=("$dir")
    # Each copy's agent syncs the pack's config to the router as it starts; a few seconds
    # apart they do not collide, and the calls still overlap for the whole set.
    sleep 3
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || echo "   a copy failed, see $out/x$level-*.log"
  done
done

finished=()
for dir in "${runs[@]}"; do
  [[ -f "$dir/summary.json" ]] && finished+=("$dir")
done
if [[ ${#finished[@]} -eq 0 ]]; then
  echo "no run finished, nothing to digest"
  exit 1
fi
post=()
if [[ "${VOICEBENCH_LOAD_POST:-0}" == "1" ]]; then
  post=(--slack)
fi
"$voicebench" digest --title "Voicebench load · $pack" --out "$out" ${post[@]+"${post[@]}"} "${finished[@]}"
echo "results in $out"
