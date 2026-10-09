#!/usr/bin/env bash
# Runs the frozen set (or VOICEBENCH_SET=short) against our stack and LiveKit, then posts the
# digest to Slack.
#
#   scripts/digest.sh                                   nightly: k=1, LiveKit Inference
#   VOICEBENCH_K=3 VOICEBENCH_LIVEKIT_ARMS="inference realtime" \
#     VOICEBENCH_DIGEST_TITLE="Voicebench weekly" scripts/digest.sh
#
# Our stack runs on a router built from this checkout, unless STREAM_ACCELERATION_URL names
# a hosted one. Packs run side by side (VOICEBENCH_PARALLEL=0 to take turns), each stopped
# after VOICEBENCH_PACK_TIMEOUT seconds, 1500 for each of the k trials by default. The digest goes to VOICEBENCH_SLACK_CHANNEL as the bot behind
# VOICEBENCH_SLACK_BOT_TOKEN, both read from benchmark/.env like the provider keys;
# VOICEBENCH_DIGEST_POST=0 writes it to the run directory without posting. Results go to
# out/digest-<time>, or to the directory given as the argument.
set -uo pipefail
cd "$(dirname "$0")/.."

k="${VOICEBENCH_K:-1}"
# frozen is the trend-line set; short is the quicker subset in scenarios/short.txt.
scenario_set="${VOICEBENCH_SET:-frozen}"
packs="${VOICEBENCH_PACKS:-restaurant healthcare telecom}"
# An empty VOICEBENCH_LIVEKIT_ARMS runs our stack alone.
arms="${VOICEBENCH_LIVEKIT_ARMS-inference}"
# The bench's own LiveKit worker registers under a name of its own, so a deployed agent
# called "voicebench" cannot be dispatched the bench's calls.
livekit_agent="${VOICEBENCH_LIVEKIT_AGENT:-voicebench-local}"
title="${VOICEBENCH_DIGEST_TITLE:-Voicebench nightly}"
profile="${VOICEBENCH_NETWORK_PROFILE:-local}"
out="${1:-out/digest-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"

# Packs run side by side against one router, each with its own agent and world server, so
# the set takes about as long as its slowest pack. A run past its time limit is killed with
# everything it started: a call that deadlocks ignores its own deadline and Ctrl-C.
parallel="${VOICEBENCH_PARALLEL:-1}"
# 25 minutes a trial: a frozen pack takes about 12 at k=1, so only a stuck call reaches it.
limit="${VOICEBENCH_PACK_TIMEOUT:-$((1500 * k))}"
# 8000 is often taken on a laptop by a local stream-api.
agent_port="${VOICEBENCH_AGENT_PORT_BASE:-8001}"

export CGO_ENABLED=1
voicebench="$(mktemp -d)/voicebench"
go build -tags webrtc -o "$voicebench" ./cmd/voicebench || exit 1

ours=(--target accelerated)
spawned=0
if [[ -n "${STREAM_ACCELERATION_URL:-}" ]]; then
  export STREAM_ACCELERATION_AUTHENTICATE="${STREAM_ACCELERATION_AUTHENTICATE:-1}"
  if [[ -n "${VOICEBENCH_TARGET_URL:-}" ]]; then
    ours+=(--target-url "$VOICEBENCH_TARGET_URL")
  fi
  # One agent someone else runs serves every pack, so they take turns.
  parallel=0
else
  export ROUTER_AUTH_MODE="${ROUTER_AUTH_MODE:-noauth}"
  export ROUTER_RATE_LIMIT_MESSAGES_PER_DAY="${ROUTER_RATE_LIMIT_MESSAGES_PER_DAY:-0}"
  export ROUTER_RATE_LIMIT_TOKENS_PER_DAY="${ROUTER_RATE_LIMIT_TOKENS_PER_DAY:-0}"
  export ROUTER_ADDR=127.0.0.1:8080
  export STREAM_ACCELERATION_URL="http://$ROUTER_ADDR"
  router="$(dirname "$voicebench")/router"
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
  ours+=(--spawn)
  spawned=1
fi

# kill_tree stops a process and everything under it, children first.
kill_tree() {
  local child
  for child in $(pgrep -P "$1" 2>/dev/null); do
    kill_tree "$child" "$2"
  done
  kill "-$2" "$1" 2>/dev/null
}

# run_logged runs one voicebench run with its whole output in the named log, showing only its
# per-call progress lines as they happen, and kills it if it runs past the limit.
run_logged() {
  local name="$1" log="$2"
  shift 2
  "$@" > "$log" 2>&1 &
  local pid=$!
  (tail -n +1 -f "$log" 2>/dev/null | grep --line-buffered '^voicebench: \[') &
  local follower=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null && (( waited < limit )); do
    sleep 5
    waited=$((waited + 5))
  done
  local status=0
  if kill -0 "$pid" 2>/dev/null; then
    echo "   $name ran past ${limit}s and was stopped; the call it was on never finished"
    kill_tree "$pid" TERM
    sleep 10
    kill_tree "$pid" KILL
    status=124
  fi
  wait "$pid" 2>/dev/null || status=$((status == 0 ? $? : status))
  sleep 1
  kill_tree "$follower" TERM
  if [[ "$status" != 0 ]]; then
    echo "   $name failed, see $log"
  fi
}

# full_log puts a run's whole log in a collapsed group on GitHub Actions.
full_log() {
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::$1: full log"
    cat "$2"
    echo "::endgroup::"
  fi
}

pids=()
i=0
for pack in $packs; do
  echo "== $pack: accelerated"
  ports=()
  if [[ "$spawned" == 1 ]]; then
    ports=(--target-url "http://127.0.0.1:$((agent_port + i))" --world-addr "127.0.0.1:$((8090 + i))")
  fi
  if [[ "$parallel" == 1 ]]; then
    run_logged "$pack: accelerated" "$out/$pack-accelerated.log" \
      "$voicebench" run --pack "$pack" "--$scenario_set" --k "$k" "${ours[@]}" ${ports[@]+"${ports[@]}"} \
      --network-profile "$profile" --out "$out/$pack-accelerated" &
    pids+=("$!")
    # Each pack's agent syncs its config to the router as it starts; a few seconds apart they
    # do not collide.
    sleep 3
  else
    run_logged "$pack: accelerated" "$out/$pack-accelerated.log" \
      "$voicebench" run --pack "$pack" "--$scenario_set" --k "$k" "${ours[@]}" ${ports[@]+"${ports[@]}"} \
      --network-profile "$profile" --out "$out/$pack-accelerated"
  fi
  i=$((i + 1))
done
for pid in ${pids[@]+"${pids[@]}"}; do
  wait "$pid"
done
for pack in $packs; do
  full_log "$pack: accelerated" "$out/$pack-accelerated.log"
done

# LiveKit's arms take turns: each spawns a worker of its own.
for pack in $packs; do
  for arm in $arms; do
    echo "== $pack: livekit-$arm"
    VOICEBENCH_LIVEKIT_PIPELINE="$arm" run_logged "$pack: livekit-$arm" "$out/$pack-livekit-$arm.log" \
      "$voicebench" run --pack "$pack" "--$scenario_set" --k "$k" \
      --target livekit --spawn --livekit-agent "$livekit_agent" --system "livekit-$arm" --network-profile "$profile" \
      --out "$out/$pack-livekit-$arm"
    full_log "$pack: livekit-$arm" "$out/$pack-livekit-$arm.log"
  done
done

runs=()
for summary in "$out"/*/summary.json; do
  [[ -e "$summary" ]] && runs+=("$(dirname "$summary")")
done
if [[ ${#runs[@]} -eq 0 ]]; then
  echo "no run finished, nothing to post"
  exit 1
fi

post=(--slack)
if [[ "${VOICEBENCH_DIGEST_POST:-1}" == "0" ]]; then
  post=()
fi
"$voicebench" digest --title "$title" --out "$out" ${post[@]+"${post[@]}"} "${runs[@]}"
