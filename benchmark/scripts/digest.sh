#!/usr/bin/env bash
# Runs the frozen set (or VOICEBENCH_SET=short) against our stack and LiveKit, then posts the
# digest to Slack.
#
#   scripts/digest.sh                                   nightly: k=1, LiveKit Inference
#   VOICEBENCH_K=3 VOICEBENCH_LIVEKIT_ARMS="inference realtime" \
#     VOICEBENCH_DIGEST_TITLE="Voicebench weekly" scripts/digest.sh
#
# Our stack runs on a router built from this checkout, unless STREAM_ACCELERATION_URL names
# a hosted one. The digest goes to VOICEBENCH_SLACK_CHANNEL as the bot behind
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

ours=(--target accelerated)
if [[ -n "${STREAM_ACCELERATION_URL:-}" ]]; then
  export STREAM_ACCELERATION_AUTHENTICATE="${STREAM_ACCELERATION_AUTHENTICATE:-1}"
else
  router="$(mktemp -d)/router"
  (cd ../acceleration && go build -o "$router" ./cmd/router) || exit 1
  ours+=(--spawn --bin "$router")
  export ROUTER_AUTH_MODE="${ROUTER_AUTH_MODE:-noauth}"
  export ROUTER_RATE_LIMIT_MESSAGES_PER_DAY="${ROUTER_RATE_LIMIT_MESSAGES_PER_DAY:-0}"
  export ROUTER_RATE_LIMIT_TOKENS_PER_DAY="${ROUTER_RATE_LIMIT_TOKENS_PER_DAY:-0}"
fi
if [[ -n "${VOICEBENCH_TARGET_URL:-}" ]]; then
  ours+=(--target-url "$VOICEBENCH_TARGET_URL")
fi

# run_logged runs one voicebench run with its whole output in the named log, showing only
# its per-call progress lines as they happen. On GitHub Actions the whole log follows in a
# collapsed group, so it can be read without downloading the artifact.
run_logged() {
  local name="$1" log="$2"
  shift 2
  "$@" 2>&1 | tee "$log" | grep --line-buffered '^voicebench: \['
  local status="${PIPESTATUS[0]}"
  if [[ "$status" != 0 ]]; then
    echo "   failed, see $log"
  fi
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::group::$name: full log"
    cat "$log"
    echo "::endgroup::"
  fi
}

export CGO_ENABLED=1
for pack in $packs; do
  echo "== $pack: accelerated"
  run_logged "$pack: accelerated" "$out/$pack-accelerated.log" \
    go run -tags webrtc ./cmd/voicebench run --pack "$pack" "--$scenario_set" --k "$k" "${ours[@]}" \
    --network-profile "$profile" --out "$out/$pack-accelerated"
  for arm in $arms; do
    echo "== $pack: livekit-$arm"
    VOICEBENCH_LIVEKIT_PIPELINE="$arm" run_logged "$pack: livekit-$arm" "$out/$pack-livekit-$arm.log" \
      go run -tags webrtc ./cmd/voicebench run --pack "$pack" "--$scenario_set" --k "$k" \
      --target livekit --spawn --livekit-agent "$livekit_agent" --system "livekit-$arm" --network-profile "$profile" \
      --out "$out/$pack-livekit-$arm"
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
go run ./cmd/voicebench digest --title "$title" --out "$out" ${post[@]+"${post[@]}"} "${runs[@]}"
