#!/usr/bin/env bash
# Runs the STT and TTS benches against one router built from this checkout: speech-to-text on
# the scenarios' caller lines, text-to-speech on their agent lines. The nightly runs this.
#
#   scripts/components.sh
#   VOICEBENCH_STT_TARGETS="deepgram/flux-general-en deepgram/nova-3" scripts/components.sh
#   VOICEBENCH_COMPONENTS_POST=0 scripts/components.sh       write results without posting
#
# Results go to out/components-<time>/{stt,tts}, or under the directory given as the argument.
# Each bench posts a summary to VOICEBENCH_SLACK_CHANNEL when Slack is configured.
set -uo pipefail
cd "$(dirname "$0")/.."

stt_targets="${VOICEBENCH_STT_TARGETS:-deepgram/flux-general-en}"
tts_targets="${VOICEBENCH_TTS_TARGETS:-elevenlabs/eleven_v4_turbo inworld/inworld-tts-2-flash}"
scenario_set="${VOICEBENCH_SET:-short}"
profile="${VOICEBENCH_NETWORK_PROFILE:-local}"
out="${1:-out/components-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"

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
export STREAM_ACCELERATION_CUSTOMER_ID="${STREAM_ACCELERATION_CUSTOMER_ID:-voicebench}"

router="$(mktemp -d)/router"
(cd ../acceleration && CGO_ENABLED=1 go build -o "$router" ./cmd/router) || exit 1
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
go build -o "$voicebench" ./cmd/voicebench || exit 1

post=()
if [[ "${VOICEBENCH_COMPONENTS_POST:-1}" != "0" && -n "${VOICEBENCH_SLACK_BOT_TOKEN:-}" && -n "${VOICEBENCH_SLACK_CHANNEL:-}" ]]; then
  post=(--slack)
fi

failed=0
stt_args=()
for target in $stt_targets; do stt_args+=(--target "$target"); done
echo "== STT on the $scenario_set set's caller lines: $stt_targets"
"$voicebench" stt --scenarios "$scenario_set" "${stt_args[@]}" --network-profile "$profile" \
  --out "$out/stt" ${post[@]+"${post[@]}"} || failed=1

tts_args=()
for target in $tts_targets; do tts_args+=(--target "$target"); done
echo "== TTS on the scenarios' agent lines: $tts_targets"
"$voicebench" tts "${tts_args[@]}" --network-profile "$profile" \
  --out "$out/tts" ${post[@]+"${post[@]}"} || failed=1

echo "results in $out"
exit "$failed"
