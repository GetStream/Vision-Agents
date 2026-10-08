#!/usr/bin/env bash
# Runs every pack against one router built from this checkout, for quick iteration.
#
#   scripts/packs.sh                                        short set, k=1: about 8 minutes
#   VOICEBENCH_K=3 scripts/packs.sh                         short set, k=3: about 22 minutes
#   VOICEBENCH_PARALLEL=0 scripts/packs.sh                  one pack after another: about 21 minutes
#   VOICEBENCH_SET=frozen VOICEBENCH_K=3 scripts/packs.sh   the frozen set
#
# Each pack gets its own agent and world server port and shares the router, so with the packs
# at once three calls run side by side. Results go to out/packs-<time>/<pack>, or to the
# directory given as the argument.
set -uo pipefail
cd "$(dirname "$0")/.."

k="${VOICEBENCH_K:-1}"
set_flag="--${VOICEBENCH_SET:-short}"
packs="${VOICEBENCH_PACKS:-restaurant healthcare telecom}"
parallel="${VOICEBENCH_PARALLEL:-1}"
profile="${VOICEBENCH_NETWORK_PROFILE:-local}"
out="${1:-out/packs-$(date -u +%Y%m%dT%H%M%SZ)}"
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

# Build voicebench once so the packs do not compile it three times over.
voicebench="$(dirname "$router")/voicebench"
go build -tags webrtc -o "$voicebench" ./cmd/voicebench || exit 1

failed=0
pids=()
i=0
for pack in $packs; do
  echo "== $pack"
  "$voicebench" run --pack "$pack" "$set_flag" --k "$k" --target accelerated --spawn \
    --target-url "http://127.0.0.1:$((8000 + i))" --world-addr "127.0.0.1:$((8090 + i))" \
    --network-profile "$profile" --out "$out/$pack" > "$out/$pack.log" 2>&1 &
  if [[ "$parallel" == "0" ]]; then
    wait "$!" || { echo "$pack failed, see $out/$pack.log"; failed=1; }
  else
    pids+=("$!")
  fi
  i=$((i + 1))
done

i=0
for pid in ${pids[@]+"${pids[@]}"}; do
  pack="$(echo $packs | cut -d' ' -f$((i + 1)))"
  wait "$pid" || { echo "$pack failed, see $out/$pack.log"; failed=1; }
  i=$((i + 1))
done
echo "results in $out"
exit "$failed"
