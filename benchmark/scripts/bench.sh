#!/usr/bin/env bash
# Runs the benchmarks against this checkout, one stage after another, and writes SUMMARY.md.
#
#   scripts/bench.sh short           the basics, about 10 minutes
#   scripts/bench.sh full            every benchmark, about 45 minutes
#   BENCH_STAGES="agents-frozen" scripts/bench.sh full      only the stages named
#
# short:
#   agents-basics     Voicebench basics set: one call for each basic behavior, four a pack
# full:
#   components        STT and TTS through the Go SDK on every scenario's lines
#   agents-frozen     Voicebench frozen set, every pack, with the STT pass on its caller lines
#   agents-extended   Voicebench extended set, every pack
#
# Every stage builds its own router from this checkout on :8080 through scripts/digest.sh or
# components.sh. Nothing is posted to Slack. Results go to out/bench-<mode>-<time>, or the
# directory given.
set -uo pipefail
cd "$(dirname "$0")/.."
root=$(cd .. && pwd)

mode="${1:-}"
case "$mode" in
  short) default_stages="agents-basics" ;;
  full) default_stages="components agents-frozen agents-extended" ;;
  *) echo "usage: scripts/bench.sh short|full [out-dir]"; exit 2 ;;
esac
stages="${BENCH_STAGES:-$default_stages}"
out="${2:-out/bench-$mode-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"
out=$(cd "$out" && pwd)

# The router digest.sh builds reads its settings from the environment, so export the file
# voicebench reads. Like voicebench, a variable already set wins over the file, and
# benchmark/.env hides the root one. STREAM_ACCELERATION_URL from the file is dropped: digest.sh
# would use that router instead of building one from the checkout. Set it in the environment to
# bench a hosted router on purpose.
preset_router="${STREAM_ACCELERATION_URL-}"
for env in .env ../.env; do
  if [[ -f "$env" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]] || continue
      [[ -n "${!BASH_REMATCH[1]+set}" ]] && continue
      eval "export $line"
    done < "$env"
    echo "env from benchmark/$env"
    break
  fi
done
if [[ -z "$preset_router" ]]; then
  unset STREAM_ACCELERATION_URL STREAM_ACCELERATION_AUTHENTICATE
fi

export CGO_ENABLED=1
export VOICEBENCH_DIGEST_POST=0 VOICEBENCH_COMPONENTS_POST=0
export VOICEBENCH_LIVEKIT_ARMS="${VOICEBENCH_LIVEKIT_ARMS-}"

branch=$(git -C "$root" rev-parse --abbrev-ref HEAD)
commit=$(git -C "$root" rev-parse --short HEAD)
dirty=$(git -C "$root" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
echo "bench $mode on $branch@$commit ($dirty modified files): $stages"
[[ "$branch" == accelerate ]] || echo "warning: this checkout is on $branch, not accelerate"

failed=0
summary_rows=()
record() {
  summary_rows+=("| $1 | $2 | $3 |")
  [[ "$2" == ok ]] || failed=1
}

preflight() {
  local missing=() k
  for k in STREAM_API_KEY STREAM_API_SECRET ELEVENLABS_API_KEY DEEPGRAM_API_KEY OPENAI_API_KEY \
    GEMMA_BASE_URL BASETEN_API_KEY INWORLD_API_KEY; do
    [[ -n "${!k:-}" ]] || missing+=("$k")
  done
  if (( ${#missing[@]} )); then
    echo "missing from .env: ${missing[*]} (fetch with: rocky agents local secrets create_env -f <file>)"
    return 1
  fi

  local busy ports=() p
  ports+=("${VOICEBENCH_ROUTER_PORT:-8080}")
  for p in 0 1 2; do
    ports+=("$(( ${VOICEBENCH_AGENT_PORT_BASE:-8001} + p ))" "$(( ${VOICEBENCH_WORLD_PORT_BASE:-8090} + p ))")
  done
  busy=$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | grep -E ":($(IFS='|'; echo "${ports[*]}")) " || true)
  if [[ -n "$busy" ]]; then
    echo "ports a stage needs are taken, stop these first:"
    echo "$busy"
    return 1
  fi

  (cd "$root" && docker compose up -d --wait postgres redis) > "$out/compose.log" 2>&1 \
    || { echo "postgres and redis did not start, see $out/compose.log"; return 1; }

  # BENCH_WAKE_URL and BENCH_WAKE_MODEL name another self-hosted model to wake instead.
  local t wake_url="${BENCH_WAKE_URL:-$GEMMA_BASE_URL}" wake_model="${BENCH_WAKE_MODEL:-google/gemma-4-26B-A4B-it}"
  for _ in $(seq 1 120); do
    t=$(curl -sS -m 60 -o /dev/null -w '%{http_code} %{time_total}' "$wake_url/chat/completions" \
      -H "Authorization: Bearer ${GEMMA_API_KEY:-$BASETEN_API_KEY}" -H 'Content-Type: application/json' \
      -d '{"model":"'"$wake_model"'","max_tokens":1,"messages":[{"role":"user","content":"Hi"}]}')
    echo "$wake_model $t"
    case "$t" in 200\ 0.*|200\ 1.*) return 0 ;; esac
    sleep 15
  done
  echo "$wake_model never answered within two seconds"
  return 1
}

# tally prints passed/total and errored calls across a stage's packs, and fails when a call
# errored: digest.sh exits 0 even when every call was infra.
tally() {
  python3 - "$1" <<'EOF'
import glob, json, sys
calls = []
for path in sorted(glob.glob(f"{sys.argv[1]}/*-accelerated/summary.json")):
    calls += json.load(open(path)).get("calls") or []
passed = sum(1 for c in calls if c.get("passed"))
errored = [c for c in calls if c.get("error")]
line = f"{passed}/{len(calls)} passed"
if errored:
    line += f", {len(errored)} errored (infra): {errored[0]['error'][:80]}"
print(line)
sys.exit(1 if errored or not calls else 0)
EOF
}

if ! preflight; then
  record preflight failed "setup, see above"
else
  record preflight ok "keys, ports, postgres, redis, ${BENCH_WAKE_MODEL:-gemma} awake"
  for stage in $stages; do
    echo "== $stage"
    started=$(date +%s)
    status=0
    detail=""
    case "$stage" in
      components)
        VOICEBENCH_SET=all scripts/components.sh "$out/$stage" || status=$?
        detail="$stage/stt/report.md, $stage/tts/report.md" ;;
      agents-basics|agents-frozen|agents-extended)
        VOICEBENCH_SET="${stage#agents-}" scripts/digest.sh "$out/$stage" || status=$?
        counts=$(tally "$out/$stage") || status=${status/#0/1}
        detail="$counts; $stage/voicebench.html" ;;
      *)
        echo "unknown stage $stage"
        status=2 ;;
    esac
    took="$(( ($(date +%s) - started) / 60 ))m"
    if [[ "$status" == 0 ]]; then
      record "$stage" ok "$took; $detail"
    else
      record "$stage" "failed ($status)" "$took; $detail"
    fi
  done
fi

{
  echo "# Bench $mode: $branch@$commit"
  echo
  echo "$(date -u +%Y-%m-%dT%H:%MZ), $dirty modified tracked files, network profile ${VOICEBENCH_NETWORK_PROFILE:-local}"
  echo
  echo "| Stage | Status | Time and results |"
  echo "|---|---|---|"
  printf '%s\n' "${summary_rows[@]}"
} > "$out/SUMMARY.md"
cat "$out/SUMMARY.md"
echo "results in $out"
exit "$failed"
