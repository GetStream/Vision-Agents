#!/usr/bin/env bash
# local.sh <name> [agents] [hold]: builds the router, runs it on :8081 with the mocks and
# pprof, holds <agents> conversations against it and profiles the steady state into
# out/<name>. Needs the repo-root .env and the compose Postgres and Redis.
set -euo pipefail
name=$1; agents=${2:-32}; hold=${3:-90s}
cd "$(dirname "$0")/../.."
root=$PWD
out=$root/performance_benchmark/out/$name
mkdir -p "$out"
(cd acceleration && CGO_ENABLED=1 go build -o "$out/router" ./cmd/router)
(cd performance_benchmark && CGO_ENABLED=1 go build -tags webrtc -o "$out/perfbench" ./cmd/perfbench)
set -a; source .env; set +a
export ROUTER_ADDR=:8081 ROUTER_PUBLIC_URL=http://localhost:8081 ROUTER_MOCK_PROVIDERS=true \
  ROUTER_PPROF_ADDR=localhost:6060 ROUTER_LOG_LEVEL=info ROUTER_AUTH_MODE=noauth \
  ROUTER_POSTGRES_DSN="$ROUTER_POSTGRES_DSN&application_name=perfloop"
"$out/router" > "$out/router.log" 2>&1 &
router=$!
trap 'kill $router 2>/dev/null' EXIT
for _ in $(seq 60); do curl -sf localhost:8081/health >/dev/null && break; sleep 1; done
cd performance_benchmark
"$out/perfbench" run -target http://localhost:8081 -namespace "" -label "$name" -start "$agents" \
  -max "$agents" -hold "$hold" -out "out/$name/report" > "$out/bench.log" 2>&1 &
bench=$!
for _ in $(seq 120); do
  [ "$(grep -c "listening to agent audio" "$out/bench.log")" -ge "$agents" ] && break; sleep 1
done
sleep 10
pprof=localhost:6060/debug/pprof
curl -sS -o "$out/cpu.pb.gz" "$pprof/profile?seconds=30"
curl -sS -o "$out/allocs.pb.gz" "$pprof/allocs"
curl -sS -o "$out/mutex.pb.gz" "$pprof/mutex"
curl -sS -o "$out/block.pb.gz" "$pprof/block"
curl -sS -o "$out/goroutine.txt" "$pprof/goroutine?debug=1"
wait $bench
echo "profiled $agents agents into $out"
