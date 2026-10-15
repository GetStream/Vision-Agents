#!/usr/bin/env bash
# profile.sh <name> <agents> [wait]: run beside `cluster/run.sh <name> ... | tee out/<name>.log`.
# Once the ramp reaches <agents>, waits [wait] seconds for the joins to settle (20 at 512),
# then profiles one benchmark router pod for 30 s into out/<name>-prof.
set -euo pipefail
name=$1; agents=$2; wait=${3:-20}
cd "$(dirname "$0")/.."
ns=accelerate-benchmark
out=out/$name-prof
mkdir -p "$out"
until grep -q "ramping.*agents=$agents " "out/$name.log" 2>/dev/null; do sleep 5; done
sleep "$wait"
pod=$(kubectl -n $ns get pods -o name | grep -v -e gateway -e rollup | head -1)
kubectl -n $ns port-forward "$pod" 16060:6060 >/dev/null 2>&1 &
pf=$!
trap 'kill $pf 2>/dev/null' EXIT
sleep 3
pprof=localhost:16060/debug/pprof
curl -sS -o "$out/cpu.pb.gz" "$pprof/profile?seconds=30"
curl -sS -o "$out/goroutine.txt" "$pprof/goroutine?debug=1"
curl -sS -o "$out/mutex.pb.gz" "$pprof/mutex"
curl -sS -o "$out/allocs.pb.gz" "$pprof/allocs"
echo "profiled $pod into $out"
