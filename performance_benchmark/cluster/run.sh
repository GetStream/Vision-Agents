#!/usr/bin/env bash
# run.sh <name> [start] [max] [hold]: ramps from the perfbench pod against accelerate-benchmark
# and copies the report to out/<name>. Ends the calls of draining router generations first,
# so the ramp measures only the router just deployed.
set -euo pipefail
name=$1; start=${2:-4}; max=${3:-1024}; hold=${4:-60s}
cd "$(dirname "$0")/.."
ns=accelerate-benchmark
for pod in $(kubectl -n $ns get pods -o name | grep -v -e gateway -e rollup); do
  for pid in $(kubectl -n $ns exec "${pod#pod/}" -c router -- /devrun/devrun __devrun status 2>/dev/null |
    jq -r '.generations[] | select(.draining_since != null) | .pid'); do
    kubectl -n $ns exec "${pod#pod/}" -c router -- kill -INT "$pid"
  done
done
sleep 60
key=$(grep '^STREAM_API_KEY=' ../.env | cut -d= -f2- | tr -d "'\"")
secret=$(grep '^STREAM_API_SECRET=' ../.env | cut -d= -f2- | tr -d "'\"")
kubectl -n perfbench exec perfbench -- env STREAM_API_KEY="$key" STREAM_API_SECRET="$secret" \
  bash -c "cd /w/performance_benchmark && rm -rf out/$name && perfbench run -target benchmark -label $name -start $start -factor 2 -max $max -hold $hold -join-rate 20 -out out/$name"
rm -rf "out/$name"
kubectl -n perfbench cp "perfbench:/w/performance_benchmark/out/$name" "out/$name" >/dev/null
