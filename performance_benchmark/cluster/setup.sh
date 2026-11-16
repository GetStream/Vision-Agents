#!/usr/bin/env bash
# setup.sh: builds perfbench inside the perfbench pod (perfbench.yaml) and gives it the
# kubectl context `-target benchmark` reads the router's pods with. Rerun after a change to
# performance_benchmark/, benchmark/ or sdks/go.
set -euo pipefail
cd "$(dirname "$0")/../.."
src=$(mktemp -d)/src.tgz
COPYFILE_DISABLE=1 tar czf "$src" --exclude='out' --exclude='*.wav' performance_benchmark sdks/go
kubectl -n perfbench cp "$src" perfbench:/src.tgz
kubectl -n perfbench exec perfbench -- bash -c '
set -e
curl -sSL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/v1.35.0/bin/linux/$(dpkg --print-architecture)/kubectl" && chmod +x /usr/local/bin/kubectl
rm -rf /w && mkdir -p /w && cd /w && tar xzf /src.tgz
cd performance_benchmark && GOWORK=off go build -tags webrtc -o /usr/local/bin/perfbench ./cmd/perfbench
SA=/var/run/secrets/kubernetes.io/serviceaccount; C=gke_stream-gcp-2026-v2_us-east1_us-east1-acceleration
kubectl config set-cluster in --server=https://kubernetes.default.svc --certificate-authority=$SA/ca.crt >/dev/null
kubectl config set-credentials sa --token=$(cat $SA/token) >/dev/null
kubectl config set-context $C --cluster=in --user=sa >/dev/null && kubectl config use-context $C >/dev/null
kubectl -n accelerate-benchmark top pod --no-headers | head -1; nproc'
