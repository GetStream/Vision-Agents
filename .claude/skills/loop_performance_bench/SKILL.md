---
name: loop_performance_bench
description: Run the router performance loop - profile locally, fix the top cost, cut needless API, SQL and Redis calls, deploy to the benchmark cluster, ramp, record agents held and $/voice-minute, repeat ten times. Use when asked to run the performance loop, to improve how many agents one router holds, or to lower the hosting cost per voice minute.
---

# The performance loop

Ten rounds of: find what the router spends its CPU and round trips on, fix the biggest one,
prove it on the benchmark cluster, and write down what moved. Read the
`performance_benchmark` skill first: this one assumes its setup, its break rules and its
report. `performance_benchmark/findings.md` holds what earlier loops found and fixed; read
it so you do not chase a cost that is already gone.

What each round reports: **agents held** on the fixed two-node benchmark router, **reply
p50/p95** at that step, **router mCPU per agent**, and **hosting $ per voice minute**:

```
$/voice-min = 2 nodes × $0.154/h ÷ agents held ÷ 60
```

Only Go hosting counts: the mocks bill nothing, and model cost is not the router's.

## 0. Once, before round 1

- Work on a branch off `accelerate` (`thierry/router-perf-loop` style). One commit per fix,
  `perf(<scope>): ...`.
- The benchmark router must stay at **2 pods on 2 n4a-standard-4 nodes** (HPA 2/2 in
  `shards/us-east1-acceleration/accelerate-benchmark.yaml` in the private infra repo, with
  `router.mockProviders: true` and `router.pprofAddr`). Never let it autoscale past two
  nodes: the cost per minute is only comparable on fixed hardware.
- Bring up the callers: `kubectl apply -f performance_benchmark/cluster/perfbench.yaml`, then
  `performance_benchmark/cluster/setup.sh`.
- Round 0 is the baseline: deploy the branch as it is and ramp (step 4) before changing
  anything.

## 1. Profile locally

```bash
performance_benchmark/loop/local.sh round-3 32 90s
```

It builds the router, runs it on `:8081` with the mocks and pprof against the compose
Postgres and Redis, holds 32 conversations, and writes CPU, allocs, mutex, block and
goroutine profiles to `performance_benchmark/out/round-3/`. Then:

```bash
go tool pprof -top -nodecount=40 performance_benchmark/out/round-3/cpu.pb.gz
go tool pprof -top -sample_index=alloc_space performance_benchmark/out/round-3/allocs.pb.gz
```

Bucket the CPU before choosing: Opus encode and decode, SRTP and UDP syscalls, GC, the
scheduler, HTTP and websocket, TLS, the SDK's stats, the voice logic, Postgres and Redis.
GC high means allocations: fix the top `alloc_space` entry, not GC. Count goroutines per
agent in `goroutine.txt`: a few hundred extra timers or loops per agent is a cost too.

Prefer the cluster profile from the last round (step 4) when there is one: 32 agents on a
laptop hides what only shows at 512.

## 2. Review the calls each request makes

Every round, also look at the round trips, not only the CPU. A query or Redis command
saved runs on every turn of every agent, and Postgres and Redis are shared by every pod.

Count them on the local router. `local.sh` names its Postgres connections `perfloop`, so
turn on statement logging, watch Redis, and run it with one agent:

```bash
docker exec vision-agents-postgres-1 psql -U postgres -c "ALTER SYSTEM SET log_statement='all'" \
  -c "ALTER SYSTEM SET log_line_prefix='%m [%a] %p '" -c "SELECT pg_reload_conf()"
docker exec vision-agents-redis-1 redis-cli monitor > /tmp/redis-monitor.log &
performance_benchmark/loop/local.sh calls 1 60s
docker logs --since 2m vision-agents-postgres-1 2>&1 | grep '\[perfloop\].*statement:'
```

Take an idle minute first so background work (health pings, asynq, sweeps) can be
subtracted, and split the log by the request that made each statement. Postgres prints the
backend pid after the application name, which separates a sweep's transaction from a
request's. `MONITOR` shows each client's address: the native router connects through the
Docker gateway, so filter on that address to drop a dockerised router's traffic. It also
lists pipelined commands one by one, so read the code to tell a pipeline from separate round
trips. Put both settings back afterwards
(`ALTER SYSTEM RESET log_statement; ALTER SYSTEM RESET log_line_prefix; SELECT pg_reload_conf()`)
and stop the monitor.

Count at least: `createSession`, `createResponse` (a whole turn), `startSessionVoice`, and
one voice turn. Then, for each, ask:

- **Repeated reads.** The same row read twice in one request, or on every turn when it
  changes once a session: read it once and pass it down, or cache it in the session.
- **N+1.** A query per item in a loop: one query with `= ANY($1)` or a join. A batch split
  into chunks takes its chunk size from evidence: the engine's limit (PostgreSQL takes at
  most 65,535 bind parameters in one statement: the Bind message counts them in an Int16,
  https://www.postgresql.org/docs/current/protocol-message-formats.html) and the input range
  the request validation allows. Not from gut feel or another constant. State the reasoning
  in the constant's doc comment and in the PR.
- **Transactions that need not be.** A single insert wrapped in its own transaction, or an
  advisory lock taken per row: batch the rows, or take the lock once.
- **Writes on the hot path.** A log or usage row written synchronously on every turn: buffer
  it and flush in batches off the turn.
- **Redis round trips.** Several commands one after another on the same request: one
  `Pipeline()`/`TxPipelined()`, or a Lua script when a later command depends on an earlier
  result. `HGETALL` per item in a loop: pipeline them or keep one hash.
- **Polling.** A `PING` or health read per request or per second per agent: share one
  checker per process.
- **Outbound API calls.** The same call to Stream or a provider on every turn: reuse the
  connection (TLS session cache, keep-alive) and cache what does not change during a call.

Record the counts before and after in the round's findings. A test that the request still
writes and returns the same thing is the evidence it still works; the count is the evidence
it mattered.

## 3. Fix one thing

The largest cost from step 1 or step 2, one per round. Prove it locally: a
`go test -bench` on the hot function (`-benchmem`, `-count 6`) for CPU and allocations, and
the package's tests, including its integration tests where they exist (`go-testing`
skill). Fixes in the WebRTC SDK go to `getstream-go-webrtc` first; bump `acceleration`,
`benchmark` and `performance_benchmark` to it afterwards.

## 4. Deploy and ramp

Deploy the branch to the benchmark release (`deploy` skill):

```bash
cd ../chat/infra && source ~/.config/rocky/gcp.env
./cli/rocky fastdeploy --service accelerate-benchmark
./cli/rocky fastdeploy --service accelerate-benchmark --status
```

Then ramp and profile the top step in parallel:

```bash
cd performance_benchmark
cluster/run.sh round-3 4 1024 60s | tee out/round-3.log
loop/profile.sh round-3 512 20      # in a second shell
```

`run.sh` ends the previous generation's calls and waits a minute first; never ramp onto a
router still draining the last run, its CPU lands in yours. Rerun `cluster/setup.sh` first
if `performance_benchmark/`, `benchmark/` or `sdks/go` changed.

## 5. Record

From `out/round-<n>/report.json`: agents held, what broke the next step, reply p50/p95,
router cores at the held step and mCPU per agent, $/voice-minute, and the audio score at the
held step (`audio`: SNR and the share broken). A fix that holds more agents while the SNR
falls or glitches climb is trading quality for capacity; say so. Compare with the
previous round:

```bash
CGO_ENABLED=1 go run -tags webrtc ./cmd/perfbench compare out/round-2 out/round-3
```

One run moves by up to 10%. When a round sits at the edge (p95 just over the 3 s bar),
run it again before calling it a regression. If a fix made things worse, revert it and
say so; a round with no gain is still a round.

## 6. After ten rounds

- Add a section to `performance_benchmark/findings.md`: a row per round (fix, held, p95,
  mCPU per agent, $/voice-minute) and what is still the largest cost.
- A chart of agents held, latency and cost across rounds, and a draft PR (`pr` skill).
- `kubectl delete -f performance_benchmark/cluster/perfbench.yaml`: the caller pod holds its
  own node until it is gone.
