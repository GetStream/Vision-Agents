"use client";

import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";

import {
  CallLink,
  duration,
  Empty,
  Failure,
  ms,
  PageHeading,
  Panel,
  Tile,
} from "@/components/ui";
import { router, type Call } from "@/lib/router";

/** The window the usage tiles cover. A week is enough to see a change and short enough to load. */
const days = 7;

export default function Overview() {
  const window = useMemo(() => {
    const to = new Date();
    const from = new Date(to.getTime() - days * 24 * 60 * 60 * 1000);
    return { from, to };
  }, []);

  const calls = useQuery({
    queryKey: ["calls", 5],
    queryFn: () => router.calls({ limit: 5 }),
    // A running call is worth seeing before somebody reloads the page.
    refetchInterval: 10_000,
  });

  const turns = useQuery({
    queryKey: ["turn-stats", window.from.toISOString()],
    queryFn: () => router.turnStats(window.from, window.to),
  });

  const spend = useQuery({
    queryKey: ["stats", "llm", window.from.toISOString()],
    queryFn: () => router.stats("llm", window.from, window.to),
  });

  const totals = useMemo(() => {
    const buckets = turns.data ?? [];
    const count = buckets.reduce((sum, bucket) => sum + bucket.turn_count, 0);
    const interrupted = buckets.reduce(
      (sum, bucket) => sum + bucket.interrupted_count,
      0,
    );
    const spoken = buckets.reduce(
      (sum, bucket) => sum + bucket.audio_out_ms_total,
      0,
    );
    // A median of medians is not a median, but for a tile that says whether the week was
    // slow it is closer than the worst bucket and honest about being a summary.
    const latencies = buckets
      .map((bucket) => bucket.roundtrip_p50_ms)
      .filter((value): value is number => typeof value === "number");
    const typical = latencies.length
      ? latencies.reduce((sum, value) => sum + value, 0) / latencies.length
      : null;

    const cost = (spend.data ?? []).reduce(
      (sum, bucket) => sum + bucket.cost_micros_total,
      0,
    );

    return { count, interrupted, spoken, typical, cost };
  }, [turns.data, spend.data]);

  return (
    <>
      <PageHeading
        title="Overview"
        description={`Calls as they happen, and what the last ${days} days cost.`}
      />

      <div className="mb-8 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile
          label="Turns"
          value={totals.count.toLocaleString()}
          hint={`${totals.interrupted.toLocaleString()} talked over`}
        />
        <Tile
          label="Typical wait"
          value={ms(totals.typical)}
          hint="Finishing a sentence to hearing an answer"
        />
        <Tile
          label="Agent spoke"
          value={`${Math.round(totals.spoken / 60_000).toLocaleString()}m`}
          hint="Audio published"
        />
        <Tile
          label="Model spend"
          value={`$${(totals.cost / 1_000_000).toFixed(2)}`}
          hint={`Completions, last ${days} days`}
        />
      </div>

      <Panel title="Latest calls">
        {calls.isError ? <Failure error={calls.error} /> : null}
        {calls.data?.length === 0 ? (
          <Empty>No calls yet. One will appear here the moment it starts.</Empty>
        ) : null}
        {calls.data?.length ? <CallTable calls={calls.data} /> : null}
      </Panel>
    </>
  );
}

function CallTable({ calls }: { calls: Call[] }) {
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs uppercase tracking-wide text-muted">
        <tr className="border-b border-line">
          <th className="px-4 py-2 font-medium">Who</th>
          <th className="px-4 py-2 font-medium">Direction</th>
          <th className="px-4 py-2 font-medium">Started</th>
          <th className="px-4 py-2 font-medium">Length</th>
          <th className="px-4 py-2 font-medium">Summary</th>
          <th className="px-4 py-2 font-medium">Score</th>
        </tr>
      </thead>
      <tbody>
        {calls.map((call) => {
          const running = !call.ended_at;
          return (
            <tr key={call.id} className="border-b border-line last:border-0">
              <td className="px-4 py-2 font-medium">
                <CallLink id={call.id}>
                  {call.direction === "inbound"
                    ? (call.from_number ?? "unknown caller")
                    : (call.to_number ?? "unknown number")}
                </CallLink>
              </td>
              <td className="px-4 py-2 text-muted">{call.direction}</td>
              <td className="px-4 py-2 text-muted tabular-nums">
                {new Date(call.started_at).toLocaleString()}
              </td>
              <td className="px-4 py-2 tabular-nums">
                {running ? (
                  <span className="inline-flex items-center gap-1.5 text-emerald-600">
                    <span className="size-1.5 rounded-full bg-emerald-500" />
                    live
                  </span>
                ) : (
                  duration(call.started_at, call.ended_at)
                )}
              </td>
              <td className="max-w-md truncate px-4 py-2 text-muted">
                {call.summary ?? "—"}
              </td>
              <td className="px-4 py-2 tabular-nums">
                {call.review_score ?? "—"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
