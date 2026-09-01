"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useMemo } from "react";

import { SessionTable } from "@/components/SessionTable";
import {
  Empty,
  Failure,
  ms,
  Notice,
  PageHeading,
  Panel,
  Tile,
} from "@/components/ui";
import { router, type AgentConfig } from "@/lib/router";

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

  const configs = useQuery({ queryKey: ["configs"], queryFn: router.configs });

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

  const agents = useMemo(
    () =>
      Object.fromEntries(
        (configs.data ?? []).map((config) => [config.id, config.name]),
      ),
    [configs.data],
  );

  return (
    <>
      <PageHeading
        title="Overview"
        description={`Calls as they happen, and what the last ${days} days cost.`}
      />

      <Notice className="mb-6">
        The easiest way to set up a new agent is to ask a coding agent for one:{" "}
        <span className="text-foreground">
          Build my voice AI with{" "}
          <a
            href="https://streamrtc.ai/skill.md"
            className="underline underline-offset-2"
            target="_blank"
            rel="noreferrer"
          >
            streamrtc.ai/skill.md
          </a>
        </span>
        . The skill knows the models, the targets and the shape of a config, so it
        writes one rather than you filling the form in by hand.
      </Notice>

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

      <Panel
        title="Agents"
        className="mb-8"
        aside={
          <Link href="/agents" className="text-xs text-muted hover:underline">
            Manage
          </Link>
        }
      >
        {configs.isError ? <Failure error={configs.error} /> : null}
        {configs.data?.length === 0 ? (
          <Empty>No agents yet. A session without one takes the defaults.</Empty>
        ) : null}
        {configs.data?.length ? <ConfigList configs={configs.data} /> : null}
      </Panel>

      <Panel
        title="Latest sessions"
        aside={
          <Link href="/sessions" className="text-xs text-muted hover:underline">
            See all
          </Link>
        }
      >
        {calls.isError ? <Failure error={calls.error} /> : null}
        {calls.data?.length === 0 ? (
          <Empty>No sessions yet. A call will appear here the moment it starts.</Empty>
        ) : null}
        {calls.data?.length ? (
          <SessionTable calls={calls.data} agents={agents} />
        ) : null}
      </Panel>
    </>
  );
}

function ConfigList({ configs }: { configs: AgentConfig[] }) {
  return (
    <ul>
      {configs.map((config) => {
        // A config need not name any of these: a session that passes its own targets
        // stores only what the agent is, so there is nothing to badge.
        const targets = [
          config.stt,
          config.llm,
          config.tts,
          config.subagent,
        ].filter((target): target is string => Boolean(target));

        return (
          <li
            key={config.id}
            className="border-b border-line px-4 py-3 last:border-0"
          >
            <Link
              href={`/agents/${config.id}`}
              className="font-medium hover:underline"
            >
              {config.name}
            </Link>
            <p className="mt-0.5 line-clamp-1 text-sm text-muted">
              {config.instructions || "No instructions."}
            </p>
            {targets.length ? (
              <div className="mt-1.5 flex flex-wrap gap-1.5 text-xs text-muted">
                {targets.map((target) => (
                  <span
                    key={target}
                    className="rounded-md border border-line px-1.5 py-0.5"
                  >
                    {target}
                  </span>
                ))}
              </div>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

