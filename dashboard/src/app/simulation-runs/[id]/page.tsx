"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { use } from "react";

import {
  endings,
  ordered,
  running,
  StateBadge,
  tally,
} from "@/components/SimulationState";
import {
  Button,
  CallLink,
  Empty,
  Failure,
  PageHeading,
  Panel,
} from "@/components/ui";
import { router, type SimulationCase } from "@/lib/router";

export default function SimulationRunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const client = useQueryClient();

  const run = useQuery({
    queryKey: ["simulation-run", id],
    queryFn: () => router.simulationRun(id),
    // The conversations land one at a time, so a run being watched keeps up with itself.
    refetchInterval: (query) => (running(query.state.data) ? 2_000 : false),
  });

  const cancel = useMutation({
    mutationFn: () => router.cancelSimulationRun(id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["simulation-run", id] }),
  });

  if (run.isError) return <Failure error={run.error} />;
  if (!run.data) return <p className="text-sm text-muted">Loading the run…</p>;

  const conversations = ordered(run.data);

  return (
    <>
      <PageHeading
        title="Simulation run"
        description={run.data.assertion ?? undefined}
        action={
          <Link
            href={`/simulations/${run.data.simulation_id}`}
            className="text-sm text-muted hover:underline"
          >
            Back to the simulation
          </Link>
        }
      />

      <Panel
        title="How it went"
        aside={
          <div className="flex items-center gap-3">
            <span className="text-xs tabular-nums text-muted">{tally(run.data)}</span>
            <StateBadge state={run.data.state} />
            {running(run.data) ? (
              <Button
                variant="quiet"
                onClick={() => cancel.mutate()}
                disabled={cancel.isPending}
              >
                {cancel.isPending ? "Stopping…" : "Stop"}
              </Button>
            ) : null}
          </div>
        }
      >
        {run.data.error ? (
          <p className="px-4 py-3 text-sm text-red-600">{run.data.error}</p>
        ) : (
          <p className="px-4 py-3 text-sm text-muted">
            {run.data.scenario}
          </p>
        )}
      </Panel>

      <div className="mt-6 space-y-4">
        {conversations.length === 0 ? (
          <Empty>The conversations have not started yet.</Empty>
        ) : (
          conversations.map((one) => <Conversation key={one.id} one={one} />)
        )}
      </div>
    </>
  );
}

/** Conversation is one way of asking, and what the judge made of how it went. */
function Conversation({ one }: { one: SimulationCase }) {
  return (
    <Panel
      title={one.variation === 0 ? "As written" : `Way of asking ${one.variation + 1}`}
      aside={
        <div className="flex items-center gap-3">
          {one.score ? (
            <span className="text-xs text-muted">confidence {one.score}/5</span>
          ) : null}
          <StateBadge state={one.state} />
        </div>
      }
    >
      <div className="space-y-3 px-4 py-3">
        <p className="text-xs text-muted">{one.scenario}</p>

        {one.verdict ? (
          <p className="rounded-lg border border-line bg-background px-3 py-2 text-sm">
            {one.verdict}
          </p>
        ) : null}
        {one.error ? (
          <p className="text-sm text-red-600">{one.error}</p>
        ) : null}

        {(one.transcript ?? []).length > 0 ? (
          <ol className="max-h-[28rem] space-y-1 overflow-y-auto">
            {(one.transcript ?? []).map((line, index) => (
              <li key={index} className="flex gap-3 text-sm">
                <span
                  className={`w-14 shrink-0 text-xs ${
                    line.caller ? "text-sky-600" : "text-muted"
                  }`}
                >
                  {line.caller ? "Caller" : "Agent"}
                </span>
                <span>
                  {line.text}
                  {line.intended && line.intended !== line.text ? (
                    <span className="ml-2 text-xs text-muted">
                      (meant: {line.intended})
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ol>
        ) : null}

        <p className="text-xs text-muted">
          {one.turns} {one.turns === 1 ? "turn" : "turns"}
          {one.ended ? ` · ${endings[one.ended] ?? one.ended}` : ""}
          {one.call_id ? " · " : ""}
          {one.call_id ? <CallLink id={one.call_id}>the call</CallLink> : null}
        </p>
      </div>
    </Panel>
  );
}
