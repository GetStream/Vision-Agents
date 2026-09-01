"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { StateBadge, tally } from "@/components/SimulationState";
import {
  Button,
  clock,
  Empty,
  Failure,
  PageHeading,
  Panel,
  Tabs,
} from "@/components/ui";
import { router, type Simulation } from "@/lib/router";

export default function Simulations() {
  const [tab, setTab] = useState("simulations");

  return (
    <>
      <PageHeading
        title="Simulations"
        description="A conversation to have with an agent and something that has to be true at the end of it. Run it again after the agent changes."
        action={
          <Link
            href="/simulations/new"
            className="rounded-lg bg-foreground px-3 py-1.5 text-sm font-medium text-background transition hover:opacity-90"
          >
            New simulation
          </Link>
        }
      />

      <Tabs
        tabs={[
          { id: "simulations", label: "Simulations" },
          { id: "logs", label: "Logs" },
        ]}
        active={tab}
        onSelect={setTab}
      />

      {tab === "simulations" ? <Written /> : <Log />}
    </>
  );
}

/** Written is every simulation the customer has, and the button that runs one. */
function Written() {
  const client = useQueryClient();
  const simulations = useQuery({
    queryKey: ["simulations"],
    queryFn: router.simulations,
  });

  const remove = useMutation({
    mutationFn: router.deleteSimulation,
    onSuccess: () => client.invalidateQueries({ queryKey: ["simulations"] }),
  });

  if (simulations.isError) return <Failure error={simulations.error} />;
  if (!simulations.data)
    return <p className="text-sm text-muted">Loading the simulations…</p>;

  return (
    <Panel title="Your simulations">
      {simulations.data.length === 0 ? (
        <Empty>
          Nothing written down yet. A simulation is a scenario and one thing that has to
          be true at the end of it.
        </Empty>
      ) : (
        <ul className="divide-y divide-line">
          {simulations.data.map((simulation) => (
            <Row
              key={simulation.id}
              simulation={simulation}
              onDelete={() => remove.mutate(simulation.id)}
            />
          ))}
        </ul>
      )}
    </Panel>
  );
}

function Row({
  simulation,
  onDelete,
}: {
  simulation: Simulation;
  onDelete: () => void;
}) {
  const client = useQueryClient();
  const run = useMutation({
    mutationFn: () => router.runSimulation(simulation.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["simulation-runs"] }),
  });

  return (
    <li className="flex items-start justify-between gap-4 px-4 py-3">
      <div className="min-w-0">
        <Link
          href={`/simulations/${simulation.id}`}
          className="text-sm font-medium hover:underline"
        >
          {simulation.name}
        </Link>
        <p className="mt-0.5 truncate text-xs text-muted">{simulation.scenario}</p>
        <p className="mt-1 text-xs text-muted">
          {simulation.mode} ·{" "}
          {simulation.variations > 1
            ? `${simulation.variations} ways of asking`
            : "one way of asking"}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {run.isError ? (
          <span className="max-w-60 text-xs text-red-600">
            {(run.error as Error).message}
          </span>
        ) : null}
        <Button onClick={() => run.mutate()} disabled={run.isPending}>
          {run.isPending ? "Starting…" : "Run"}
        </Button>
        <Button variant="danger" onClick={onDelete}>
          Delete
        </Button>
      </div>
    </li>
  );
}

/** Log is everything that has been run lately, newest first. */
function Log() {
  const runs = useQuery({
    queryKey: ["simulation-runs"],
    queryFn: () => router.simulationRuns(),
    // A run in the list is still going, so the log keeps up with it on its own.
    refetchInterval: 5_000,
  });
  const simulations = useQuery({
    queryKey: ["simulations"],
    queryFn: router.simulations,
  });

  if (runs.isError) return <Failure error={runs.error} />;
  if (!runs.data) return <p className="text-sm text-muted">Loading the log…</p>;

  const named = new Map((simulations.data ?? []).map((one) => [one.id, one.name]));

  return (
    <Panel title="Simulation log" aside={<span className="text-xs text-muted">{runs.data.length} runs</span>}>
      {runs.data.length === 0 ? (
        <Empty>Nothing has been run yet.</Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase tracking-wide text-muted">
              <tr>
                <th className="px-4 py-2 font-medium">Started</th>
                <th className="px-4 py-2 font-medium">Simulation</th>
                <th className="px-4 py-2 font-medium">Result</th>
                <th className="px-4 py-2 font-medium">Conversations</th>
              </tr>
            </thead>
            <tbody>
              {runs.data.map((one) => (
                <tr key={one.id} className="border-b border-line last:border-0">
                  <td className="px-4 py-2 font-mono text-xs tabular-nums text-muted">
                    {clock(one.started_at)}
                  </td>
                  <td className="px-4 py-2">
                    <Link
                      href={`/simulation-runs/${one.id}`}
                      className="hover:underline"
                    >
                      {named.get(one.simulation_id) ?? one.simulation_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2">
                    <StateBadge state={one.state} />
                  </td>
                  <td className="px-4 py-2 tabular-nums text-muted">{tally(one)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
