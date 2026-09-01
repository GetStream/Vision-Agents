"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { use, useState } from "react";

import { SimulationForm } from "@/components/SimulationForm";
import { StateBadge, tally } from "@/components/SimulationState";
import {
  Button,
  clock,
  Empty,
  Failure,
  PageHeading,
  Panel,
} from "@/components/ui";
import {
  router,
  type Simulation,
  type SimulationRequest,
} from "@/lib/router";

/** draftOf turns a stored simulation back into what asking for it looks like. */
function draftOf(simulation: Simulation): SimulationRequest {
  return {
    name: simulation.name,
    mode: simulation.mode,
    config_id: simulation.config_id,
    scenario: simulation.scenario,
    assertion: simulation.assertion,
    variations: simulation.variations,
    max_turns: simulation.max_turns,
    judge_target: simulation.judge_target,
    caller_target: simulation.caller_target,
    tags: simulation.tags,
  };
}

export default function SimulationPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const client = useQueryClient();
  const [edits, setEdits] = useState<SimulationRequest | null>(null);

  const simulation = useQuery({
    queryKey: ["simulation", id],
    queryFn: () => router.simulation(id),
  });
  const configs = useQuery({ queryKey: ["configs"], queryFn: router.configs });

  const runs = useQuery({
    queryKey: ["simulation-runs", id],
    queryFn: () => router.simulationRuns({ simulationID: id }),
    // A run that is still going is what somebody who just pressed Run is looking at.
    refetchInterval: (query) =>
      (query.state.data ?? []).some((one) => one.state === "running") ? 3_000 : false,
  });

  const save = useMutation({
    mutationFn: (body: SimulationRequest) => router.updateSimulation(id, body),
    onSuccess: (updated) => {
      client.setQueryData(["simulation", id], updated);
      client.invalidateQueries({ queryKey: ["simulations"] });
      setEdits(null);
    },
  });

  const run = useMutation({
    mutationFn: () => router.runSimulation(id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["simulation-runs", id] }),
  });

  if (simulation.isError) return <Failure error={simulation.error} />;
  if (!simulation.data)
    return <p className="text-sm text-muted">Loading the simulation…</p>;

  const draft = edits ?? draftOf(simulation.data);

  return (
    <>
      <PageHeading
        title={simulation.data.name}
        description="What to ask, and what has to be true at the end of it."
        action={
          <Link href="/simulations" className="text-sm text-muted hover:underline">
            Back to simulations
          </Link>
        }
      />

      <div className="space-y-6">
        <Panel
          title="What it asks"
          aside={
            <div className="flex items-center gap-2">
              {run.isError ? (
                <span className="max-w-60 text-xs text-red-600">
                  {(run.error as Error).message}
                </span>
              ) : null}
              <Button
                onClick={() => run.mutate()}
                disabled={run.isPending || Boolean(edits)}
                title={edits ? "Save the changes first" : undefined}
              >
                {run.isPending ? "Starting…" : "Run"}
              </Button>
            </div>
          }
        >
          <form
            className="space-y-4 px-4 py-4"
            onSubmit={(event) => {
              event.preventDefault();
              save.mutate(draft);
            }}
          >
            <SimulationForm
              simulation={draft}
              configs={configs.data ?? []}
              onChange={setEdits}
            />

            {save.error ? <Failure error={save.error} /> : null}

            <div className="flex gap-2">
              <Button type="submit" disabled={save.isPending || !draft.name || !edits}>
                {save.isPending ? "Saving…" : edits ? "Save" : "Saved"}
              </Button>
              <Button
                variant="quiet"
                onClick={() => setEdits(null)}
                disabled={!edits}
              >
                Reset
              </Button>
            </div>
          </form>
        </Panel>

        <Panel title="Past runs">
          {(runs.data ?? []).length === 0 ? (
            <Empty>This has not been run yet.</Empty>
          ) : (
            <ul className="divide-y divide-line">
              {(runs.data ?? []).map((one) => (
                <li
                  key={one.id}
                  className="flex items-center justify-between gap-4 px-4 py-3"
                >
                  <Link
                    href={`/simulation-runs/${one.id}`}
                    className="font-mono text-xs tabular-nums text-muted hover:underline"
                  >
                    {clock(one.started_at)}
                  </Link>
                  <div className="flex items-center gap-3">
                    <span className="text-xs tabular-nums text-muted">{tally(one)}</span>
                    <StateBadge state={one.state} />
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </>
  );
}
