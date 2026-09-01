import type { SimulationCase, SimulationRun } from "@/lib/router";

/** How a run or one of its conversations came out, at a glance. */
const stateStyles: Record<string, string> = {
  running: "bg-sky-500/10 text-sky-600 border-sky-500/20",
  pending: "bg-line/40 text-muted border-line",
  passed: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
  failed: "bg-red-500/10 text-red-600 border-red-500/20",
  errored: "bg-amber-500/10 text-amber-600 border-amber-500/20",
  cancelled: "bg-line/40 text-muted border-line",
};

export function StateBadge({ state }: { state: string }) {
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-xs ${
        stateStyles[state] ?? stateStyles.pending
      }`}
    >
      {state}
    </span>
  );
}

/** running reports whether a run is still having its conversations. */
export function running(run: SimulationRun | undefined): boolean {
  return run?.state === "running";
}

/** tally is how a run went, in the shape somebody scans a list in. */
export function tally(run: SimulationRun): string {
  if (run.state === "running") {
    return `${run.passed + run.failed} of ${run.cases} done`;
  }
  return `${run.passed} of ${run.cases} passed`;
}

/** Why a conversation stopped, said the way somebody reading it would say it. */
export const endings: Record<string, string> = {
  complete: "the caller had asked everything",
  turns: "it ran out of turns",
  timeout: "it ran out of time",
  failed: "the agent stopped answering",
};

/** ordered is a run's conversations in the order they were asked. */
export function ordered(run: SimulationRun): SimulationCase[] {
  return [...(run.conversations ?? [])].sort((l, r) => l.variation - r.variation);
}
