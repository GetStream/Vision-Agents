"use client";

import { useEffect, useRef } from "react";

import { clock, Empty, ms, Panel } from "@/components/ui";
import type { CallEvent, DecisionKind } from "@/lib/router";
import type { Decision } from "@/lib/useSession";

/**
 * What each judgement looks like, so a log can be scanned rather than read. The colours
 * separate the three things a reader is looking for: the agent answering, the agent
 * choosing not to, and the agent taking the floor from somebody.
 */
const styles: Record<string, string> = {
  answer: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
  ask: "bg-sky-500/10 text-sky-600 border-sky-500/20",
  wait: "bg-amber-500/10 text-amber-600 border-amber-500/20",
  ignore: "bg-zinc-500/10 text-zinc-500 border-zinc-500/20",
  queue: "bg-amber-500/10 text-amber-600 border-amber-500/20",
  interrupt: "bg-red-500/10 text-red-600 border-red-500/20",
  shorten: "bg-orange-500/10 text-orange-600 border-orange-500/20",
  backchannel: "bg-violet-500/10 text-violet-600 border-violet-500/20",
  supersede: "bg-zinc-500/10 text-zinc-500 border-zinc-500/20",
  compact: "bg-blue-500/10 text-blue-600 border-blue-500/20",
  delegate: "bg-indigo-500/10 text-indigo-600 border-indigo-500/20",
  fail: "bg-red-500/10 text-red-600 border-red-500/20",
};

export type Judgement = {
  at: string;
  kind: DecisionKind;
  reason: string;
  turn_id?: string;
  said?: string;
  latency_ms?: number;
};

/**
 * merge puts the stored trail and the live one together.
 *
 * A page opened mid-call has both: everything up to now was fetched, and everything since
 * arrived on the socket. They overlap, and the timestamp is what settles it, since a
 * judgement is written down with the same clock reading it was reported with.
 */
export function merge(stored: CallEvent[], live: Decision[]): Judgement[] {
  const seen = new Set(stored.map((event) => `${event.at}:${event.kind}`));
  const merged: Judgement[] = [...stored];
  for (const decision of live) {
    if (!seen.has(`${decision.at}:${decision.kind}`)) {
      merged.push(decision);
    }
  }
  return merged.sort((left, right) => left.at.localeCompare(right.at));
}

export function DecisionLog({
  stored,
  live,
}: {
  stored: CallEvent[];
  live: Decision[];
}) {
  const judgements = merge(stored, live);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "nearest" });
  }, [judgements.length]);

  return (
    <Panel
      title="What the conversation decided"
      aside={
        <span className="text-xs text-muted">{judgements.length} judgements</span>
      }
    >
      {judgements.length === 0 ? (
        <Empty>Nothing decided yet.</Empty>
      ) : (
        <ol className="max-h-[28rem] overflow-y-auto">
          {judgements.map((judgement, index) => (
            <li
              key={`${judgement.at}-${index}`}
              className="flex gap-3 border-b border-line px-4 py-2 text-sm last:border-0"
            >
              <span className="shrink-0 pt-0.5 font-mono text-xs tabular-nums text-muted">
                {clock(judgement.at)}
              </span>
              <span
                className={`h-fit shrink-0 rounded-md border px-1.5 py-0.5 text-xs font-medium ${
                  styles[judgement.kind] ?? styles.ignore
                }`}
              >
                {judgement.kind}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block">{judgement.reason}</span>
                {judgement.said ? (
                  <span className="mt-0.5 block truncate text-xs text-muted">
                    “{judgement.said}”
                  </span>
                ) : null}
              </span>
              {judgement.latency_ms ? (
                <span className="shrink-0 pt-0.5 font-mono text-xs tabular-nums text-muted">
                  {ms(judgement.latency_ms)}
                </span>
              ) : null}
            </li>
          ))}
          <div ref={bottom} />
        </ol>
      )}
    </Panel>
  );
}
