"use client";

import { merge } from "@/components/DecisionLog";
import { Empty, Panel } from "@/components/ui";
import type { CallEvent, TimelineEntry } from "@/lib/router";
import type { Decision } from "@/lib/useSession";

/** One stretch of the call where one side had the floor. */
type Span = {
  agent: boolean;
  from: number;
  to: number;
  /** cut marks a stretch that ended because the other side took the floor. */
  cut: boolean;
};

/**
 * spansOf reads who was talking out of the decision trail.
 *
 * Nothing records a speaking span directly, and nothing should: the agent knows when it
 * started answering and when it stopped, and it knows when it was hearing somebody,
 * because it decided something about each. Reading the spans back out of those judgements
 * means the picture cannot drift from the reasoning underneath it.
 */
export function spansOf(judgements: { at: string; kind: string }[]): Span[] {
  if (judgements.length === 0) {
    return [];
  }

  const spans: Span[] = [];
  let hearing: number | null = null;
  let answering: number | null = null;

  const close = (agent: boolean, at: number, cut: boolean) => {
    const from = agent ? answering : hearing;
    if (from === null) {
      return;
    }
    // A span with no width is a judgement made about a moment rather than a stretch, and
    // drawing it as a hairline is more honest than dropping it.
    spans.push({ agent, from, to: Math.max(at, from + 200), cut });
    if (agent) {
      answering = null;
    } else {
      hearing = null;
    }
  };

  for (const judgement of judgements) {
    const at = new Date(judgement.at).getTime();
    switch (judgement.kind) {
      case "ask":
        if (hearing === null) {
          hearing = at;
        }
        break;
      case "wait":
        break;
      case "ignore":
      case "queue":
        close(false, at, false);
        break;
      case "answer":
        close(false, at, false);
        if (answering === null) {
          answering = at;
        }
        break;
      case "interrupt":
        close(true, at, true);
        break;
      case "shorten":
        close(true, at, true);
        break;
      default:
        break;
    }
  }

  const last = new Date(judgements[judgements.length - 1].at).getTime();
  close(false, last, false);
  close(true, last, false);

  return spans.sort((left, right) => left.from - right.from);
}

/** spansFromTimeline is the fallback for a call whose decisions were never written down. */
function spansFromTimeline(entries: TimelineEntry[]): Span[] {
  return entries.map((entry) => {
    const started = new Date(entry.started_at).getTime();
    const spoken = entry.audio_out_ms ?? 0;
    const waited = entry.roundtrip_ms ?? 0;
    return {
      agent: true,
      from: started + waited,
      to: started + waited + Math.max(spoken, 200),
      cut: Boolean(entry.interrupted),
    };
  });
}

export function SpeakingTimeline({
  startedAt,
  endedAt,
  stored,
  live,
  fallback,
  className = "",
}: {
  startedAt: string;
  endedAt?: string | null;
  stored: CallEvent[];
  live: Decision[];
  fallback: TimelineEntry[];
  className?: string;
}) {
  const judgements = merge(stored, live);
  const spans = judgements.length
    ? spansOf(judgements)
    : spansFromTimeline(fallback);

  const from = new Date(startedAt).getTime();
  // A running call has no end, so the window ends at the last thing anybody did. Reading
  // the clock instead would make the picture shift under a re-render nobody asked for.
  const to = endedAt
    ? new Date(endedAt).getTime()
    : spans.reduce((latest, span) => Math.max(latest, span.to), from);
  const width = Math.max(to - from, 1000);

  const placed = spans
    .filter((span) => span.to > from)
    .map((span) => ({
      ...span,
      left: ((Math.max(span.from, from) - from) / width) * 100,
      size: ((Math.min(span.to, to) - Math.max(span.from, from)) / width) * 100,
    }));

  return (
    <Panel
      title="Who was talking"
      aside={
        <span className="flex gap-3 text-xs text-muted">
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-sm bg-sky-500" /> caller
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2 rounded-sm bg-emerald-500" /> agent
          </span>
        </span>
      }
      className={className}
    >
      {placed.length === 0 ? (
        <Empty>Nobody has said anything yet.</Empty>
      ) : (
        <div className="space-y-2 px-4 py-4">
          {[false, true].map((agent) => (
            <div
              key={String(agent)}
              className="relative h-7 overflow-hidden rounded-md bg-line/40"
            >
              {placed
                .filter((span) => span.agent === agent)
                .map((span, index) => (
                  <div
                    key={index}
                    title={`${agent ? "agent" : "caller"}${span.cut ? ", cut short" : ""}`}
                    className={`absolute inset-y-1 rounded-sm ${
                      agent ? "bg-emerald-500" : "bg-sky-500"
                    } ${span.cut ? "opacity-60" : ""}`}
                    style={{
                      left: `${span.left}%`,
                      width: `${Math.max(span.size, 0.3)}%`,
                    }}
                  />
                ))}
            </div>
          ))}
          <div className="flex justify-between pt-1 font-mono text-xs text-muted">
            <span>0:00</span>
            <span>
              {Math.floor(width / 60000)}:
              {String(Math.floor((width % 60000) / 1000)).padStart(2, "0")}
            </span>
          </div>
        </div>
      )}
    </Panel>
  );
}
