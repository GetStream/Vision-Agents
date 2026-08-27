"use client";

import { Empty, ms, Panel } from "@/components/ui";
import type { TimelineEntry } from "@/lib/router";
import type { Timing } from "@/lib/useSession";

type Row = {
  turn: string;
  stt: number;
  llm: number;
  tts: number;
  roundtrip: number;
  interrupted: boolean;
};

/**
 * The bar is the wait broken into the three legs that made it up, so a slow turn can be
 * blamed on the transcriber, the model or the voice by looking rather than by comparing
 * numbers. Anything the three do not account for is the agent's own handling.
 */
const legs = [
  { key: "stt" as const, label: "transcribe", colour: "bg-sky-500" },
  { key: "llm" as const, label: "think", colour: "bg-indigo-500" },
  { key: "tts" as const, label: "speak", colour: "bg-emerald-500" },
];

export function LatencyPanel({
  live,
  stored,
  className = "",
}: {
  live: Timing[];
  stored: TimelineEntry[];
  className?: string;
}) {
  const rows: Row[] = live.length
    ? live.map((timing) => ({
        turn: timing.turn_id,
        stt: timing.stt_latency_ms ?? 0,
        llm: timing.llm_ttft_ms ?? 0,
        tts: timing.tts_ttfb_ms ?? 0,
        roundtrip: timing.roundtrip_ms ?? 0,
        interrupted: Boolean(timing.interrupted),
      }))
    : stored.map((entry) => ({
        turn: entry.turn_id,
        stt: 0,
        llm: 0,
        tts: 0,
        roundtrip: entry.roundtrip_ms ?? 0,
        interrupted: Boolean(entry.interrupted),
      }));

  const worst = Math.max(1, ...rows.map((row) => row.roundtrip));

  return (
    <Panel
      title="What each turn cost the caller in waiting"
      aside={
        <span className="flex gap-3 text-xs text-muted">
          {legs.map((leg) => (
            <span key={leg.key} className="flex items-center gap-1.5">
              <span className={`size-2 rounded-sm ${leg.colour}`} />
              {leg.label}
            </span>
          ))}
        </span>
      }
      className={className}
    >
      {rows.length === 0 ? (
        <Empty>No turns have finished yet.</Empty>
      ) : (
        <ol className="max-h-96 overflow-y-auto">
          {rows.map((row, index) => (
            <li
              key={`${row.turn}-${index}`}
              className="flex items-center gap-3 border-b border-line px-4 py-2 text-sm last:border-0"
            >
              <span className="w-8 shrink-0 font-mono text-xs tabular-nums text-muted">
                {index + 1}
              </span>
              <span className="flex h-3 flex-1 overflow-hidden rounded-sm bg-line/40">
                {legs.map((leg) => (
                  <span
                    key={leg.key}
                    className={leg.colour}
                    style={{ width: `${(row[leg.key] / worst) * 100}%` }}
                    title={`${leg.label} ${ms(row[leg.key])}`}
                  />
                ))}
              </span>
              <span className="w-16 shrink-0 text-right font-mono text-xs tabular-nums">
                {ms(row.roundtrip)}
              </span>
              <span className="w-20 shrink-0 text-right text-xs text-muted">
                {row.interrupted ? "talked over" : ""}
              </span>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}
