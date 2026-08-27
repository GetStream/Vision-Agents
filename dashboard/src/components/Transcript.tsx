"use client";

import { useEffect, useRef } from "react";

import { clock, Empty, Panel } from "@/components/ui";
import type { TranscriptMessage } from "@/lib/router";
import type { Hearing, Line } from "@/lib/useSession";

export function Transcript({
  running,
  hearing,
  lines,
  stored,
}: {
  running: boolean;
  hearing: Hearing | null;
  lines: Line[];
  stored: TranscriptMessage[];
}) {
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "nearest" });
  }, [lines.length, hearing?.text]);

  const said = running
    ? lines.map((line) => ({
        agent: line.agent,
        speaker: line.speaker,
        text: line.text,
        at: new Date(line.at).toISOString(),
      }))
    : stored.map((message) => ({
        agent: message.speaker === "agent",
        speaker: message.speaker,
        text: message.text,
        at: message.created_at,
      }));

  return (
    <Panel
      title={running ? "What is being said" : "What was said"}
      aside={
        running ? (
          <span className="text-xs text-muted">live from the call</span>
        ) : null
      }
    >
      {said.length === 0 && !hearing ? (
        <Empty>
          {running
            ? "Waiting for somebody to speak."
            : "Nothing was written down for this call."}
        </Empty>
      ) : (
        <div className="max-h-[28rem] space-y-3 overflow-y-auto px-4 py-3">
          {said.map((line, index) => (
            <div key={`${line.at}-${index}`} className="text-sm">
              <div className="mb-0.5 flex items-baseline gap-2">
                <span
                  className={`text-xs font-medium ${
                    line.agent ? "text-emerald-600" : "text-sky-600"
                  }`}
                >
                  {line.speaker}
                </span>
                <span className="font-mono text-xs tabular-nums text-muted">
                  {clock(line.at)}
                </span>
              </div>
              <p className="leading-relaxed">{line.text}</p>
            </div>
          ))}
          {hearing ? (
            <div className="text-sm">
              <div className="mb-0.5 text-xs font-medium text-sky-600">
                {hearing.participant.name ??
                  hearing.participant.user_id ??
                  "caller"}
              </div>
              {/* The words are still changing, which is what the agent is looking at
                  while it decides whether they have finished. */}
              <p className="leading-relaxed text-muted italic">
                {hearing.text}
                <span className="ml-0.5 animate-pulse">▍</span>
              </p>
            </div>
          ) : null}
          <div ref={bottom} />
        </div>
      )}
    </Panel>
  );
}
