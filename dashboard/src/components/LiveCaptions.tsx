"use client";

import type { Hearing, Line } from "@/lib/useSession";

/** How many settled lines sit under the orb: enough to follow, short of a transcript. */
const shown = 3;

/**
 * LiveCaptions is what is being said right now, directly under the orb and outside it.
 *
 * The orb says who holds the floor; this says what they are using it for. It deliberately
 * keeps only the last few turns — the whole conversation is in the transcript panel, and
 * repeating it here would make the top of the page scroll.
 */
export function LiveCaptions({
  lines,
  hearing,
  className = "",
}: {
  lines: Line[];
  hearing: Hearing | null;
  className?: string;
}) {
  const recent = lines.slice(-shown);

  if (recent.length === 0 && !hearing) {
    return (
      <p className={`px-1 text-sm text-muted ${className}`}>
        Nothing has been said yet.
      </p>
    );
  }

  return (
    <div className={`space-y-1.5 px-1 ${className}`}>
      {recent.map((line, index) => (
        <p
          key={`${line.at}-${index}`}
          // Older turns are dimmed so the newest one is found without it having to move.
          className={`text-sm leading-snug ${
            index === recent.length - 1 && !hearing ? "" : "text-muted"
          }`}
        >
          <span
            className={`mr-2 text-xs font-medium ${
              line.agent ? "text-emerald-600" : "text-sky-600"
            }`}
          >
            {line.speaker}
          </span>
          {line.text}
        </p>
      ))}

      {hearing ? (
        <p className="text-sm leading-snug">
          <span className="mr-2 text-xs font-medium text-sky-600">
            {hearing.participant.name ??
              hearing.participant.user_id ??
              "caller"}
          </span>
          {/* Still changing, which is what the agent is looking at while it decides
              whether they have finished. */}
          <span className="italic">{hearing.text}</span>
          <span className="ml-0.5 animate-pulse">▍</span>
        </p>
      ) : null}
    </div>
  );
}
