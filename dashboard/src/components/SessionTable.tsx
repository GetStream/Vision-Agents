import { CallLink, duration } from "@/components/ui";
import type { Call } from "@/lib/router";

/**
 * How this session reached the agent.
 *
 * Everything listed today is a voice or dashboard call. SMS, Slack and the rest will
 * arrive as their own rows once those channels write into the same list.
 */
export function sessionChannel(_call: Call): string {
  return "Call";
}

export function SessionTable({
  calls,
  agents,
}: {
  calls: Call[];
  agents?: Record<string, string>;
}) {
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs uppercase tracking-wide text-muted">
        <tr className="border-b border-line">
          <th className="px-4 py-2 font-medium">Channel</th>
          <th className="px-4 py-2 font-medium">Who</th>
          <th className="px-4 py-2 font-medium">Agent</th>
          <th className="px-4 py-2 font-medium">Direction</th>
          <th className="px-4 py-2 font-medium">Started</th>
          <th className="px-4 py-2 font-medium">Length</th>
          <th className="px-4 py-2 font-medium">Summary</th>
          <th className="px-4 py-2 font-medium">Score</th>
        </tr>
      </thead>
      <tbody>
        {calls.map((call) => {
          const running = !call.ended_at;
          const agent = call.config_id ? agents?.[call.config_id] : undefined;
          return (
            <tr key={call.id} className="border-b border-line last:border-0">
              <td className="px-4 py-2">
                <span className="rounded-md border border-line px-1.5 py-0.5 text-xs text-muted">
                  {sessionChannel(call)}
                </span>
              </td>
              <td className="px-4 py-2 font-medium">
                <CallLink id={call.id}>{party(call, agent)}</CallLink>
              </td>
              <td className="px-4 py-2 text-muted">{agent ?? "—"}</td>
              <td className="px-4 py-2 text-muted">{call.direction}</td>
              <td className="px-4 py-2 text-muted tabular-nums">
                {new Date(call.started_at).toLocaleString()}
              </td>
              <td className="px-4 py-2 tabular-nums">
                {running ? (
                  <span className="inline-flex items-center gap-1.5 text-emerald-600">
                    <span className="size-1.5 rounded-full bg-emerald-500" />
                    live
                  </span>
                ) : (
                  duration(call.started_at, call.ended_at)
                )}
              </td>
              <td className="max-w-md truncate px-4 py-2 text-muted">
                {call.summary ?? "—"}
              </td>
              <td className="px-4 py-2 tabular-nums">
                {call.review_score ?? "—"}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** party is who the session was with, or the agent when there was no number. */
function party(call: Call, agent?: string): string {
  const number =
    call.direction === "inbound" ? call.from_number : call.to_number;
  return number || agent || "Session";
}
