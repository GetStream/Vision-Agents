// Visible activity only: no model reasoning, arguments, or raw tool output.
export type ToolActivity = {
  type: string;
  tool_call_id: string;
  name: string;
  title: string;
  status: string;
  phase: string;
  summary?: string;
  product?: string;
  sdk?: string;
  started_at: string;
  execution_started_at?: string;
  finished_at?: string;
  duration_ms?: number;
};

export type ConversationMessage = {
  id: string;
  role: string;
  text: string;
  state: string;
  response_started_at: string;
  state_started_at: string;
  finished_at?: string;
  duration_ms?: number;
  attachments?: ToolActivity[] | null;
  saved: boolean;
  persistence_error?: string;
};

export type ConversationPage = {
  messages: ConversationMessage[];
  before?: string;
  context_truncated: boolean;
};

export function active(state: string): boolean {
  return [
    "thinking",
    "queued",
    "tools",
    "writing",
    "waiting_for_research",
    "running_tools",
    "writing_answer",
  ].includes(state);
}

export function activityLabel(state: string): string {
  return (
    (
      {
        queued: "Waiting for research",
        tools: "Running tools",
        writing: "Writing answer",
        running: "Running",
        thinking: "Thinking…",
        waiting_for_research: "Waiting for research",
        running_tools: "Running tools",
        writing_answer: "Writing answer",
        completed: "Completed",
        failed: "Failed",
        cancelled: "Cancelled",
        interrupted: "Interrupted",
      } as Record<string, string>
    )[state] ?? state.replaceAll("_", " ")
  );
}

export function elapsed(
  start: string,
  end: string | undefined,
  durationMS: number | undefined,
  now: number,
): string {
  const ms = end
    ? (durationMS ?? Date.parse(end) - Date.parse(start))
    : now - Date.parse(start);
  return Number.isFinite(ms) ? `${(Math.max(0, ms) / 1000).toFixed(1)}s` : "—";
}

export function mergeMessages(
  current: ConversationMessage[],
  incoming: ConversationMessage[],
): ConversationMessage[] {
  const merged = new Map(current.map((message) => [message.id, message]));
  for (const message of incoming) merged.set(message.id, message);
  return [...merged.values()].sort(
    (a, b) =>
      a.response_started_at.localeCompare(b.response_started_at) ||
      (a.role === "user" ? -1 : 1),
  );
}

// A link may point to the separate local TUI backend, never an arbitrary remote server.
export function localRouter(
  value: string | undefined,
  fallback: string,
): string {
  if (!value) return fallback;
  const url = new URL(value);
  if (
    !["http:", "https:"].includes(url.protocol) ||
    !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) ||
    url.username ||
    url.password ||
    url.pathname !== "/" ||
    url.search ||
    url.hash
  ) {
    throw new Error(
      "Dashboard router overrides must be a local HTTP address without credentials or a path.",
    );
  }
  return url.origin;
}
