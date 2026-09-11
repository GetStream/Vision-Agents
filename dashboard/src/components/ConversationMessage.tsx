"use client";

import { useEffect, useState } from "react";
import { renderText } from "stream-chat-react";
import {
  active,
  activityLabel,
  elapsed,
  type ConversationMessage as Message,
  type ToolActivity,
} from "@/lib/conversation";

function Tool({
  tool,
  now,
  ticking,
}: {
  tool: ToolActivity;
  now: number;
  ticking: boolean;
}) {
  const running =
    !tool.finished_at &&
    !["completed", "failed", "cancelled", "interrupted"].includes(tool.status);
  const queueEnd = tool.execution_started_at ?? tool.finished_at;
  return (
    <li
      className="rounded-xl border border-line bg-background px-4 py-3"
      data-tool-id={tool.tool_call_id}
    >
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span
          aria-hidden
          className={
            running && ticking ? "animate-pulse text-sky-500" : "text-muted"
          }
        >
          {running ? "◌" : tool.status === "completed" ? "✓" : "✕"}
        </span>
        <span className="font-medium">{tool.title || tool.name}</span>
        <span className="ml-auto font-mono text-xs tabular-nums text-muted">
          {elapsed(tool.started_at, tool.finished_at, tool.duration_ms, now)}
        </span>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
        <span>
          {activityLabel(tool.status)}
          {tool.phase && tool.phase !== tool.status
            ? ` · ${tool.phase.replaceAll("_", " ")}`
            : ""}
        </span>
        {tool.product && (
          <span>
            {tool.product}/{tool.sdk}
          </span>
        )}
        {tool.execution_started_at ? (
          <span>
            Queued {elapsed(tool.started_at, queueEnd, undefined, now)} ·
            Execution{" "}
            {elapsed(
              tool.execution_started_at,
              tool.finished_at,
              undefined,
              now,
            )}
          </span>
        ) : tool.phase === "queued" ? (
          <span>
            Queued {elapsed(tool.started_at, undefined, undefined, now)}
          </span>
        ) : null}
      </div>
      {tool.summary && (
        <p className="mt-2 text-sm text-muted">{tool.summary}</p>
      )}
    </li>
  );
}

export function ConversationMessage({
  message,
  live = true,
  showPersistence = true,
  showTiming = true,
  speaker,
}: {
  message: Message;
  live?: boolean;
  showPersistence?: boolean;
  showTiming?: boolean;
  speaker?: string;
}) {
  const working = active(message.state) && !message.finished_at;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!working || !live) return;
    const timer = setInterval(() => setNow(Date.now()), 100);
    return () => clearInterval(timer);
  }, [working, live]);
  const user = message.role === "user";
  return (
    <article
      className={`min-w-0 rounded-2xl border px-5 py-4 ${user ? "ml-auto w-fit max-w-[90%] border-sky-500/15 bg-sky-500/5" : "w-full border-line bg-surface"}`}
      data-message-id={message.id}
    >
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <span
          className={`font-semibold ${user ? "text-sky-600" : "text-emerald-600"}`}
        >
          {speaker ?? (user ? "You" : "Agent")}
        </span>
        {!user && (
          <>
            <span className={working ? "text-sky-600" : "text-muted"}>
              {working && (
                <span
                  aria-hidden
                  className={live ? "mr-1 inline-block animate-pulse" : "mr-1"}
                >
                  ●
                </span>
              )}
              {activityLabel(message.state)}
            </span>
            {working && showTiming && (
              <span className="font-mono tabular-nums text-muted">
                {elapsed(message.state_started_at, undefined, undefined, now)}
              </span>
            )}
            {showTiming && (
              <span className="ml-auto font-mono tabular-nums text-muted">
                {elapsed(
                  message.response_started_at,
                  message.finished_at,
                  message.duration_ms,
                  now,
                )}{" "}
                total
              </span>
            )}
          </>
        )}
      </div>
      {message.text && (
        <div className="conversation-markdown text-sm leading-7">
          {renderText(message.text)}
        </div>
      )}
      {!!message.attachments?.length && (
        <ul aria-label="Tool activity" className="mt-4 space-y-2">
          {message.attachments
            .filter((tool) => tool.type === "tool_calling")
            .map((tool) => (
              <Tool
                key={tool.tool_call_id}
                tool={tool}
                now={now}
                ticking={live}
              />
            ))}
        </ul>
      )}
      {!user && showPersistence && (
        <p
          className={`mt-3 text-xs ${message.persistence_error ? "text-amber-600" : "text-muted"}`}
        >
          {message.persistence_error
            ? `Save pending · ${message.persistence_error}`
            : message.saved
              ? "Saved to Stream"
              : "Syncing to Stream…"}
          {working && !live
            ? " · Live updates disconnected; timers paused"
            : ""}
        </p>
      )}
    </article>
  );
}
