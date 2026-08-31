"use client";

import { clock, Empty, Panel } from "@/components/ui";
import type { TranscriptMessage } from "@/lib/router";

/**
 * Transcript is the conversation as it was written down, from the Stream Chat channel the
 * agent logs into rather than from this page's socket.
 *
 * Reading the channel rather than the frames means a page opened halfway through a call
 * shows the whole of it, and shows the same thing a finished call does. What is happening
 * this second is above, under the orb.
 */
export function Transcript({
  running,
  stored,
}: {
  running: boolean;
  stored: TranscriptMessage[];
}) {
  // The panel deliberately does not follow new lines. It scrolls on its own, and jumping
  // to the newest message takes the one being read out from under the reader.
  return (
    <Panel
      title={running ? "What is being said" : "What was said"}
      aside={<span className="text-xs text-muted">from the chat channel</span>}
    >
      {stored.length === 0 ? (
        <Empty>
          {running
            ? "Nothing has reached the channel yet."
            : "Nothing was written down for this call."}
        </Empty>
      ) : (
        <div className="max-h-[28rem] space-y-3 overflow-y-auto px-4 py-3">
          {stored.map((message, index) => (
            <div key={`${message.created_at}-${index}`} className="text-sm">
              <div className="mb-0.5 flex items-baseline gap-2">
                <span
                  className={`text-xs font-medium ${
                    message.speaker === "agent"
                      ? "text-emerald-600"
                      : "text-sky-600"
                  }`}
                >
                  {message.speaker}
                </span>
                <span className="font-mono text-xs tabular-nums text-muted">
                  {clock(message.created_at)}
                </span>
              </div>
              <p className="leading-relaxed">{message.text}</p>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
