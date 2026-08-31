"use client";

import { useState } from "react";

import { Panel } from "@/components/ui";
import type { Call } from "@/lib/router";

/**
 * The four models a call ran on, in the order a turn passes through them: heard,
 * answered, spoken, and whatever the answer was too hard for.
 */
const modalities = [
  {
    label: "Transcribe",
    of: (call: Call) => call.stt,
    hint: "Speech to text",
  },
  {
    label: "Answer",
    of: (call: Call) => call.llm,
    hint: "Holds the conversation",
  },
  {
    label: "Speak",
    of: (call: Call) => call.tts,
    hint: "Text to speech",
  },
  {
    label: "Think",
    of: (call: Call) => call.subagent,
    hint: "Runs delegated work",
  },
] as const;

/**
 * Target splits a `provider/model` name so the model is what stands out. A capability
 * shortcut such as `llm-fast` has no slash and is shown as it was asked for, because that
 * is what it is: a request that several models can answer.
 */
function Target({ value }: { value?: string }) {
  if (!value) {
    return <span className="text-sm text-muted">—</span>;
  }

  const slash = value.indexOf("/");
  if (slash < 0) {
    return <span className="text-sm font-medium">{value}</span>;
  }
  return (
    <span className="text-sm font-medium break-all">
      <span className="text-muted">{value.slice(0, slash + 1)}</span>
      {value.slice(slash + 1)}
    </span>
  );
}

export function Pipeline({
  call,
  className = "",
}: {
  call: Call;
  className?: string;
}) {
  const [showing, setShowing] = useState(false);

  const skills = call.skills ?? [];
  const instructions = call.instructions ?? "";

  return (
    <Panel
      title="Pipeline"
      className={className}
      aside={
        <span className="text-xs text-muted">
          What was asked for, not what each turn resolved to
        </span>
      }
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-3 px-4 py-3 lg:grid-cols-4">
        {modalities.map((modality) => (
          <div key={modality.label}>
            <div className="text-xs uppercase tracking-wide text-muted">
              {modality.label}
            </div>
            <div className="mt-1">
              <Target value={modality.of(call)} />
            </div>
            <div className="mt-0.5 text-xs text-muted">{modality.hint}</div>
          </div>
        ))}
      </div>

      <div className="border-t border-line px-4 py-3">
        <div className="text-xs uppercase tracking-wide text-muted">Skills</div>
        {skills.length ? (
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {skills.map((skill) => (
              <span
                key={skill}
                className="rounded-md border border-line px-1.5 py-0.5 text-xs"
              >
                {skill}
              </span>
            ))}
          </div>
        ) : (
          <p className="mt-1 text-sm text-muted">
            {call.subagent
              ? "None were offered."
              : "Nothing was delegated, so none were offered."}
          </p>
        )}
      </div>

      <div className="border-t border-line px-4 py-3">
        {instructions ? (
          <>
            <button
              type="button"
              onClick={() => setShowing((open) => !open)}
              aria-expanded={showing}
              className="flex w-full items-center justify-between gap-3 text-left"
            >
              <span className="text-xs uppercase tracking-wide text-muted">
                Instructions
              </span>
              <span className="text-xs text-muted">
                {showing ? "Hide" : "Show"}
              </span>
            </button>
            {showing ? (
              <pre className="mt-2 max-h-80 overflow-y-auto whitespace-pre-wrap font-sans text-sm leading-relaxed">
                {instructions}
              </pre>
            ) : (
              <p className="mt-1 line-clamp-1 text-sm text-muted">
                {instructions}
              </p>
            )}
          </>
        ) : (
          <>
            <div className="text-xs uppercase tracking-wide text-muted">
              Instructions
            </div>
            <p className="mt-1 text-sm text-muted">
              The agent was told nothing in particular.
            </p>
          </>
        )}
      </div>
    </Panel>
  );
}
