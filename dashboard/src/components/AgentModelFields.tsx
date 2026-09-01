"use client";

import { useQuery } from "@tanstack/react-query";
import { useId } from "react";

import { Field, inputStyle } from "@/components/ui";
import { router, type AgentConfigRequest, type AgentMode } from "@/lib/router";

/**
 * AgentModelFields is what an agent is and who answers for it.
 *
 * The speech targets are shown only for a voice agent: a text agent transcribes nothing
 * and speaks nothing, so offering it a transcriber would be offering a setting that does
 * not apply rather than one it has not filled in.
 */
export function AgentModelFields({
  config,
  onChange,
  compact = false,
}: {
  config: AgentConfigRequest;
  onChange: (config: AgentConfigRequest) => void;
  compact?: boolean;
}) {
  const set = (patch: Partial<AgentConfigRequest>) =>
    onChange({ ...config, ...patch });

  const spoken = (config.mode ?? "voice") === "voice";

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Name">
          <input
            className={inputStyle}
            value={config.name}
            onChange={(event) => set({ name: event.target.value })}
            required
          />
        </Field>
        <Field
          label="Kind"
          hint={
            spoken
              ? "Joins a call, listens and speaks."
              : "Holds the conversation in writing. No call, no transcriber, no voice."
          }
        >
          <select
            className={inputStyle}
            value={config.mode ?? "voice"}
            onChange={(event) => set({ mode: event.target.value as AgentMode })}
          >
            <option value="voice">Voice agent</option>
            <option value="text">Text only</option>
          </select>
        </Field>
      </div>

      <Field label="Instructions" hint="What the agent is told to be.">
        <textarea
          className={`${inputStyle} h-28 resize-y`}
          value={config.instructions ?? ""}
          onChange={(event) => set({ instructions: event.target.value })}
        />
      </Field>

      {spoken ? (
        <Field
          label="Greeting"
          hint="Said without asking a model, so it costs nothing."
        >
          <input
            className={inputStyle}
            value={config.greeting ?? ""}
            onChange={(event) => set({ greeting: event.target.value })}
          />
        </Field>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Target
          modality="llm"
          label="Model"
          hint="Holds the conversation."
          placeholder="llm-fast"
          value={config.llm}
          onChange={(llm) => set({ llm })}
        />
        <Target
          modality="llm"
          label="Subagent"
          hint="Does the thinking. Empty means no skills."
          placeholder="llm-quality"
          value={config.subagent}
          onChange={(subagent) => set({ subagent })}
        />
        <Target
          modality="search"
          label="Search"
          hint="How it finds out what is true today."
          placeholder="search-fast"
          value={config.search}
          onChange={(search) => set({ search })}
        />
      </div>

      {spoken ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Target
            modality="stt"
            label="Transcriber"
            hint="Turns what it hears into words."
            placeholder="en-low-latency"
            value={config.stt}
            onChange={(stt) => set({ stt })}
          />
          <Target
            modality="tts"
            label="Voice provider"
            hint="Speaks the replies."
            placeholder="en-low-latency"
            value={config.tts}
            onChange={(tts) => set({ tts })}
          />
          <Field label="Voice" hint="A voice id the provider knows, or one of your own.">
            <input
              className={inputStyle}
              value={config.voice ?? ""}
              onChange={(event) => set({ voice: event.target.value })}
            />
          </Field>
        </div>
      ) : null}

      {compact || !spoken ? null : (
        <Field
          label="Keyterms"
          hint="Comma separated. Words the transcriber would otherwise get wrong."
        >
          <input
            className={inputStyle}
            value={(config.keyterms ?? []).join(", ")}
            onChange={(event) =>
              set({
                keyterms: event.target.value
                  .split(",")
                  .map((term) => term.trim())
                  .filter(Boolean),
              })
            }
          />
        </Field>
      )}
    </div>
  );
}

/**
 * Target is a routing target: a provider/model, or one of the capability shortcuts.
 *
 * The catalog is offered as suggestions rather than as a closed list, because a shortcut
 * such as llm-fast is a valid target and is not a row in it.
 */
export function Target({
  modality,
  label,
  hint,
  placeholder,
  value,
  onChange,
}: {
  modality: string;
  label: string;
  hint: string;
  placeholder: string;
  value: string | undefined;
  onChange: (value: string) => void;
}) {
  const listID = useId();
  const providers = useQuery({
    queryKey: ["providers", modality],
    queryFn: () => router.providers(modality),
    staleTime: 5 * 60_000,
  });

  return (
    <Field label={label} hint={hint}>
      <input
        className={inputStyle}
        list={listID}
        placeholder={placeholder}
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value)}
      />
      <datalist id={listID}>
        {(providers.data ?? []).map((provider) => (
          <option
            key={`${provider.provider}/${provider.model}`}
            value={`${provider.provider}/${provider.model}`}
          />
        ))}
      </datalist>
    </Field>
  );
}
