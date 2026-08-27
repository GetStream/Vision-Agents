"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  Button,
  Empty,
  Failure,
  Field,
  inputStyle,
  PageHeading,
  Panel,
} from "@/components/ui";
import { router, type AgentConfig, type AgentConfigRequest } from "@/lib/router";

/** A config nobody has filled in yet. Every field but the name is allowed to stay empty. */
const blank: AgentConfigRequest = { name: "" };

export default function Agents() {
  const client = useQueryClient();
  const [editing, setEditing] = useState<AgentConfigRequest | null>(null);
  const [editingID, setEditingID] = useState<string | null>(null);

  const configs = useQuery({ queryKey: ["configs"], queryFn: router.configs });

  const save = useMutation({
    mutationFn: (config: AgentConfigRequest) =>
      editingID
        ? router.updateConfig(editingID, config)
        : router.createConfig(config),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["configs"] });
      setEditing(null);
      setEditingID(null);
    },
  });

  const remove = useMutation({
    mutationFn: router.deleteConfig,
    onSuccess: () => client.invalidateQueries({ queryKey: ["configs"] }),
  });

  const edit = (config: AgentConfig) => {
    setEditingID(config.id);
    setEditing({
      name: config.name,
      stt: config.stt,
      tts: config.tts,
      voice: config.voice,
      llm: config.llm,
      subagent: config.subagent,
      instructions: config.instructions,
      greeting: config.greeting,
      skills: config.skills,
      keyterms: config.keyterms,
      knowledge_namespace: config.knowledge_namespace,
    });
  };

  return (
    <>
      <PageHeading
        title="Agents"
        description="What each agent is told to be, which models answer for it, and what it may hand over."
        action={
          <Button
            onClick={() => {
              setEditingID(null);
              setEditing(blank);
            }}
          >
            New agent
          </Button>
        }
      />

      {configs.isError ? <Failure error={configs.error} /> : null}

      {editing ? (
        <ConfigForm
          config={editing}
          onChange={setEditing}
          onCancel={() => {
            setEditing(null);
            setEditingID(null);
          }}
          onSave={() => save.mutate(editing)}
          saving={save.isPending}
          error={save.error}
        />
      ) : null}

      <Panel title="Configured agents" className="mt-6">
        {configs.data?.length === 0 ? (
          <Empty>No agents yet. A session without one takes the defaults.</Empty>
        ) : null}
        <ul>
          {(configs.data ?? []).map((config) => (
            <li
              key={config.id}
              className="flex items-start gap-4 border-b border-line px-4 py-3 last:border-0"
            >
              <div className="min-w-0 flex-1">
                <div className="font-medium">{config.name}</div>
                <p className="mt-0.5 line-clamp-2 text-sm text-muted">
                  {config.instructions || "No instructions."}
                </p>
                <div className="mt-1.5 flex flex-wrap gap-1.5 text-xs text-muted">
                  {[
                    ["llm", config.llm],
                    ["subagent", config.subagent],
                    ["stt", config.stt],
                    ["tts", config.tts],
                    ["voice", config.voice],
                    ["knowledge", config.knowledge_namespace],
                  ]
                    .filter(([, value]) => value)
                    .map(([label, value]) => (
                      <span
                        key={label}
                        className="rounded-md border border-line px-1.5 py-0.5"
                      >
                        {label}: {value}
                      </span>
                    ))}
                  {(config.skills ?? []).map((skill) => (
                    <span
                      key={skill}
                      className="rounded-md border border-line px-1.5 py-0.5"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex shrink-0 gap-2">
                <Button variant="quiet" onClick={() => edit(config)}>
                  Edit
                </Button>
                <Button
                  variant="danger"
                  onClick={() => remove.mutate(config.id)}
                  disabled={remove.isPending}
                >
                  Delete
                </Button>
              </div>
            </li>
          ))}
        </ul>
      </Panel>
    </>
  );
}

function ConfigForm({
  config,
  onChange,
  onCancel,
  onSave,
  saving,
  error,
}: {
  config: AgentConfigRequest;
  onChange: (config: AgentConfigRequest) => void;
  onCancel: () => void;
  onSave: () => void;
  saving: boolean;
  error: unknown;
}) {
  const set = (patch: Partial<AgentConfigRequest>) =>
    onChange({ ...config, ...patch });

  const list = (value: string) =>
    value
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean);

  return (
    <Panel title={config.name ? `Editing ${config.name}` : "New agent"}>
      <form
        className="space-y-4 px-4 py-4"
        onSubmit={(event) => {
          event.preventDefault();
          onSave();
        }}
      >
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
            label="Knowledge namespace"
            hint="What the agent may look things up in."
          >
            <input
              className={inputStyle}
              value={config.knowledge_namespace ?? ""}
              onChange={(event) =>
                set({ knowledge_namespace: event.target.value })
              }
            />
          </Field>
        </div>

        <Field label="Instructions" hint="What the agent is told to be.">
          <textarea
            className={`${inputStyle} h-28 resize-y`}
            value={config.instructions ?? ""}
            onChange={(event) => set({ instructions: event.target.value })}
          />
        </Field>

        <Field label="Greeting" hint="Said without asking a model, so it costs nothing.">
          <input
            className={inputStyle}
            value={config.greeting ?? ""}
            onChange={(event) => set({ greeting: event.target.value })}
          />
        </Field>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {(
            [
              ["llm", "Model", "llm-fast"],
              ["subagent", "Subagent", "llm-quality"],
              ["stt", "Transcriber", "en-low-latency"],
              ["tts", "Voice provider", "en-low-latency"],
              ["voice", "Voice", ""],
            ] as const
          ).map(([key, label, placeholder]) => (
            <Field key={key} label={label}>
              <input
                className={inputStyle}
                placeholder={placeholder}
                value={config[key] ?? ""}
                onChange={(event) => set({ [key]: event.target.value })}
              />
            </Field>
          ))}
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field
            label="Skills"
            hint="Comma separated. Empty takes think, recall and explain."
          >
            <input
              className={inputStyle}
              value={(config.skills ?? []).join(", ")}
              onChange={(event) => set({ skills: list(event.target.value) })}
            />
          </Field>
          <Field
            label="Keyterms"
            hint="Words the transcriber would otherwise get wrong."
          >
            <input
              className={inputStyle}
              value={(config.keyterms ?? []).join(", ")}
              onChange={(event) => set({ keyterms: list(event.target.value) })}
            />
          </Field>
        </div>

        {error ? <Failure error={error} /> : null}

        <div className="flex gap-2">
          <Button type="submit" disabled={saving || !config.name}>
            {saving ? "Saving…" : "Save"}
          </Button>
          <Button variant="quiet" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </form>
    </Panel>
  );
}
