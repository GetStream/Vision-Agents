"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button, Empty, Failure, Field, inputStyle } from "@/components/ui";
import {
  router,
  type AgentConfig,
  type Skill,
  type SkillRequest,
} from "@/lib/router";

/**
 * The skills every agent gets without defining them. They are named by a config the same
 * way its own are, so they appear here as something to switch on rather than to write.
 */
const builtin: { name: string; description: string }[] = [
  {
    name: "think",
    description: "a question that needs careful reasoning, arithmetic or several steps",
  },
  {
    name: "recall",
    description: "something established earlier in a long conversation",
  },
  {
    name: "explain",
    description: "a detailed explanation the caller has actually asked to hear",
  },
];

const blank = (configID: string): SkillRequest => ({
  config_id: configID,
  name: "",
  description: "",
  instructions: "",
});

/**
 * AgentSkills is what this agent may hand to its subagent.
 *
 * A skill belongs to one agent, so writing one here cannot change what another agent
 * means by the same word. The config's own list of names is kept in step as skills are
 * added and removed, since a row nothing names is a row the model is never offered.
 */
export function AgentSkills({ config }: { config: AgentConfig }) {
  const client = useQueryClient();
  const [editing, setEditing] = useState<SkillRequest | null>(null);
  const [editingID, setEditingID] = useState<string | null>(null);

  const skills = useQuery({
    queryKey: ["skills", config.id],
    queryFn: () => router.skills(config.id),
  });

  const named = new Set(config.skills ?? []);

  const refresh = () => {
    client.invalidateQueries({ queryKey: ["skills", config.id] });
    client.invalidateQueries({ queryKey: ["config", config.id] });
    client.invalidateQueries({ queryKey: ["configs"] });
  };

  /**
   * Writes the config's skill names, leaving everything else on it as it was saved.
   *
   * The saved config is what is written back rather than whatever the form is holding, so
   * switching a skill on never quietly saves an edit somebody was still making elsewhere
   * on the page.
   */
  const rename = useMutation({
    mutationFn: (names: string[]) =>
      router.updateConfig(config.id, { ...config, skills: names }),
    onSuccess: refresh,
  });

  const save = useMutation({
    mutationFn: async (skill: SkillRequest) => {
      const stored = editingID
        ? await router.updateSkill(editingID, skill)
        : await router.createSkill(skill);
      if (!named.has(stored.name)) {
        await router.updateConfig(config.id, {
          ...config,
          skills: [...(config.skills ?? []), stored.name],
        });
      }
      return stored;
    },
    onSuccess: () => {
      refresh();
      setEditing(null);
      setEditingID(null);
    },
  });

  const remove = useMutation({
    mutationFn: async (skill: Skill) => {
      await router.deleteSkill(skill.id);
      await router.updateConfig(config.id, {
        ...config,
        skills: (config.skills ?? []).filter((name) => name !== skill.name),
      });
    },
    onSuccess: refresh,
  });

  const toggle = (name: string) =>
    rename.mutate(
      named.has(name)
        ? (config.skills ?? []).filter((entry) => entry !== name)
        : [...(config.skills ?? []), name],
    );

  const edit = (skill: Skill) => {
    setEditingID(skill.id);
    setEditing({
      config_id: config.id,
      name: skill.name,
      description: skill.description,
      instructions: skill.instructions,
      deadline_ms: skill.deadline_ms,
    });
  };

  return (
    <div className="px-4 py-4">
      {config.subagent ? null : (
        <p className="mb-4 rounded-lg border border-line px-3 py-2 text-xs text-muted">
          This agent has no subagent, so it answers everything itself and no skill is ever
          offered. Name one under Model to make these mean something.
        </p>
      )}

      <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
        This agent&apos;s skills
      </h3>

      {skills.isError ? <Failure error={skills.error} /> : null}

      <ul className="mt-2">
        {skills.data?.length === 0 ? (
          <Empty>
            No skills of its own yet. The built-in three below are still available.
          </Empty>
        ) : null}
        {(skills.data ?? []).map((skill) => (
          <li
            key={skill.id}
            className="flex items-start gap-4 border-b border-line py-3 first:border-t"
          >
            <div className="min-w-0 flex-1">
              <div className="text-sm font-medium">{skill.name}</div>
              <p className="mt-0.5 text-sm text-muted">{skill.description}</p>
              <p className="mt-1 line-clamp-2 text-xs text-muted">
                {skill.instructions}
              </p>
            </div>
            <div className="flex shrink-0 gap-2">
              <Button variant="quiet" onClick={() => edit(skill)}>
                Edit
              </Button>
              <Button
                variant="danger"
                onClick={() => remove.mutate(skill)}
                disabled={remove.isPending}
              >
                Remove
              </Button>
            </div>
          </li>
        ))}
      </ul>

      {editing ? (
        <SkillForm
          skill={editing}
          onChange={setEditing}
          onCancel={() => {
            setEditing(null);
            setEditingID(null);
          }}
          onSave={() => save.mutate(editing)}
          saving={save.isPending}
          error={save.error}
        />
      ) : (
        <div className="mt-3">
          <Button onClick={() => setEditing(blank(config.id))}>Add skill</Button>
        </div>
      )}

      <h3 className="mt-6 text-xs font-medium uppercase tracking-wide text-muted">
        Built in
      </h3>
      <ul className="mt-2">
        {builtin.map((skill) => (
          <li
            key={skill.name}
            className="flex items-center gap-3 border-b border-line py-2 first:border-t"
          >
            <input
              type="checkbox"
              id={`builtin-${skill.name}`}
              checked={named.has(skill.name)}
              disabled={rename.isPending}
              onChange={() => toggle(skill.name)}
            />
            <label htmlFor={`builtin-${skill.name}`} className="min-w-0 flex-1">
              <span className="text-sm font-medium">{skill.name}</span>
              <span className="ml-2 text-sm text-muted">{skill.description}</span>
            </label>
          </li>
        ))}
      </ul>
      {rename.error ? <Failure error={rename.error} /> : null}
    </div>
  );
}

function SkillForm({
  skill,
  onChange,
  onCancel,
  onSave,
  saving,
  error,
}: {
  skill: SkillRequest;
  onChange: (skill: SkillRequest) => void;
  onCancel: () => void;
  onSave: () => void;
  saving: boolean;
  error: unknown;
}) {
  const set = (patch: Partial<SkillRequest>) => onChange({ ...skill, ...patch });

  return (
    <form
      className="mt-3 space-y-4 rounded-lg border border-line px-3 py-3"
      onSubmit={(event) => {
        event.preventDefault();
        onSave();
      }}
    >
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Name" hint="What the model writes to hand work over.">
          <input
            className={inputStyle}
            value={skill.name}
            onChange={(event) => set({ name: event.target.value })}
            required
          />
        </Field>
        <Field label="Deadline" hint="Milliseconds. Empty leaves the default.">
          <input
            className={inputStyle}
            type="number"
            min={0}
            value={skill.deadline_ms ?? ""}
            onChange={(event) =>
              set({
                deadline_ms: event.target.value
                  ? Number(event.target.value)
                  : undefined,
              })
            }
          />
        </Field>
      </div>

      <Field
        label="Description"
        hint="The one line the fast model sees, and the whole of how it decides when to use this."
      >
        <input
          className={inputStyle}
          value={skill.description}
          onChange={(event) => set({ description: event.target.value })}
          required
        />
      </Field>

      <Field
        label="Instructions"
        hint="The full prompt, which only the subagent sees."
      >
        <textarea
          className={`${inputStyle} h-32 resize-y`}
          value={skill.instructions}
          onChange={(event) => set({ instructions: event.target.value })}
          required
        />
      </Field>

      {error ? <Failure error={error} /> : null}

      <div className="flex gap-2">
        <Button type="submit" disabled={saving || !skill.name}>
          {saving ? "Saving…" : "Save skill"}
        </Button>
        <Button variant="quiet" onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </form>
  );
}
