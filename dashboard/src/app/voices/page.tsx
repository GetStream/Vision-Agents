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
import { router, type Voice } from "@/lib/router";

/** How a provider is getting on with a voice, at a glance. */
const bindingStyles: Record<string, string> = {
  ready: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
  pending: "bg-amber-500/10 text-amber-600 border-amber-500/20",
  failed: "bg-red-500/10 text-red-600 border-red-500/20",
};

export default function Voices() {
  const client = useQueryClient();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const voices = useQuery({ queryKey: ["voices"], queryFn: router.voices });

  const refresh = () => client.invalidateQueries({ queryKey: ["voices"] });

  const create = useMutation({
    mutationFn: () => router.createVoice({ name, description }),
    onSuccess: () => {
      setName("");
      setDescription("");
      refresh();
    },
  });

  const prepare = useMutation({
    // No providers named means every provider this deployment can clone with, which is
    // what somebody clicking "prepare" is asking for.
    mutationFn: (id: string) => router.prepareVoice(id, []),
    onSuccess: refresh,
  });

  const remove = useMutation({
    mutationFn: router.deleteVoice,
    onSuccess: refresh,
  });

  return (
    <>
      <PageHeading
        title="Voices"
        description="Voices of your own. A config names one of these rather than a provider's id, so the router can still fail over mid-call."
      />

      <Panel title="Add a voice">
        <form
          className="flex flex-wrap items-end gap-3 px-4 py-4"
          onSubmit={(event) => {
            event.preventDefault();
            create.mutate();
          }}
        >
          <div className="w-48">
            <Field label="Name">
              <input
                className={inputStyle}
                value={name}
                onChange={(event) => setName(event.target.value)}
                required
              />
            </Field>
          </div>
          <div className="min-w-64 flex-1">
            <Field label="Description">
              <input
                className={inputStyle}
                value={description}
                onChange={(event) => setDescription(event.target.value)}
              />
            </Field>
          </div>
          <Button type="submit" disabled={!name || create.isPending}>
            {create.isPending ? "Adding…" : "Add"}
          </Button>
        </form>
        {create.error ? (
          <div className="px-4 pb-4">
            <Failure error={create.error} />
          </div>
        ) : null}
      </Panel>

      {voices.isError ? (
        <div className="mt-6">
          <Failure error={voices.error} />
        </div>
      ) : null}

      <Panel title="Your voices" className="mt-6">
        {voices.data?.length === 0 ? (
          <Empty>
            No voices yet. Without one an agent speaks in the provider&apos;s own.
          </Empty>
        ) : null}
        <ul>
          {(voices.data ?? []).map((voice) => (
            <VoiceRow
              key={voice.id}
              voice={voice}
              onPrepare={() => prepare.mutate(voice.id)}
              onDelete={() => remove.mutate(voice.id)}
              busy={prepare.isPending || remove.isPending}
            />
          ))}
        </ul>
      </Panel>
    </>
  );
}

function VoiceRow({
  voice,
  onPrepare,
  onDelete,
  busy,
}: {
  voice: Voice;
  onPrepare: () => void;
  onDelete: () => void;
  busy: boolean;
}) {
  const samples = voice.samples ?? [];
  const bindings = voice.bindings ?? [];

  return (
    <li className="flex items-start gap-4 border-b border-line px-4 py-3 last:border-0">
      <div className="min-w-0 flex-1">
        <div className="font-medium">{voice.name}</div>
        {voice.description ? (
          <p className="mt-0.5 text-sm text-muted">{voice.description}</p>
        ) : null}
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-muted">
            {samples.length} recording{samples.length === 1 ? "" : "s"}
          </span>
          {bindings.map((binding) => (
            <span
              key={binding.provider}
              title={binding.error ?? undefined}
              className={`rounded-md border px-1.5 py-0.5 ${
                bindingStyles[binding.state] ?? bindingStyles.pending
              }`}
            >
              {binding.provider}: {binding.state}
            </span>
          ))}
          {bindings.length === 0 ? (
            <span className="text-muted">
              not prepared, so nothing can speak in it yet
            </span>
          ) : null}
        </div>
      </div>
      <div className="flex shrink-0 gap-2">
        <Button
          variant="quiet"
          onClick={onPrepare}
          disabled={busy || samples.length === 0}
        >
          Prepare
        </Button>
        <Button variant="danger" onClick={onDelete} disabled={busy}>
          Delete
        </Button>
      </div>
    </li>
  );
}
