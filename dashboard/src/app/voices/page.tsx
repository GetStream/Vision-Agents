"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";

import {
  Button,
  Empty,
  Failure,
  Field,
  inputStyle,
  PageHeading,
  Panel,
} from "@/components/ui";
import { router, type Voice, type VoiceSampleRequest } from "@/lib/router";

/** How a provider is getting on with a voice, at a glance. */
const bindingStyles: Record<string, string> = {
  ready: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20",
  pending: "bg-amber-500/10 text-amber-600 border-amber-500/20",
  failed: "bg-red-500/10 text-red-600 border-red-500/20",
};

const audioAccept =
  "audio/*,.wav,.mp3,.m4a,.ogg,.webm,.flac,audio/wav,audio/mpeg,audio/mp4";

export default function Voices() {
  const client = useQueryClient();
  const picker = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [transcript, setTranscript] = useState("");
  const [recording, setRecording] = useState<File | null>(null);

  const voices = useQuery({ queryKey: ["voices"], queryFn: router.voices });

  const refresh = () => client.invalidateQueries({ queryKey: ["voices"] });

  const create = useMutation({
    mutationFn: async () => {
      const voice = await router.createVoice({ name, description });
      if (recording) {
        return router.addVoiceSample(voice.id, await sampleOf(recording, transcript));
      }
      return voice;
    },
    onSuccess: () => {
      setName("");
      setDescription("");
      setTranscript("");
      setRecording(null);
      if (picker.current) {
        picker.current.value = "";
      }
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
          className="space-y-4 px-4 py-4"
          onSubmit={(event) => {
            event.preventDefault();
            create.mutate();
          }}
        >
          <div className="flex flex-wrap items-end gap-3">
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
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <Field
              label="Recording"
              hint="A minute of clean speech is plenty. The extension is how a provider knows what it was given."
            >
              <input
                ref={picker}
                className={inputStyle}
                type="file"
                accept={audioAccept}
                onChange={(event) =>
                  setRecording(event.target.files?.[0] ?? null)
                }
              />
            </Field>
            <Field
              label="Transcript"
              hint="What is said in the recording. Optional, and the providers that use one clone more faithfully with it."
            >
              <input
                className={inputStyle}
                value={transcript}
                onChange={(event) => setTranscript(event.target.value)}
                disabled={!recording}
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
              onAdded={refresh}
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
  onAdded,
  busy,
}: {
  voice: Voice;
  onPrepare: () => void;
  onDelete: () => void;
  onAdded: () => void;
  busy: boolean;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const samples = voice.samples ?? [];
  const bindings = voice.bindings ?? [];

  const upload = useMutation({
    mutationFn: async (file: File) =>
      router.addVoiceSample(voice.id, await sampleOf(file)),
    onSuccess: onAdded,
  });

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
          {samples.map((sample) => (
            <span
              key={sample.id}
              className="rounded-md border border-line px-1.5 py-0.5 text-muted"
            >
              {sample.filename || "recording"}
            </span>
          ))}
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
        {upload.error ? (
          <div className="mt-2">
            <Failure error={upload.error} />
          </div>
        ) : null}
      </div>
      <div className="flex shrink-0 gap-2">
        <input
          ref={picker}
          type="file"
          accept={audioAccept}
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              upload.mutate(file);
            }
            event.target.value = "";
          }}
        />
        <Button
          variant="quiet"
          onClick={() => picker.current?.click()}
          disabled={busy || upload.isPending}
        >
          {upload.isPending ? "Uploading…" : "Add recording"}
        </Button>
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

/** sampleOf is the recording as the router wants it: raw bytes, named, as base64. */
async function sampleOf(
  file: File,
  transcript?: string,
): Promise<VoiceSampleRequest> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  const sample: VoiceSampleRequest = {
    audio: btoa(binary),
    filename: file.name,
    content_type: file.type || undefined,
  };
  const said = transcript?.trim();
  if (said) {
    sample.transcript = said;
  }
  return sample;
}
