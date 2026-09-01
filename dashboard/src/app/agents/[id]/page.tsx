"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useState } from "react";

import { AgentConnections } from "@/components/AgentConnections";
import { AgentKnowledge } from "@/components/AgentKnowledge";
import { AgentModelFields } from "@/components/AgentModelFields";
import { AgentSkills } from "@/components/AgentSkills";
import { AgentTest } from "@/components/AgentTest";
import {
  Button,
  Failure,
  Notice,
  PageHeading,
  Section,
} from "@/components/ui";
import { router, type AgentConfig, type AgentConfigRequest } from "@/lib/router";

type SectionName = "model" | "skills" | "connections" | "test";

/** What a saved config is as something to edit: the id and timestamps are not fields. */
function draftOf(config: AgentConfig): AgentConfigRequest {
  return {
    name: config.name,
    mode: config.mode,
    stt: config.stt,
    tts: config.tts,
    voice: config.voice,
    llm: config.llm,
    subagent: config.subagent,
    search: config.search,
    instructions: config.instructions,
    greeting: config.greeting,
    skills: config.skills,
    plugins: config.plugins,
    keyterms: config.keyterms,
    knowledge_namespace: config.knowledge_namespace,
    tags: config.tags,
  };
}

export default function AgentPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const client = useQueryClient();
  const navigate = useRouter();

  const [open, setOpen] = useState<Set<SectionName>>(new Set(["model"]));
  const [edits, setEdits] = useState<AgentConfigRequest | null>(null);

  const config = useQuery({
    queryKey: ["config", id],
    queryFn: () => router.config(id),
  });

  // The form is what was saved until somebody types, and what they typed after that.
  // Deriving it rather than copying it is what stops a refetch overwriting a half-written
  // change, and what makes discarding one a matter of forgetting the edits.
  const saved = config.data;
  const draft = edits ?? (saved ? draftOf(saved) : null);

  const save = useMutation({
    mutationFn: (body: AgentConfigRequest) => router.updateConfig(id, body),
    onSuccess: (updated) => {
      client.setQueryData(["config", id], updated);
      client.invalidateQueries({ queryKey: ["configs"] });
      setEdits(null);
    },
  });

  const remove = useMutation({
    mutationFn: () => router.deleteConfig(id),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["configs"] });
      navigate.push("/agents");
    },
  });

  const toggle = (section: SectionName) =>
    setOpen((current) => {
      const next = new Set(current);
      if (!next.delete(section)) {
        next.add(section);
      }
      return next;
    });

  if (config.isError) {
    return <Failure error={config.error} />;
  }
  if (!saved || !draft) {
    return <p className="text-sm text-muted">Loading the agent…</p>;
  }

  return (
    <>
      <PageHeading
        title={saved.name}
        description={
          saved.mode === "text"
            ? "A text agent, held in writing."
            : "A voice agent, which joins a call and speaks."
        }
        action={
          <div className="flex items-center gap-2">
            <Link href="/agents" className="text-sm text-muted hover:underline">
              All agents
            </Link>
            <Button
              variant="danger"
              onClick={() => remove.mutate()}
              disabled={remove.isPending}
            >
              Delete
            </Button>
          </div>
        }
      />

      <Notice className="mb-6">
        This screen is one way to change an agent. The same config can be written from any
        of the server-side SDKs, or by asking a coding agent / the CLI to use{" "}
        <a
          href="https://streamrtc.ai/skill.md"
          className="text-foreground underline underline-offset-2"
          target="_blank"
          rel="noreferrer"
        >
          streamrtc.ai/skill.md
        </a>
        .
      </Notice>

      <div className="space-y-3">
        <Section
          title="Model"
          description="What the agent is, and who answers for it."
          open={open.has("model")}
          onToggle={() => toggle("model")}
        >
          <form
            className="space-y-4 px-4 py-4"
            onSubmit={(event) => {
              event.preventDefault();
              save.mutate(draft);
            }}
          >
            <AgentModelFields config={draft} onChange={setEdits} />

            {save.error ? <Failure error={save.error} /> : null}

            <div className="flex items-center gap-2">
              <Button
                type="submit"
                disabled={save.isPending || !draft.name || !edits}
              >
                {save.isPending ? "Saving…" : edits ? "Save" : "Saved"}
              </Button>
              <Button
                variant="quiet"
                onClick={() => setEdits(null)}
                disabled={!edits}
              >
                Reset
              </Button>
            </div>
          </form>
        </Section>

        <Section
          title="Skills & Knowledge"
          description="What it may hand to the slower model, and what it can look things up in."
          open={open.has("skills")}
          onToggle={() => toggle("skills")}
        >
          <AgentSkills config={saved} />
          <div className="border-t border-line">
            <AgentKnowledge
              config={saved}
              onNamespace={(namespace) =>
                save.mutate({ ...draftOf(saved), knowledge_namespace: namespace })
              }
            />
          </div>
        </Section>

        <Section
          title="Connections"
          description="What the agent can reach beyond what it was told."
          open={open.has("connections")}
          onToggle={() => toggle("connections")}
        >
          <AgentConnections config={saved} />
        </Section>

        <Section
          title="Test my agent"
          description="Chat with it, or call it and hear it."
          open={open.has("test")}
          onToggle={() => toggle("test")}
        >
          <AgentTest config={saved} />
        </Section>
      </div>
    </>
  );
}
