"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { AgentModelFields } from "@/components/AgentModelFields";
import { Button, Failure, Notice, PageHeading, Panel } from "@/components/ui";
import { router, type AgentConfigRequest } from "@/lib/router";

/** A config nobody has filled in yet. Every field but the name is allowed to stay empty. */
const blank: AgentConfigRequest = { name: "", mode: "voice" };

export default function NewAgent() {
  const client = useQueryClient();
  const navigate = useRouter();
  const [config, setConfig] = useState<AgentConfigRequest>(blank);

  // Skills, knowledge and the test chat all hang off an agent that exists, so this page
  // asks only for what makes one and hands over to the editor for the rest.
  const create = useMutation({
    mutationFn: () => router.createConfig(config),
    onSuccess: (created) => {
      client.invalidateQueries({ queryKey: ["configs"] });
      navigate.push(`/agents/${created.id}`);
    },
  });

  return (
    <>
      <PageHeading
        title="New agent"
        description="Name it and say who answers for it. Skills, knowledge and testing come next."
        action={
          <Link href="/agents" className="text-sm text-muted hover:underline">
            Back to agents
          </Link>
        }
      />

      <Notice className="mb-6">
        Prefer not to fill this in? Ask a coding agent instead:{" "}
        <span className="text-foreground">
          Build my voice AI with{" "}
          <a
            href="https://streamrtc.ai/skill.md"
            className="underline underline-offset-2"
            target="_blank"
            rel="noreferrer"
          >
            streamrtc.ai/skill.md
          </a>
        </span>
        .
      </Notice>

      <Panel>
        <form
          className="space-y-4 px-4 py-4"
          onSubmit={(event) => {
            event.preventDefault();
            create.mutate();
          }}
        >
          <AgentModelFields config={config} onChange={setConfig} compact />

          {create.error ? <Failure error={create.error} /> : null}

          <div className="flex gap-2">
            <Button type="submit" disabled={create.isPending || !config.name}>
              {create.isPending ? "Creating…" : "Create agent"}
            </Button>
          </div>
        </form>
      </Panel>
    </>
  );
}
