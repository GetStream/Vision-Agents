"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";

import { Button, Empty, Failure, PageHeading, Panel } from "@/components/ui";
import { router, type AgentConfig } from "@/lib/router";

export default function Agents() {
  const client = useQueryClient();

  const configs = useQuery({ queryKey: ["configs"], queryFn: router.configs });

  const remove = useMutation({
    mutationFn: router.deleteConfig,
    onSuccess: () => client.invalidateQueries({ queryKey: ["configs"] }),
  });

  return (
    <>
      <PageHeading
        title="Agents"
        description="What each agent is told to be, which models answer for it, and what it may hand over."
        action={
          <Link
            href="/agents/new"
            className="rounded-lg bg-foreground px-3 py-1.5 text-sm font-medium text-background transition hover:opacity-90"
          >
            New agent
          </Link>
        }
      />

      {configs.isError ? <Failure error={configs.error} /> : null}

      <Panel title="Configured agents">
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
                <Link
                  href={`/agents/${config.id}`}
                  className="font-medium hover:underline"
                >
                  {config.name}
                </Link>
                <p className="mt-0.5 line-clamp-2 text-sm text-muted">
                  {config.instructions || "No instructions."}
                </p>
                <Targets config={config} />
              </div>
              <div className="flex shrink-0 gap-2">
                <Link
                  href={`/agents/${config.id}`}
                  className="rounded-lg border border-line px-3 py-1.5 text-sm font-medium transition hover:bg-line/40"
                >
                  Edit
                </Link>
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

function Targets({ config }: { config: AgentConfig }) {
  const labelled: [string, string | undefined][] = [
    ["mode", config.mode],
    ["llm", config.llm],
    ["subagent", config.subagent],
    ["stt", config.stt],
    ["tts", config.tts],
    ["voice", config.voice],
    ["knowledge", config.knowledge_namespace],
  ];

  return (
    <div className="mt-1.5 flex flex-wrap gap-1.5 text-xs text-muted">
      {labelled
        .filter(([, value]) => value)
        .map(([label, value]) => (
          <span key={label} className="rounded-md border border-line px-1.5 py-0.5">
            {label}: {value}
          </span>
        ))}
      {(config.skills ?? []).map((skill) => (
        <span key={skill} className="rounded-md border border-line px-1.5 py-0.5">
          {skill}
        </span>
      ))}
    </div>
  );
}
