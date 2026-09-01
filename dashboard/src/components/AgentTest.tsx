"use client";

import { useMutation } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { AgentChat } from "@/components/AgentChat";
import { Button, Failure } from "@/components/ui";
import { router, type AgentConfig } from "@/lib/router";

/**
 * The chat an agent is tested in is the same one every time, so what was said last time is
 * still there. A voice call is not: each one is its own call, because two people cannot
 * join the same finished call and hear anything.
 */
const chatAgentID = (configID: string) => `chat-${configID}`;

/**
 * AgentTest is the two ways of talking to what has been configured.
 *
 * Both start a session from this config, so what is being tested is the agent as saved
 * rather than as it is being edited: an unsaved change is not part of the agent yet.
 */
export function AgentTest({ config }: { config: AgentConfig }) {
  const navigate = useRouter();
  const [chatting, setChatting] = useState<string | null>(null);

  const startChat = useMutation({
    mutationFn: () =>
      router.createSession({
        config_id: config.id,
        text: true,
        agent_id: chatAgentID(config.id),
      }),
    onSuccess: (session) => setChatting(session.id),
  });

  const endChat = useMutation({
    mutationFn: (sessionID: string) => router.closeSession(sessionID),
    onSuccess: () => setChatting(null),
  });

  // A voice test is a call like any other, so it is handed to the call page rather than
  // rebuilt here: joining, the transcript and what the agent decided are already there.
  const startCall = useMutation({
    mutationFn: () =>
      router.createSession({
        config_id: config.id,
        call_id: `test-${config.id}-${Date.now().toString(36)}`,
      }),
    onSuccess: (session) => navigate.push(`/calls/${session.id}`),
  });

  const spoken = config.mode === "voice";

  return (
    <div className="space-y-4 px-4 py-4">
      <div className="flex flex-wrap items-center gap-2">
        {chatting ? (
          <Button
            variant="quiet"
            onClick={() => endChat.mutate(chatting)}
            disabled={endChat.isPending}
          >
            {endChat.isPending ? "Ending…" : "End chat"}
          </Button>
        ) : (
          <Button
            onClick={() => startChat.mutate()}
            disabled={startChat.isPending}
          >
            {startChat.isPending ? "Starting…" : "Chat with my agent"}
          </Button>
        )}

        <Button
          variant="quiet"
          onClick={() => startCall.mutate()}
          disabled={!spoken || startCall.isPending}
          title={
            spoken ? undefined : "A text agent has no voice to talk to."
          }
        >
          {startCall.isPending ? "Starting…" : "Talk to my agent"}
        </Button>

        {spoken ? null : (
          <span className="text-xs text-muted">
            This is a text agent, so there is nothing to call.
          </span>
        )}
      </div>

      {startChat.error ? <Failure error={startChat.error} /> : null}
      {startCall.error ? <Failure error={startCall.error} /> : null}

      {chatting ? (
        <AgentChat agentID={chatAgentID(config.id)} sessionID={chatting} />
      ) : (
        <p className="text-sm text-muted">
          Start a chat to talk to this agent in writing, or call it to hear it. Either one
          runs the agent as it is saved.
        </p>
      )}
    </div>
  );
}
