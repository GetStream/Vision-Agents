"use client";

import { useEffect, useState } from "react";
import { StreamChat, type Channel as ChatChannel } from "stream-chat";
import {
  Channel,
  Chat,
  MessageComposer,
  MessageList,
  Window,
  useMessageContext,
  useChannelStateContext,
} from "stream-chat-react";

import { ConversationMessage } from "@/components/ConversationMessage";
import type { ConversationMessage as ActivityMessage } from "@/lib/conversation";
import { Failure } from "@/components/ui";
import { router } from "@/lib/router";

import "stream-chat-react/css/index.css";

/**
 * AgentChat is the conversation itself, read out of the channel the agent writes it to.
 *
 * Both halves are the channel. What the agent says is written server-side as it is
 * generated, so a reply appears here a piece at a time without this component knowing
 * anything about the model. What the person types is an ordinary chat message, which
 * reaches the agent because the router is watching the channel for one.
 */
export function AgentChat({ agentID }: { agentID: string }) {
  const [client, setClient] = useState<StreamChat | null>(null);
  const [channel, setChannel] = useState<ChatChannel | null>(null);
  const [failure, setFailure] = useState<unknown>(null);

  useEffect(() => {
    let live = true;
    let connected: StreamChat | null = null;

    const open = async () => {
      const credentials = await router.chatToken({ agent_id: agentID });
      if (!live) {
        return;
      }

      const chat = new StreamChat(credentials.api_key);
      connected = chat;
      await chat.connectUser(
        { id: credentials.user_id, name: credentials.user_name },
        credentials.token,
      );
      if (!live) {
        await chat.disconnectUser();
        return;
      }

      const watching = chat.channel(
        credentials.channel_type,
        credentials.channel_id,
      );
      await watching.watch();

      if (!live) {
        return;
      }
      setClient(chat);
      setChannel(watching);
    };

    open().catch((error) => {
      if (live) {
        setFailure(error);
      }
    });

    return () => {
      live = false;
      setClient(null);
      setChannel(null);
      connected?.disconnectUser();
    };
  }, [agentID]);

  if (failure) {
    return <Failure error={failure} />;
  }
  if (!client || !channel) {
    return <p className="text-sm text-muted">Opening the conversation…</p>;
  }

  return (
    <div className="h-[32rem] overflow-hidden rounded-lg border border-line">
      <Chat client={client}>
        <Channel channel={channel}>
          <Window>
            <MessageList Message={ChatMessage} />
            <MessageComposer />
          </Window>
        </Channel>
      </Chat>
    </div>
  );
}

function ChatMessage() {
  const { message } = useMessageContext();
  const { channel } = useChannelStateContext();
  const custom = message as typeof message & {
    support_message?: ActivityMessage;
    generating?: boolean;
    source?: string;
  };
  const metadata = custom.support_message;
  const start = message.created_at?.toISOString() ?? "";
  const generating = custom.generating === true;
  const activity: ActivityMessage = metadata ?? {
    id: message.id,
    role:
      custom.source === "agent" || message.user?.id === channel.id
        ? "assistant"
        : "user",
    text: message.text ?? "",
    state: generating ? (message.text ? "writing" : "thinking") : "completed",
    response_started_at: start,
    state_started_at: start,
    finished_at: generating
      ? undefined
      : (message.updated_at?.toISOString() ?? start),
    saved: !generating,
  };
  return (
    <div className="w-full px-4 py-2">
      <ConversationMessage
        message={activity}
        showPersistence={!!metadata}
        showTiming={!!metadata}
        speaker={
          metadata
            ? undefined
            : activity.role === "assistant"
              ? "Agent"
              : message.user?.name || message.user?.id
        }
      />
    </div>
  );
}
