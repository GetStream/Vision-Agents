"use client";

import { useEffect, useState } from "react";
import { StreamChat, type Channel as ChatChannel } from "stream-chat";
import {
  Channel,
  Chat,
  MessageComposer,
  MessageList,
  Window,
} from "stream-chat-react";

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

      const chat = StreamChat.getInstance(credentials.api_key);
      await chat.connectUser(
        { id: credentials.user_id, name: credentials.user_name },
        credentials.token,
      );
      connected = chat;

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
            <MessageList />
            <MessageComposer />
          </Window>
        </Channel>
      </Chat>
    </div>
  );
}
