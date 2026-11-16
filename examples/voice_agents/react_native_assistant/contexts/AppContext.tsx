import type { PropsWithChildren } from 'react';
import React, { useEffect, useMemo, useRef, useState } from 'react';

import { type ConversationSnapshot, type Session } from '@stream-io/stream-agents';
import {
  AgentChat,
  agentChatTransport,
  type OpenAgentSession,
} from '@stream-io/stream-agents/chat';
import type { LocalMessage, StreamChat } from 'stream-chat';
import { useStableCallback, useStateStore } from 'stream-chat-expo';
import {
  AIClientIdentity,
  AIConversationController,
  type AIToolApprover,
  createAuthSessionAuthorizer,
  fromStreamMessage,
  parseMessageParts,
} from 'stream-chat-expo/ai-components';

import { agents, agentSessions, streamToken } from '../agent/agents';
import { AgentConfig } from '../agent/config';
import {
  type Chat,
  type ChatState,
  chatStore,
  isAgentReply,
  whenUsed,
} from '../agent/conversation';
import { createDeviceTools } from '../agent/deviceTools';
import { imagesOf } from '../agent/photos';
import { THINK_HARDER } from '../chatOptions';

const IDLE: ConversationSnapshot = { kind: 'idle', skills: [] };

// This install, for the tool calls addressed to it.
const clientId = AIClientIdentity.installId();

// The tools this phone runs for the agent, such as telling it the local time.
const deviceTools = createDeviceTools({ clientId, userId: AgentConfig.user.id });

// Where a sign-in comes back to: the app's own scheme, which the web auth session hands back.
// The router takes only https or a native app's own scheme in reverse domain form (RFC 8252).
const SIGN_IN_SCHEME = 'io.getstream.expochatgptsample';
const SIGN_IN_RETURN_URL = `${SIGN_IN_SCHEME}://sign-in`;

// The person's answer to a call that asks first. stream-agents' `decide` answers only a call this
// session was asked on its own socket, and leaves any other alone, so a question shown from the
// channel, as after reopening the conversation or reloading the app, would wait forever. Such a
// call is answered with the frames `decide` sends, which the router matches by the call's id: an
// allowed client tool's call then waits for this phone, whose runner runs it.
const answerApproval = (session: Session, toolCallId: string, allowed: boolean) => {
  if (session.approvals.some((request) => request.id === toolCallId)) {
    session.decide(toolCallId, { allowed });
    return;
  }
  const socket = (
    session as unknown as { held(): { send(frame: Record<string, unknown>): void } }
  ).held();
  socket.send({ allowed, tool_call_id: toolCallId, type: 'tool_approval' });
  if (!allowed) {
    socket.send({
      error: 'The person declined this call.',
      tool_call_id: toolCallId,
      type: 'tool_result',
    });
  }
};

// Answers go over the session's socket, which opens for them in a conversation that was only
// read so far. Each throws when it can't send, so the card lets the person answer again.
const approverFor = (chat: Chat): AIToolApprover => ({
  authorizer: createAuthSessionAuthorizer({
    callbackScheme: SIGN_IN_SCHEME,
    decline: async (call) => (await chat.session()).declineLogin(call.id),
  }),
  clientId,
  decide: async (call, allowed) => answerApproval(await chat.session(), call.id, allowed),
  userId: AgentConfig.user.id,
});

export type AppContextValue = {
  /** What the agent is doing in the open conversation, from the session's socket. */
  agentState: ConversationSnapshot;
  /** The open conversation's messages and live reasoning. */
  chat: Chat | undefined;
  /** The open conversation's state, and what the person does in it. */
  conversation: AIConversationController | undefined;
  /** Bumped when the drawer's list of conversations may have changed. */
  conversationsVersion: number;
  /** Why the last conversation couldn't be opened. */
  failure: string | undefined;
  /** The open conversation's session id, or `undefined` for a new one that wasn't asked anything. */
  sessionId: string | undefined;
  /**
   * Carries on a conversation from the drawer, by its session id. Its messages show as soon as its
   * channel, `type:id` as the router lists it, is read.
   */
  openConversation: (sessionId: string, conversationId?: string) => void;
  /** Starts a new conversation. The router creates it with its first question. */
  startConversation: () => void;
};

export const AppContext = React.createContext<AppContextValue>({
  agentState: IDLE,
  chat: undefined,
  conversation: undefined,
  conversationsVersion: 0,
  failure: undefined,
  openConversation: () => {},
  sessionId: undefined,
  startConversation: () => {},
});

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

// What the provider follows of the messages: the latest reply, whose calls this phone runs, and
// the session, once one is open.
const selector = ({ messages, session }: ChatState) => ({
  lastReply: [...messages].reverse().find(isAgentReply),
  session,
});

// Carries a conversation on, or creates one, with the phone's tools declared on it.
const openSession: OpenAgentSession = (sessionId) =>
  sessionId
    ? agentSessions().resume(sessionId, { tools: deviceTools.sessionTools() })
    : agentSessions().create({
        modelOverwrites: { llm: AgentConfig.chatModel },
        return_url: SIGN_IN_RETURN_URL,
        tools: deviceTools.sessionTools(),
      });

const newChat = (sessionId?: string): Chat =>
  new AgentChat<LocalMessage>({ open: openSession, sessionId });

// The conversation's state over the chat. Asking goes through the session, which records the
// question in the channel and answers it there; the router takes photos as data URIs, and
// thinking harder as a high reasoning effort.
const conversationOf = (chat: Chat) =>
  new AIConversationController({
    // The agents router keeps the conversation in Stream Chat and can't rewind it, so a reply
    // can't be written again or a message edited: either would only be asked again as new.
    actions: { edit: false, regenerate: false },
    approver: approverFor(chat),
    transport: agentChatTransport(chat, {
      adapt: (message, reasoningText) => fromStreamMessage(message, { reasoningText }),
      images: imagesOf,
      reasoningEffort: (option) => (option?.id === THINK_HARDER.id ? 'high' : undefined),
    }),
  });

type Opened = {
  chat: Chat;
  conversation: AIConversationController;
  /** Stops waiting for a new conversation to be used. */
  stopWaiting: () => void;
};

export const AppProvider = ({ client, children }: PropsWithChildren<{ client: StreamChat }>) => {
  const [opened, setOpened] = useState<Opened>();
  const [failure, setFailure] = useState<string>();
  const [agentState, setAgentState] = useState<ConversationSnapshot>(IDLE);
  const [conversationsVersion, setConversationsVersion] = useState(0);
  const current = useRef<Opened | undefined>(undefined);
  const chat = opened?.chat;

  const followed = useStateStore(chat ? chatStore(chat) : undefined, selector);
  const session = followed?.session;

  // Runs the calls of the latest reply that wait for this phone, each once.
  const lastReply = followed?.lastReply;
  useEffect(() => {
    if (!lastReply) return;
    deviceTools.runner.run(parseMessageParts(lastReply.attachments), (call, result) =>
      deviceTools.send(call.id, result),
    );
  }, [lastReply]);

  // Shows a conversation in place of the one before it, at once, and lets the one before go.
  const show = (next: Chat) => {
    const previous = current.current;
    current.current = {
      chat: next,
      conversation: conversationOf(next),
      // The router lists a new conversation once its first question created it.
      stopWaiting: whenUsed(next, () => setConversationsVersion((version) => version + 1)),
    };
    setFailure(undefined);
    setOpened(current.current);
    previous?.stopWaiting();
    previous?.conversation.dispose();
    void previous?.chat.close().catch(() => {});
  };

  // A new conversation is only shown: the router creates it with its first question, so one
  // that is never asked anything isn't left in the drawer.
  const startConversation = useStableCallback(() => show(newChat()));

  const openConversation = useStableCallback((sessionId: string, conversationId?: string) => {
    if (sessionId === current.current?.chat.id) {
      return;
    }
    const next = newChat(sessionId);
    show(next);
    // The channel, when the drawer named it: reading it is quick, and carrying the session on,
    // which waits for something to need it, would also mark the conversation active.
    const [type, ...id] = (conversationId ?? '').split(':');
    const following = conversationId
      ? next.follow(client.channel(type || 'agent', id.join(':')))
      : next.session();
    void following.catch((error: unknown) => {
      if (current.current?.chat === next) setFailure(messageOf(error));
    });
  });

  // The open conversation: what the agent is doing, from the session's socket, once one is open.
  useEffect(() => {
    if (!session) {
      setAgentState(IDLE);
      return;
    }
    setAgentState(session.state.get());
    return session.state.subscribe(() => {
      const state = session.state.get();
      setAgentState(state);
      // The router titles a conversation as it goes.
      if (state.kind === 'idle') setConversationsVersion((version) => version + 1);
    });
  }, [session]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // The router and Stream Chat as the same user, on the app's own chat client.
        await agents.setUser(AgentConfig.user, streamToken);
        agents.use({ chat: client });
        if (!cancelled) {
          startConversation();
        }
      } catch (error) {
        setFailure(messageOf(error));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [client, startConversation]);

  const sessionId = session?.id ?? chat?.id;
  const conversation = opened?.conversation;
  const contextValue = useMemo(
    () => ({
      agentState,
      chat,
      conversation,
      conversationsVersion,
      failure,
      openConversation,
      sessionId,
      startConversation,
    }),
    [
      agentState,
      chat,
      conversation,
      conversationsVersion,
      failure,
      openConversation,
      sessionId,
      startConversation,
    ],
  );

  return <AppContext.Provider value={contextValue}>{children}</AppContext.Provider>;
};

export const useAppContext = () => React.useContext(AppContext);
