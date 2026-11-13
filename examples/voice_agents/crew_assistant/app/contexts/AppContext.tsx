import type { PropsWithChildren } from 'react';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AppState } from 'react-native';

import { type ConversationSnapshot, type Session } from '@stream-io/stream-agents';
import {
  AgentChat,
  agentChatTransport,
  type OpenAgentSession,
} from '@stream-io/stream-agents/chat';
import type { Channel, LocalMessage, StreamChat } from 'stream-chat';
import { useStableCallback, useStateStore } from 'stream-chat-expo';
import {
  AIClientIdentity,
  AIConversationController,
  type AIToolApprover,
  createAuthSessionAuthorizer,
  fromStreamMessage,
  parseMessageParts,
} from 'stream-chat-expo/ai-components';

import { agents, agentSessions, crewBackend, streamToken } from '../agent/agents';
import { AgentConfig } from '../agent/config';
import {
  type Chat,
  type ChatState,
  chatStore,
  isAgentReply,
  showsAgentReply,
  whenUsed,
  withTitles,
} from '../agent/conversation';
import { createDeviceTools } from '../agent/deviceTools';
import { imagesOf } from '../agent/photos';

const IDLE: ConversationSnapshot = { kind: 'idle', skills: [] };

// This install, for the tool calls addressed to it.
const clientId = AIClientIdentity.installId();

// The tools this phone runs for the agent, such as telling it the local time.
const deviceTools = createDeviceTools({ clientId, userId: AgentConfig.user.id });

// Where a sign-in comes back to: the app's own scheme, which the web auth session hands back.
// The router takes only https or a native app's own scheme in reverse domain form (RFC 8252).
const SIGN_IN_SCHEME = 'io.getstream.expocrewassistant';
const SIGN_IN_RETURN_URL = `${SIGN_IN_SCHEME}://sign-in`;

// Answers go over the session's socket, which opens for them in a conversation that was only
// read so far. Each throws when it can't send, so the card lets the person answer again.
const approverFor = (chat: Chat): AIToolApprover => ({
  authorizer: createAuthSessionAuthorizer({
    callbackScheme: SIGN_IN_SCHEME,
    decline: async (call) => (await chat.session()).declineLogin(call.id),
  }),
  clientId,
  decide: (call, allowed) => chat.decide(call.id, { allowed }),
  userId: AgentConfig.user.id,
});

export type AppContextValue = {
  /** What the agent is doing in the open conversation, from the session's socket. */
  agentState: ConversationSnapshot;
  /** The open conversation's messages and live reasoning. */
  chat: Chat | undefined;
  /** Whether the agent was asked to open the open conversation and hasn't written yet. */
  awaitingAgent: boolean;
  /** The open conversation's state, and what the person does in it. */
  conversation: AIConversationController | undefined;
  /** Bumped when the drawer's list of conversations may have changed. */
  conversationsVersion: number;
  /** Why the last conversation couldn't be opened. */
  failure: string | undefined;
  /** Whether the router knows who this is, so conversations can start. */
  isReady: boolean;
  /** The open conversation's session id, or `undefined` for a new one that wasn't asked anything. */
  sessionId: string | undefined;
  /**
   * Carries on a conversation from the drawer, by its session id. Its messages show as soon as its
   * channel, `type:id` as the router lists it, is read.
   */
  openConversation: (sessionId: string, conversationId?: string) => void;
  /** Starts a new conversation. The router creates it with its first question. */
  startConversation: () => void;
  /**
   * Asks with a voice memo recorded on the phone: it goes up to the conversation's channel, and the
   * router transcribes it, answers it and shows it on the person's message.
   */
  sendVoiceMemo: (uri: string) => Promise<void>;
  /** Asks again what the agent couldn't answer: the last question, or the event it was told of. */
  retry: () => void;
  /** Starts a new conversation and asks it `text`, as a flow or a scanned code does. */
  startConversationWith: (text: string) => void;
  /**
   * Starts a new conversation that the agent opens, about something that happened, such as a
   * clock-in: the restaurant's systems would tell the backend, here the app does.
   */
  startConversationAbout: (event: CrewEvent) => void;
};

export const AppContext = React.createContext<AppContextValue>({
  agentState: IDLE,
  awaitingAgent: false,
  chat: undefined,
  conversation: undefined,
  conversationsVersion: 0,
  failure: undefined,
  isReady: false,
  openConversation: () => {},
  retry: () => {},
  sendVoiceMemo: () => Promise.resolve(),
  sessionId: undefined,
  startConversation: () => {},
  startConversationAbout: () => {},
  startConversationWith: () => {},
});

/** What the restaurant's systems tell the agent about, so it speaks first. */
export type CrewEvent = 'clock_in' | 'clock_out' | 'back_from_time_off' | 'onboarding';

/** What was last asked of the agent here, so it can be asked again when it couldn't answer. */
type AgentRequest =
  | { text: string; custom?: Record<string, unknown> }
  | { event: CrewEvent }
  | { voiceMemoUrl: string };

// Tells the backend what happened, which has the agent open the conversation about it.
const tellBackend = (kind: CrewEvent, sessionId: string) =>
  crewBackend('events', {
    body: {
      kind,
      session_id: sessionId,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    },
  });

// Asks with a voice memo at `url`. stream-agents' `responses.create` has no voice note yet, so this is
// the request it makes, with the voice note the router takes beside the text.
const askWithVoiceMemo = async (session: Session, url: string) => {
  deviceTools.addressed = false;
  await agents.post('/v1/agents/sessions/{id}/responses', {
    body: { audio: { url }, request_id: crypto.randomUUID(), text: '' },
    path: { id: session.id },
  });
};

// Tells the backend of an event, or asks with a voice memo, in `chat`, and only then counts the
// conversation as used, as a question does: one where neither got through isn't listed, and is
// deleted when the person leaves it.
const tellAbout = async (chat: Chat, kind: CrewEvent) => {
  const session = await chat.session();
  await tellBackend(kind, session.id);
  await chat.use();
};

const askVoiceMemo = async (chat: Chat, url: string) => {
  await askWithVoiceMemo(await chat.session(), url);
  await chat.use();
};

// What the recorder wrote, by its file's extension.
const audioTypeOf = (uri: string) => (/\.(m4a|mp4)$/i.test(uri) ? 'audio/mp4' : 'audio/aac');

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

// What the provider follows of the messages: the latest reply, whose calls this phone runs,
// whether the agent has written anything yet, and the session, once one is open.
const selector = ({ messages, session }: ChatState) => ({
  hasShownReply: messages.some(showsAgentReply),
  lastReply: [...messages].reverse().find(isAgentReply),
  session,
});

// Carries a conversation on, or creates one, with the phone's tools declared on it.
const openSession: OpenAgentSession = (sessionId) =>
  sessionId
    ? agentSessions().resume(sessionId, { tools: deviceTools.sessionTools() })
    : agentSessions().create({
        return_url: SIGN_IN_RETURN_URL,
        tools: deviceTools.sessionTools(),
      });

const newChat = (sessionId?: string): Chat =>
  new AgentChat<LocalMessage>({ open: openSession, sessionId });

// The conversation's state over the chat. Asking goes through the session, which records the
// question in the channel and answers it there, with photos as data URIs. `asked` hears each
// request as it goes out: the question, or `undefined` for a retry of the last one.
const conversationOf = (chat: Chat, asked: (request: AgentRequest | undefined) => void) => {
  // Every tool's step is titled, the backend's too.
  const transport = agentChatTransport(chat, {
    adapt: (message, reasoningText) => withTitles(fromStreamMessage(message, { reasoningText })),
    images: imagesOf,
  });
  return new AIConversationController({
    // The agents router keeps the conversation in Stream Chat and can't rewind it, so a reply
    // can't be written again or a message edited: either would only be asked again as new.
    actions: { edit: false, regenerate: false },
    approver: approverFor(chat),
    transport: {
      ...transport,
      send: (request) => {
        asked(
          // A retry asks the same question again without what it went out with, such as a
          // generated-UI action, which the request it retries still has.
          request.kind === 'regenerate'
            ? undefined
            : { custom: request.custom, text: request.text },
        );
        return transport.send(request);
      },
    },
  });
};

type Opened = {
  chat: Chat;
  conversation: AIConversationController;
  /** Stops waiting for a new conversation to be used. */
  stopWaiting: () => void;
  /** What was last asked of the agent here. */
  lastRequest?: AgentRequest;
};

export const AppProvider = ({ client, children }: PropsWithChildren<{ client: StreamChat }>) => {
  const [opened, setOpened] = useState<Opened>();
  const [failure, setFailure] = useState<string>();
  const [isReady, setIsReady] = useState(false);
  const [agentState, setAgentState] = useState<ConversationSnapshot>(IDLE);
  const [conversationsVersion, setConversationsVersion] = useState(0);
  // The conversation the agent was asked to open, about an event.
  const [awaitedChat, setAwaitedChat] = useState<Chat>();
  const current = useRef<Opened | undefined>(undefined);
  const chat = opened?.chat;

  const followed = useStateStore(chat ? chatStore(chat) : undefined, selector);
  const session = followed?.session;
  const awaitingAgent = chat !== undefined && chat === awaitedChat && !followed?.hasShownReply;

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
    const shown: Opened = {
      chat: next,
      conversation: conversationOf(next, (request) => {
        if (request) shown.lastRequest = request;
        setFailure(undefined);
      }),
      // The router lists a new conversation once something used it.
      stopWaiting: whenUsed(next, () => setConversationsVersion((version) => version + 1)),
    };
    current.current = shown;
    setFailure(undefined);
    setOpened(shown);
    previous?.stopWaiting();
    previous?.conversation.dispose();
    void previous?.chat.close().catch(() => {});
    return shown;
  };

  // A new conversation is only shown: the router creates it with its first question, so one
  // that is never asked anything isn't left in the drawer.
  const startConversation = useStableCallback(() => {
    show(newChat());
  });

  const startConversationWith = useStableCallback((text: string) => {
    void show(newChat()).conversation.send({ text });
  });

  const startConversationAbout = useStableCallback((kind: CrewEvent) => {
    const next = newChat();
    show(next).lastRequest = { event: kind };
    setAwaitedChat(next);
    // Nothing is asked here, but once the backend has the event the conversation is used: the
    // agent writes first, and it is kept and listed.
    tellAbout(next, kind).catch((error: unknown) => {
      setAwaitedChat((awaited) => (awaited === next ? undefined : awaited));
      if (current.current?.chat === next) setFailure(messageOf(error));
    });
  });

  const sendVoiceMemo = useStableCallback(async (uri: string) => {
    const shown = current.current;
    if (!shown) return;
    setFailure(undefined);
    try {
      const session = await shown.chat.session();
      const { channel } = await session.chat();
      const name = uri.split('/').pop() ?? 'voice-memo.aac';
      const { file } = await (channel as Channel).uploadChannelFile({
        file: { name, type: audioTypeOf(uri), uri },
      });
      if (!file) throw new Error('The voice memo did not upload');
      shown.lastRequest = { voiceMemoUrl: file };
      await askVoiceMemo(shown.chat, file);
    } catch (error) {
      setFailure(messageOf(error));
    }
  });

  const retry = useStableCallback(() => {
    const shown = current.current;
    const last = shown?.lastRequest;
    if (!shown || !last) return;
    setFailure(undefined);
    if ('voiceMemoUrl' in last) {
      askVoiceMemo(shown.chat, last.voiceMemoUrl).catch((error: unknown) =>
        setFailure(messageOf(error)),
      );
      return;
    }
    if ('event' in last) {
      tellAbout(shown.chat, last.event).catch((error: unknown) => setFailure(messageOf(error)));
      return;
    }
    void shown.conversation.send({ custom: last.custom, text: last.text });
  });

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

  // The session's socket drops once the app has been in the background a while. Coming back, the
  // open conversation is carried on in a new one, so what the agent does shows again.
  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      const shown = current.current?.chat;
      if (state !== 'active' || !shown) return;
      const { session: held } = shown.state.getLatestValue();
      if (held && !held.live) void shown.prepare().catch(() => {});
    });
    return () => subscription.remove();
  }, []);

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
          setIsReady(true);
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
      awaitingAgent,
      chat,
      conversation,
      conversationsVersion,
      failure,
      isReady,
      openConversation,
      retry,
      sendVoiceMemo,
      sessionId,
      startConversation,
      startConversationAbout,
      startConversationWith,
    }),
    [
      agentState,
      awaitingAgent,
      chat,
      conversation,
      conversationsVersion,
      failure,
      isReady,
      openConversation,
      retry,
      sendVoiceMemo,
      sessionId,
      startConversation,
      startConversationAbout,
      startConversationWith,
    ],
  );

  return <AppContext.Provider value={contextValue}>{children}</AppContext.Provider>;
};

export const useAppContext = () => React.useContext(AppContext);
