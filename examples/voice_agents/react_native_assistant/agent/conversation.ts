import type { StateStore } from '@stream-io/state-store';
import type { AgentChat, AgentChatState } from '@stream-io/stream-agents/chat';
import type { LocalMessage } from 'stream-chat';

/** A written conversation with the agent, over the messages of the channel it is kept in. */
export type Chat = AgentChat<LocalMessage>;

export type ChatState = AgentChatState<LocalMessage>;

/**
 * The chat's state for `useStateStore`. The SDK's store has the shape of `@stream-io/state-store`'s
 * `StateStore` (`getLatestValue` and `subscribeWithSelector`, which is all `useStateStore` calls)
 * without depending on it, so the type is only told so here.
 */
export const chatStore = (chat: Chat) => chat.state as unknown as StateStore<ChatState>;

let keys = 0;
const chatKeys = new WeakMap<Chat, string>();

/** Tells a conversation apart from the one shown before it, for a React key. */
export const keyOf = (chat: Chat) => {
  let key = chatKeys.get(chat);
  if (!key) {
    key = `chat-${++keys}`;
    chatKeys.set(chat, key);
  }
  return key;
};

/** A reply the agent wrote. */
export const isAgentReply = (message: LocalMessage) => message.custom?.ai_generated === true;

/**
 * Calls `used` when a new conversation is used, such as asked something: the router lists it from
 * then on. A first question that never reached the router has its session deleted, and the next
 * one creates another, which is listed in its turn. A conversation carried on is listed already,
 * and isn't waited for. Returns a function that stops waiting.
 */
export const whenUsed = (chat: Chat, used: () => void) => {
  if (chat.id !== undefined) return () => {};
  let listed: string | undefined;
  return chat.state.subscribeWithSelector(
    ({ usedSessionId }) => ({ usedSessionId }),
    ({ usedSessionId }) => {
      if (usedSessionId === undefined || usedSessionId === listed) return;
      listed = usedSessionId;
      used();
    },
  );
};
