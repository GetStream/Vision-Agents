import { AgentChatStore } from '@stream-io/stream-agents/chat';
import type { LocalMessage } from 'stream-chat';

import { type Chat, isAgentReply, keyOf, whenUsed } from '../agent/conversation';

const chatWith = (id?: string) =>
  ({ id, state: new AgentChatStore({ usedSessionId: undefined as string | undefined }) }) as unknown as Chat;

const message = (fields: Partial<LocalMessage>) => ({ id: 'm1', ...fields }) as LocalMessage;

describe('isAgentReply', () => {
  it('reads ai_generated from custom, where stream-chat 10 keeps it', () => {
    expect(isAgentReply(message({ custom: { ai_generated: true } }))).toBe(true);
    expect(isAgentReply(message({ custom: {} }))).toBe(false);
    expect(isAgentReply(message({ ai_generated: true } as Partial<LocalMessage>))).toBe(false);
  });
});

describe('keyOf', () => {
  it('keeps one key per conversation', () => {
    const first = chatWith();
    const second = chatWith();
    expect(keyOf(first)).toBe(keyOf(first));
    expect(keyOf(first)).not.toBe(keyOf(second));
  });
});

describe('whenUsed', () => {
  it('tells once per session a new conversation is used in', () => {
    const chat = chatWith();
    const used = jest.fn();
    const stop = whenUsed(chat, used);
    expect(used).not.toHaveBeenCalled();

    chat.state.partialNext({ usedSessionId: 's1' });
    chat.state.partialNext({ usedSessionId: 's1' });
    expect(used).toHaveBeenCalledTimes(1);

    // A first question that never reached the router is asked again in a new session.
    chat.state.partialNext({ usedSessionId: 's2' });
    expect(used).toHaveBeenCalledTimes(2);

    stop();
    chat.state.partialNext({ usedSessionId: 's3' });
    expect(used).toHaveBeenCalledTimes(2);
  });

  it("doesn't wait for a conversation carried on, which is listed already", () => {
    const chat = chatWith('s1');
    const used = jest.fn();
    whenUsed(chat, used);
    chat.state.partialNext({ usedSessionId: 's1' });
    expect(used).not.toHaveBeenCalled();
  });
});
