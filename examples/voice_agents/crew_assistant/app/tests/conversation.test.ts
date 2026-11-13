import { AgentChatStore } from '@stream-io/stream-agents/chat';
import type { LocalMessage } from 'stream-chat';
import type { AIMessage } from 'stream-chat-expo/ai-components';

import {
  type Chat,
  isAgentReply,
  keyOf,
  showsAgentReply,
  whenUsed,
  withTitles,
} from '../agent/conversation';

jest.mock('stream-chat-expo/ai-components', () => jest.requireActual('./sdkMock').aiComponentsMock());
jest.mock('../agent/agents', () => ({ crewBackend: jest.fn() }));

const chatWith = (id?: string) =>
  ({ id, state: new AgentChatStore({ usedSessionId: undefined as string | undefined }) }) as unknown as Chat;

const message = (fields: Partial<LocalMessage>) => ({ id: 'm1', ...fields }) as LocalMessage;

describe('isAgentReply', () => {
  it('reads ai_generated from custom, where stream-chat 10 keeps it', () => {
    expect(isAgentReply(message({ custom: { ai_generated: true } }))).toBe(true);
    expect(isAgentReply(message({ ai_generated: true } as Partial<LocalMessage>))).toBe(false);
  });
});

describe('showsAgentReply', () => {
  it('hides a reply the router created empty until something is written into it', () => {
    const generating = { ai_generated: true, generating: true };
    expect(showsAgentReply(message({ custom: generating }))).toBe(false);
    expect(showsAgentReply(message({ custom: generating, text: 'Hi' }))).toBe(true);
    expect(
      showsAgentReply(message({ attachments: [{ custom: {}, type: 'ai_reasoning' }], custom: generating })),
    ).toBe(true);
    expect(showsAgentReply(message({ custom: { ai_generated: true } }))).toBe(true);
    expect(showsAgentReply(message({ text: 'Hi' }))).toBe(false);
  });
});

describe('withTitles', () => {
  const reply = (toolCall: { name: string; displayTitle?: string }) =>
    ({ id: 'm1', parts: [{ text: 'Hi' }, { toolCall }] }) as unknown as AIMessage;

  it("titles a backend tool's step, which the router shows without one", () => {
    const titled = withTitles(reply({ name: 'search_handbook' }));
    expect(titled.parts?.[1].toolCall?.displayTitle).toBe('Searching the handbook');
    expect(titled.parts?.[0]).toEqual({ text: 'Hi' });
  });

  it('leaves a message alone when nothing needs a title', () => {
    const titled = reply({ displayTitle: 'Own title', name: 'search_handbook' });
    expect(withTitles(titled)).toBe(titled);
    const unknown = reply({ name: 'not_a_crew_tool' });
    expect(withTitles(unknown)).toBe(unknown);
  });
});

describe('keyOf', () => {
  it('keeps one key per conversation', () => {
    const first = chatWith();
    expect(keyOf(first)).toBe(keyOf(first));
    expect(keyOf(first)).not.toBe(keyOf(chatWith()));
  });
});

describe('whenUsed', () => {
  it('tells once per session a new conversation is used in, and never for one carried on', () => {
    const fresh = chatWith();
    const used = jest.fn();
    whenUsed(fresh, used);
    fresh.state.partialNext({ usedSessionId: 's1' });
    fresh.state.partialNext({ usedSessionId: 's1' });
    expect(used).toHaveBeenCalledTimes(1);

    const carriedOn = chatWith('s2');
    const listed = jest.fn();
    whenUsed(carriedOn, listed);
    carriedOn.state.partialNext({ usedSessionId: 's2' });
    expect(listed).not.toHaveBeenCalled();
  });
});
