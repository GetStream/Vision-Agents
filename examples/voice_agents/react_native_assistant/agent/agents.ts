import { Client } from '@stream-io/stream-agents';
// Lets `session.chat()` open the conversation's channel on the app's Stream Chat client.
import '@stream-io/stream-agents/chat';

import { AgentConfig } from './config';

/** The router, as this app's user. */
export const agents = new Client({
  apiKey: AgentConfig.streamApiKey,
  customerId: AgentConfig.customerId,
  url: AgentConfig.routerUrl,
});

/** The agent's conversations: opening a new one and carrying an old one on. */
export const agentSessions = () => agents.agent(AgentConfig.agentName).sessions;

/** Asks the token backend for this user's Stream token, which both the chat and the router take. */
export const streamToken = async (): Promise<string> => {
  const response = await fetch(`${AgentConfig.tokenUrl}/stream-token`, {
    body: JSON.stringify({ user_id: AgentConfig.user.id }),
    headers: { 'Content-Type': 'application/json' },
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`The token backend answered ${response.status}: is \`go run ./backend\` up?`);
  }
  const { token } = (await response.json()) as { token?: string };
  if (!token) {
    throw new Error('The token backend answered without a token.');
  }
  return token;
};
