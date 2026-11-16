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

/**
 * Signs in as `userId` with the crew backend, which answers the user's Stream token. The chat,
 * the router and the backend itself all take it. It stands in for the restaurant's own sign-in.
 */
export const streamTokenFor = async (userId: string): Promise<string> => {
  const response = await fetch(`${AgentConfig.backendUrl}/stream-token`, {
    body: JSON.stringify({ user_id: userId }),
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

/** This app's user's Stream token. */
export const streamToken = () => streamTokenFor(AgentConfig.user.id);

// A token is used for 50 minutes of its hour, so one never expires on the way.
const TOKEN_REUSED_FOR_MS = 50 * 60 * 1000;
const signedIn = new Map<string, { expiresAt: number; token: Promise<string> }>();

const tokenOf = (userId: string) => {
  const held = signedIn.get(userId);
  if (held && held.expiresAt > Date.now()) return held.token;
  const token = streamTokenFor(userId);
  signedIn.set(userId, { expiresAt: Date.now() + TOKEN_REUSED_FOR_MS, token });
  token.catch(() => signedIn.delete(userId));
  return token;
};

/**
 * Asks the crew backend as `userId`, this app's user unless another is named, with their Stream
 * token. It answers the JSON the backend answered.
 */
export const crewBackend = async (
  path: string,
  { body, userId = AgentConfig.user.id }: { body?: unknown; userId?: string } = {},
) => {
  const response = await fetch(`${AgentConfig.backendUrl}/${path}`, {
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    headers: {
      Authorization: `Bearer ${await tokenOf(userId)}`,
      ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
    },
    method: body === undefined ? 'GET' : 'POST',
  });
  if (!response.ok) {
    throw new Error(`The crew backend answered ${response.status}: ${await response.text()}`);
  }
  return (await response.json()) as Record<string, unknown>;
};
