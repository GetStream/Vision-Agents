import * as streamChat from "stream-chat";

/**
 * A real stream-chat client reading Chat from a test router, for the installed major: 9, or 10
 * in CI's js-stream-chat-10 job (`npm install --no-save stream-chat@^10.0.2 && npm test`).
 *
 * Typed by shape rather than by either version's declarations, so the suite compiles against both.
 * What differs between them, and why a 10 client pointed at the router the way a 9 one is reached
 * production instead (a 401 for the key `key`):
 *
 * - 10 takes its base URL under `client`; it ignores a top-level `baseURL`.
 * - 10 reads a channel at `/api/v2/chat/channels/...`, 9 at `/channels/...`.
 * - 10's `userID` is a getter: who the client is comes from `user`.
 * - 10's `watch` waits for a socket's connection id; this client has no socket, so one is set.
 * - 10 keeps the messages in `messagePaginator`, which takes in only messages whose `cid` is the
 *   channel's, keeps the app's fields under `custom`, and times them in unix nanoseconds.
 */

/** The installed major, from the `version` 10.0.2 on exports and 9 does not. */
export const CHAT_MAJOR = (streamChat as unknown as { version?: { major?: number } }).version?.major ?? 9;
const chat10 = CHAT_MAJOR >= 10;

export const CID = "agent:support-1";

/** Where Chat answers the channel's query, which `watch` and a re-read also send. */
export const QUERY_PATH = chat10 ? "/api/v2/chat/channels/agent/support-1/query" : "/channels/agent/support-1/query";

/** As much of a client as the tests touch. */
export interface TestChat {
  channel(type: string, id: string): unknown;
  dispatchEvent(event: Record<string, unknown>): void;
}

/** As much of a channel as the tests touch. */
interface TestChannel {
  watch(options?: { messages: { limit: number } }): Promise<unknown>;
  readonly state: { readonly messages?: readonly { id: string }[] };
  readonly messagePaginator?: { readonly items?: readonly { id: string }[]; flushState?(): void };
}

/** A client connected as john without a socket, reading Chat from `url`. */
export async function chatClient(url: string): Promise<TestChat> {
  const Chat = streamChat.StreamChat as unknown as new (key: string, options: Record<string, unknown>) => Record<string, unknown>;
  const chat = new Chat("key", chat10 ? { client: { baseURL: url } } : { baseURL: url });
  chat["user"] = { id: "john" };
  if (!chat10) {
    chat["userID"] = "john";
  }
  const claims = Buffer.from(JSON.stringify({ user_id: "john" })).toString("base64url");
  const tokens = chat["tokenManager"] as { setTokenOrProvider(token: string, user: { id: string }): Promise<void> };
  await tokens.setTokenOrProvider(`eyJhbGciOiJIUzI1NiJ9.${claims}.sig`, { id: "john" });
  if (chat10) {
    (chat["connectionIdManager"] as { resolveConnectionId(id: string): void }).resolveConnectionId("conn_1");
  }
  return chat as unknown as TestChat;
}

/**
 * A top-level message of the channel, written `second` seconds into 2026 by `user`, carrying the
 * app's `fields` where the installed stream-chat reads them.
 */
export function chatMessage(
  id: string,
  second: number,
  fields: Record<string, unknown> = {},
  user = "john",
): Record<string, unknown> {
  const ms = Date.UTC(2026, 0, 1, 0, 0, second);
  const at = chat10 ? ms * 1e6 : new Date(ms).toISOString();
  const message = { id, cid: CID, text: id, type: "regular", user: { id: user }, created_at: at, updated_at: at };
  return chat10 ? { ...message, custom: fields } : { ...message, ...fields };
}

/** What a channel query answers with these messages. */
export function channelPage(messages: Record<string, unknown>[]): { body: unknown } {
  return {
    body: {
      channel: { id: "support-1", type: "agent", cid: CID, config: {} },
      members: [],
      messages,
      duration: "1ms",
    },
  };
}

/** Reads the channel's first page, as a page showing it does. */
export async function watch(channel: unknown): Promise<void> {
  await (channel as TestChannel).watch({ messages: { limit: 30 } });
}

/** The ids of the messages the channel holds, oldest first. */
export function heldIds(channel: unknown): string[] {
  const held = channel as TestChannel;
  if (held.messagePaginator) {
    held.messagePaginator.flushState?.();
    return (held.messagePaginator.items ?? []).map((one) => one.id);
  }
  return (held.state.messages ?? []).map((one) => one.id);
}

/** A message arriving in the channel, as Chat's socket delivers it. */
export function arrive(chat: TestChat, message: Record<string, unknown>): void {
  chat.dispatchEvent({ type: "message.new", cid: CID, channel_type: "agent", channel_id: "support-1", message, user: message["user"] });
}

/** A message of the channel changing, as Chat's socket delivers it. */
export function update(chat: TestChat, message: Record<string, unknown>): void {
  chat.dispatchEvent({ type: "message.updated", cid: CID, channel_type: "agent", channel_id: "support-1", message, user: message["user"] });
}

/** A rewind of the channel to `messageId`, made at `second` seconds into 2026. */
export function rewind(chat: TestChat, messageId: string, inclusive: boolean, second: number, cid = CID): void {
  chat.dispatchEvent({
    type: "channel.rewound",
    cid,
    rewind: { id: "rw_1", message_id: messageId, inclusive, created_at: new Date(Date.UTC(2026, 0, 1, 0, 0, second)).toISOString() },
  });
}

/** Waits for something the SDK does after an event, without a fixed sleep. */
export async function until(done: () => boolean, what = "it never happened"): Promise<void> {
  // stream-chat 10 publishes a channel's messages at most every 500 ms outside Vitest and Jest.
  for (let tries = 0; !done(); tries++) {
    if (tries >= 1000) {
      throw new Error(what);
    }
    await new Promise((resolve) => setTimeout(resolve, 2));
  }
}
