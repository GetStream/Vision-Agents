import type { Schemas } from "./client.js";
import { RouterError } from "./errors.js";
import { messageField } from "./message-field.js";

/** Where a regenerate, an edit or an activation rewound the conversation's channel to. */
export type RewoundTo = Schemas["RewoundTo"];

/**
 * The codes a regenerate, an edit or an activation is refused with that a page shows the
 * person, since each says to carry on in a new chat or that the branch is gone:
 *
 * - `branch_limit_reached` (409): the conversation has used its 100 edit and regenerate
 *   points. Asking and reading still work; only another regenerate or edit is refused.
 * - `branch_too_long` (409): bringing the branch back would send more than 100 messages again.
 * - `branch_unavailable` (410): the branch's copies were deleted, as a memory truncate deletes
 *   them, so it can be neither read nor brought back.
 */
export type BranchErrorCode = "branch_limit_reached" | "branch_too_long" | "branch_unavailable";

const BRANCH_ERROR_CODES: ReadonlySet<string> = new Set<BranchErrorCode>([
  "branch_limit_reached",
  "branch_too_long",
  "branch_unavailable",
]);

/** The branch refusal `error` is, or undefined for any other failure. */
export function branchErrorOf(error: unknown): BranchErrorCode | undefined {
  return error instanceof RouterError && error.code && BRANCH_ERROR_CODES.has(error.code)
    ? (error.code as BranchErrorCode)
    : undefined;
}

/**
 * Where a message stands among its alternatives, as the router marked it in Stream Chat: what
 * a ‹ i/n › control shows, and the response `responses.siblings` lists it by.
 */
export interface SupportBranch {
  /** reply for an answer and its regenerations, question for a question and its edits. */
  readonly of: "reply" | "question";
  /** Its place, from 1, oldest first: the i of ‹ i/n ›. */
  readonly index: number;
  /** How many there are: the n. */
  readonly count: number;
  /** The response the message belongs to. */
  readonly responseId: string;
  /** The response it was answered after, or the empty string for a conversation's first. */
  readonly parentResponseId: string;
  /** The first response that answered its question, or the empty string. */
  readonly questionResponseId: string;
}

/**
 * The `support_branch` mark of a Stream Chat message, or undefined for a message without one:
 * one somebody else wrote, one from before branches, or one typed during a call.
 *
 * The router leaves `of`, `index` and `count` out of the mark of a message with no
 * alternatives, which is 1 of 1, a reply or a question by who wrote it (`support_runtime.role`);
 * a place it cannot read is 1 of 1 too, as the router reads it.
 *
 * For an app rendering ‹ i/n › without the React components, whose `fromStreamMessage` reads
 * the same mark. It takes a message from stream-chat 9 or 10, or a `BranchMessage` that
 * `responses.branch` read back.
 */
export function supportBranchOf(message: object): SupportBranch | undefined {
  const mark = messageField(message, "support_branch");
  if (typeof mark !== "object" || mark === null || Array.isArray(mark)) {
    return undefined;
  }
  const field = (key: string): unknown => (mark as Record<string, unknown>)[key];
  const responseId = field("response_id");
  if (typeof responseId !== "string" || !responseId) {
    return undefined;
  }
  const of = field("of");
  const index = field("index");
  const count = field("count");
  const placed =
    (of === "reply" || of === "question") && typeof index === "number" && typeof count === "number" &&
    index >= 1 && index <= count;
  const alone = placed ? undefined : roleOf(message);
  if (!placed && !alone) {
    return undefined;
  }
  const parent = field("parent_response_id");
  const question = field("question_response_id");
  return {
    of: placed ? of : (alone as "reply" | "question"),
    index: placed ? index : 1,
    count: placed ? count : 1,
    responseId,
    parentResponseId: typeof parent === "string" ? parent : "",
    questionResponseId: typeof question === "string" ? question : "",
  };
}

/** reply for a message of the agent's, question for one of the person's, by `support_runtime.role`. */
function roleOf(message: object): "reply" | "question" | undefined {
  const runtime = messageField(message, "support_runtime");
  const role = typeof runtime === "object" && runtime !== null ? (runtime as Record<string, unknown>)["role"] : undefined;
  return role === "assistant" ? "reply" : role === "user" ? "question" : undefined;
}

/** What a chat client delivers events with: `StreamChat.on`, in 9 and 10. */
interface EventSource {
  on(type: string, handler: (event: Record<string, unknown>) => void): { unsubscribe(): void };
}

/** A message held in a channel's state, as far as a rewind needs it. */
interface Held {
  readonly id: string;
  /** `created_at` in milliseconds. */
  readonly at: number;
}

/** What one rewind hides of what this device holds. */
interface Hidden {
  /** The message the channel was rewound to. */
  readonly messageId: string;
  /** Whether that message is hidden too. */
  readonly inclusive: boolean;
  /**
   * Its `created_at` in milliseconds, when known from an earlier prune: an inclusive rewind
   * leaves nothing to find it by once it has been dropped.
   */
  readonly from?: number | undefined;
  /** When the rewind was made: a message after it was written since, and is not hidden. */
  readonly until?: number | undefined;
  /** Only these ids may be dropped: what was held before the request that rewound. */
  readonly only?: ReadonlySet<string> | undefined;
}

/** The two events Chat sends a rewind as: to a watcher, and to a member who is not watching. */
const REWOUND_EVENTS = ["channel.rewound", "notification.channel_rewound"] as const;

/**
 * Keeps one channel's messages in step with the rewinds Chat makes of it, until `stop`.
 *
 * Chat hides the rewound messages from every read, but a client that already holds them keeps
 * them: stream-chat has no handler for `channel.rewound`, and `channel.query` merges what it
 * reads into what is held rather than replacing it, so querying again alone never takes
 * anything away. So the messages are dropped from the state first, and the channel is read
 * again after, for whatever the rewind brought into the window.
 *
 * Listens on the client rather than the channel, so it hears the notification a member who is
 * not watching is sent, and so it runs before the channel's own listeners, such as a UI's,
 * which then read the state without the hidden messages.
 *
 * Returns undefined for a client that delivers no events, such as an app's stand-in.
 */
export function followRewinds(client: unknown, channel: unknown, cid: string): (() => void) | undefined {
  return onRewound(client, cid, (event) => void rewound(channel, event));
}

/**
 * Calls `handler` with each rewind Chat makes of the channel `cid`, as a watcher or as a member
 * who is not watching hears it, until the returned function is called. Undefined for a client
 * that delivers no events.
 */
export function onRewound(
  client: unknown,
  cid: string,
  handler: (event: Record<string, unknown>) => void,
): (() => void) | undefined {
  if (typeof (client as Partial<EventSource> | null)?.on !== "function") {
    return undefined;
  }
  const source = client as EventSource;
  const subscriptions = REWOUND_EVENTS.map((type) =>
    source.on(type, (event) => {
      if (cidOf(event) === cid) {
        handler(event);
      }
    }),
  );
  return () => {
    for (const subscription of subscriptions) {
      subscription.unsubscribe();
    }
  };
}

/**
 * The ids this device holds of the channel, for `pruneRewound` to drop only what was there
 * before a request rewound it.
 */
export function heldIds(channel: unknown): ReadonlySet<string> {
  return new Set(held(channel).map(({ id }) => id));
}

/**
 * Drops what a regenerate, an edit or an activation hid, from the answer's `rewound_to`, so
 * the device that asked never shows the old and the new answer together, even before the
 * `channel.rewound` event arrives. Only what was held before the request is dropped: the new
 * question, the new answer and the messages an activation sent again may have arrived while
 * it was answered.
 */
export function pruneRewound(channel: unknown, to: RewoundTo, before: ReadonlySet<string>): void {
  prune(channel, { messageId: to.message_id, inclusive: to.inclusive, only: before });
}

/** One rewind event: drop at once, then read the channel again. */
async function rewound(channel: unknown, event: Record<string, unknown>): Promise<void> {
  const rewind = event["rewind"];
  if (typeof rewind !== "object" || rewind === null) {
    return;
  }
  const { message_id: messageId, inclusive, created_at: createdAt } = rewind as Record<string, unknown>;
  if (typeof messageId !== "string" || !messageId) {
    return;
  }
  const made = typeof createdAt === "string" ? Date.parse(createdAt) : NaN;
  const hidden: Hidden = {
    messageId,
    inclusive: inclusive === true,
    ...(Number.isFinite(made) ? { until: made } : {}),
  };
  const loaded = held(channel).length > 0 || (channel as { initialized?: unknown } | null)?.initialized === true;
  const from = prune(channel, hidden);
  if (from === undefined && loaded) {
    // The message rewound to is not held, so where the hidden range starts is not known here:
    // what is held cannot be trusted, and the read below is all that is shown.
    reset(channel);
  }
  const read = readAgain(channel);
  if (!loaded || !read) {
    return;
  }
  try {
    await read();
  } catch {
    // The messages are already dropped; the next read fills the window in.
    return;
  }
  // A read just after a rewind can still return what it hid, and the read merges it back.
  prune(channel, { ...hidden, from });
}

/**
 * What reads the channel's latest messages again: `reload` in stream-chat 10, whose `query`
 * places no messages, and `query` in 9, which merges them into what is held.
 */
function readAgain(channel: unknown): (() => Promise<unknown>) | undefined {
  const held = channel as { reload?: unknown; query?: unknown } | null;
  if (paginatorOf(channel) && typeof held?.reload === "function") {
    return () => (held as { reload(): Promise<unknown> }).reload();
  }
  return typeof held?.query === "function" ? () => (held as { query(): Promise<unknown> }).query() : undefined;
}

/**
 * Drops the messages a rewind hides from the channel's state, and answers the rewound-to
 * message's `created_at`, or undefined when it is not held and was not known.
 */
function prune(channel: unknown, hidden: Hidden): number | undefined {
  const messages = held(channel);
  const from = messages.find(({ id }) => id === hidden.messageId)?.at ?? hidden.from;
  if (from === undefined) {
    return undefined;
  }
  const drop = messages.filter(({ id, at }) =>
    (id === hidden.messageId ? hidden.inclusive : at > from) &&
    (hidden.until === undefined || at <= hidden.until) &&
    (hidden.only === undefined || hidden.only.has(id)),
  );
  remove(channel, drop);
  return from;
}

/**
 * stream-chat 10 keeps a channel's messages in its paginator, which publishes `items` at most
 * every 500 ms unless flushed.
 */
interface PaginatedChannel {
  messagePaginator: {
    readonly items?: readonly unknown[];
    removeItem(params: { id: string }): unknown;
    flushState?(): void;
  };
}

/** stream-chat 9 keeps them in message sets: the latest, and any a jump loaded. */
interface SetChannel {
  state: {
    messageSets: readonly { readonly messages: readonly unknown[] }[];
    removeMessage(message: { id: string; messageSetIndex?: number }): unknown;
    initMessages?(): void;
  };
}

/** Every top-level message the channel holds, in any message set. */
function held(channel: unknown): Held[] {
  const found: Held[] = [];
  const add = (message: unknown) => {
    if (typeof message !== "object" || message === null) {
      return;
    }
    const { id, created_at: createdAt } = message as { id?: unknown; created_at?: unknown };
    const at = millisecondsOf(createdAt);
    if (typeof id === "string" && Number.isFinite(at)) {
      found.push({ id, at });
    }
  };
  const paginator = paginatorOf(channel);
  if (paginator) {
    // What a rewind or a read changed a moment ago may not be published yet.
    paginator.flushState?.();
    for (const message of paginator.items ?? []) {
      add(message);
    }
    return found;
  }
  for (const set of setsOf(channel)?.messageSets ?? []) {
    for (const message of set.messages) {
      add(message);
    }
  }
  return found;
}

/**
 * A message's `created_at` in milliseconds: a Date in stream-chat 9, unix nanoseconds in 10, or
 * the string either was given. NaN for anything else.
 */
function millisecondsOf(at: unknown): number {
  if (at instanceof Date) {
    return at.getTime();
  }
  if (typeof at === "string") {
    return Date.parse(at);
  }
  // Nanoseconds since 1970 pass 1e14 in 1973; milliseconds do not until the year 5138.
  return typeof at === "number" ? (at > 1e14 ? at / 1e6 : at) : NaN;
}

function remove(channel: unknown, messages: readonly Held[]): void {
  const paginator = paginatorOf(channel);
  const state = paginator ? undefined : setsOf(channel);
  for (const { id } of messages) {
    if (paginator) {
      paginator.removeItem({ id });
      continue;
    }
    // A message can be in more than one set, and removeMessage takes it out of one.
    state?.messageSets.forEach((set, messageSetIndex) => {
      if (set.messages.some((message) => (message as { id?: unknown }).id === id)) {
        state.removeMessage({ id, messageSetIndex });
      }
    });
  }
  // Shown at once, rather than after the paginator's throttle, beside the answer that replaced them.
  paginator?.flushState?.();
}

/** Forgets every message held, for the next read to fill in. */
function reset(channel: unknown): void {
  const paginator = paginatorOf(channel);
  if (paginator) {
    remove(channel, held(channel));
    return;
  }
  setsOf(channel)?.initMessages?.();
}

function paginatorOf(channel: unknown): PaginatedChannel["messagePaginator"] | undefined {
  const paginator = (channel as Partial<PaginatedChannel> | null)?.messagePaginator;
  return typeof paginator?.removeItem === "function" ? paginator : undefined;
}

function setsOf(channel: unknown): SetChannel["state"] | undefined {
  const state = (channel as Partial<SetChannel> | null)?.state;
  return Array.isArray(state?.messageSets) && typeof state.removeMessage === "function" ? state : undefined;
}

/** The channel an event is about: `cid`, or its `channel`'s on a notification without one. */
function cidOf(event: Record<string, unknown>): string | undefined {
  if (typeof event["cid"] === "string") {
    return event["cid"];
  }
  const channel = event["channel"];
  const cid = typeof channel === "object" && channel !== null ? (channel as Record<string, unknown>)["cid"] : undefined;
  return typeof cid === "string" ? cid : undefined;
}
