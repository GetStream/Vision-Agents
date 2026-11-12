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
  const of = field("of");
  const index = field("index");
  const count = field("count");
  const responseId = field("response_id");
  if ((of !== "reply" && of !== "question") || typeof index !== "number" || typeof count !== "number" ||
    typeof responseId !== "string" || !responseId) {
    return undefined;
  }
  const parent = field("parent_response_id");
  const question = field("question_response_id");
  return {
    of,
    index,
    count,
    responseId,
    parentResponseId: typeof parent === "string" ? parent : "",
    questionResponseId: typeof question === "string" ? question : "",
  };
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
  if (typeof (client as Partial<EventSource> | null)?.on !== "function") {
    return undefined;
  }
  const source = client as EventSource;
  const subscriptions = REWOUND_EVENTS.map((type) =>
    source.on(type, (event) => {
      if (cidOf(event) === cid) {
        void rewound(channel, event);
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
  if (!loaded || typeof (channel as { query?: unknown } | null)?.query !== "function") {
    return;
  }
  try {
    await (channel as { query(): Promise<unknown> }).query();
  } catch {
    // The messages are already dropped; the next read fills the window in.
    return;
  }
  // A read just after a rewind can still return what it hid, and the read merges it back.
  prune(channel, { ...hidden, from });
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

/** stream-chat 10 keeps a channel's messages in its paginator. */
interface PaginatedChannel {
  messagePaginator: { readonly items?: readonly unknown[]; removeItem(params: { id: string }): unknown };
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
    const at = createdAt instanceof Date ? createdAt.getTime() : typeof createdAt === "string" ? Date.parse(createdAt) : NaN;
    if (typeof id === "string" && Number.isFinite(at)) {
      found.push({ id, at });
    }
  };
  const paginator = paginatorOf(channel);
  if (paginator) {
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
