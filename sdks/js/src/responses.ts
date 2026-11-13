import type { RewoundTo } from "./branches.js";
import { onRetry, type Client, type Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** How many items are read per request while unwinding. */
const ITEM_PAGE = 200;

/** How many turns are read per request while looking for the one a repeated question started. */
const RESPONSE_PAGE = 100;

/** An image handed to the model alongside the question. */
export type ImageSource = Schemas["ImageSource"];

export interface CreateResponseOptions {
  images?: ImageSource[];
  /** Caps this answer, reasoning included, and the replies that finish it after its tools. */
  maxOutputTokens?: number;
  /** How long the model may think first: one of the efforts the session's model accepts. */
  reasoningEffort?: string;
  /**
   * What the person did in the UI an earlier reply showed, which the text says in their words:
   * the reply (`message_id`), the UI's format and the action (`name`), with what it carries
   * (`payload`), such as an `AIUIAction`'s JSON. Their message carries it as
   * `custom.ai_ui_action`, and the model is told it after the text. An empty one is not sent.
   */
  uiAction?: UIAction;
  /**
   * The id the router answers this question once under. Left out, a fresh one is made. Pass
   * the same one to ask again after a failure that may have reached the router, such as one
   * with status 0, so the agent does not answer the same question twice.
   */
  requestId?: string;
}

/** What the person did in a reply's generated UI, as the router takes it. */
export type UIAction = NonNullable<Schemas["CreateResponseRequest"]["ui_action"]>;

/**
 * What may go with a regenerate or an edit. Images and a UI action may not: the router refuses
 * them beside `replaces`, since the question asked again is text alone.
 */
export type BranchResponseOptions = Pick<CreateResponseOptions, "maxOutputTokens" | "reasoningEffort" | "requestId">;

/** A response named by its handle, its record, one of its items, or its id. */
export type ResponseRef = AgentResponse | Schemas["AgentResponse"] | Schemas["AgentResponseItem"] | string;

/**
 * Called before a regenerate, an edit or an activation is sent, and answering what to do with
 * the `rewound_to` it comes back with: the session drops the hidden messages from its channel.
 */
export type Branching = () => (rewoundTo: RewoundTo | undefined) => void;

/**
 * The things one turn was made of, in the order they happened.
 *
 * Read rather than watched: this is what the backend wrote down, so it is the same whether
 * the conversation is still going or ended last week. Deltas are not here — a hundred
 * fragments of one sentence are the sentence — so a caller who wants to watch words arrive
 * reads `session.events()` and a caller who wants the shape of a turn reads this.
 */
export class Items {
  private readonly client: Client;
  private readonly sessionId: string;
  /** Empty for every turn in the session, set for one response's own items. */
  private readonly responseId: string;

  constructor(client: Client, sessionId: string, responseId = "") {
    this.client = client;
    this.sessionId = sessionId;
    this.responseId = responseId;
  }

  /**
   * Yields every item, oldest first, fetching a page at a time.
   *
   * Paging is inside rather than outside because a conversation's length is not something
   * the caller chose: `for await (const item of session.responses.items.unwind())` reads a
   * turn or a thousand the same way.
   */
  async *unwind(options: { limit?: number } = {}): AsyncGenerator<Schemas["AgentResponseItem"]> {
    const limit = Math.min(options.limit ?? ITEM_PAGE, 1000);
    let cursor: string | undefined;
    while (true) {
      const page = await this.list({ limit, ...(cursor ? { cursor } : {}) });
      for (const item of page.items) {
        yield item;
      }
      if (!page.has_more || !page.next_cursor) {
        return;
      }
      cursor = page.next_cursor;
    }
  }

  /** One page of items, for a caller doing its own paging. */
  list(options: { limit?: number; cursor?: string } = {}): Promise<Schemas["AgentResponseItemPage"]> {
    return this.client.get("/v1/agents/sessions/{id}/responses/items", {
      path: { id: this.sessionId },
      query: {
        ...(this.responseId ? { response_id: this.responseId } : {}),
        limit: options.limit,
        cursor: options.cursor,
      },
    });
  }

  /** Everything in one array, for a conversation short enough to hold. */
  async all(): Promise<readonly Schemas["AgentResponseItem"][]> {
    const collected: Schemas["AgentResponseItem"][] = [];
    for await (const item of this.unwind()) {
      collected.push(item);
    }
    return collected;
  }
}

/**
 * One turn, and a way to read what it was made of.
 *
 * `create` returns as soon as the agent has started answering rather than when it has
 * finished, because a model takes seconds and a request that waited them out would time out
 * on anything worth asking. So this is a handle on an answer in progress: `items.unwind()`
 * reads what has been written down so far, and `session.events()` is what watches it arrive.
 */
export class AgentResponse {
  readonly created: Schemas["AgentResponse"];
  readonly items: Items;

  constructor(client: Client, created: Schemas["AgentResponse"]) {
    this.created = created;
    this.items = new Items(client, created.session_id, created.id);
  }

  /** The backend's id for this turn, empty for a session that records nothing. */
  get id(): string {
    return this.created.id;
  }

  get status(): Schemas["AgentResponse"]["status"] {
    return this.created.status;
  }
}

/**
 * A session's turns.
 *
 * `items` here is the whole conversation flattened, which is how a conversation reads and
 * how it gets rendered: the question, what the agent did about it, what it said, then the
 * next question. A single turn's items come off the handle `create` returns.
 */
export class Responses {
  readonly items: Items;

  private readonly client: Client;
  private readonly sessionId: string;
  private readonly asked: ((text: string) => () => void) | undefined;
  private readonly branching: Branching | undefined;

  /**
   * `asked` hears each question as it goes, for a transcript that shows it before the router
   * does, and returns what takes it back when the question could not be asked. `branching`
   * hears a regenerate, an edit or an activation, and where it rewound the channel to.
   */
  constructor(
    client: Client,
    sessionId: string,
    asked?: (text: string) => () => void,
    branching?: Branching,
  ) {
    this.client = client;
    this.sessionId = sessionId;
    this.asked = asked;
    this.branching = branching;
    this.items = new Items(client, sessionId);
  }

  /**
   * Asks the agent something and names the turn it answers as.
   *
   * Every question goes with a request id, a fresh one unless `requestId` names it, so a retry
   * of it is answered once; a failure that may not have reached the router is retried with the
   * same one. A personal conversation kept in Stream Chat needs one, and with it the images, as
   * data URIs, are uploaded to the channel and shown on the person's message.
   *
   * A question that could not be asked is taken back out of `session.conversation`. When the
   * router took the question but its answer was lost, the retry is answered with no id; the turn
   * is then looked up by what was asked, so the handle still names it.
   */
  async create(text: string, options: CreateResponseOptions = {}): Promise<AgentResponse> {
    const takeBack = this.asked?.(text);
    try {
      return await createResponse(this.client, this.sessionId, text, options);
    } catch (cause) {
      takeBack?.();
      throw cause;
    }
  }

  /**
   * Goes back to a response and carries on from there, as though nothing after it was said.
   *
   * The model forgets the later turns and they drop out of `list` and `items`. An item
   * stands for the response it belongs to, so the thing a transcript renders is enough to
   * rewind to. A conversation kept in Stream Chat is rewound there too: the later turns
   * are hidden in the channel and kept as a branch that can be brought back.
   */
  async rewind(to: ResponseRef): Promise<void> {
    const responseId = responseIdOf(to);
    if (!responseId) {
      throw new ConfigurationError("a response that was never recorded cannot be rewound to");
    }
    await this.client.post("/v1/agents/sessions/{id}/rewind", {
      path: { id: this.sessionId },
      body: { response_id: responseId },
    });
  }

  /** The turns so far, oldest first, a page at a time. */
  list(options: { limit?: number; cursor?: string } = {}): Promise<Schemas["AgentResponsePage"]> {
    return this.client.get("/v1/agents/sessions/{id}/responses", {
      path: { id: this.sessionId },
      query: { limit: options.limit, cursor: options.cursor },
    });
  }

  /**
   * Answers the question of an earlier response again, as a new response beside it, in a
   * conversation kept in Stream Chat. The reply and everything after it are hidden in Chat and
   * kept as a branch, which `siblings` lists, `branch` reads and `activate` brings back.
   *
   * Returns as `create` does, once the agent has started answering: the answer arrives the same
   * way, in the channel and through `session.events()`. Its handle's `created.branch` says where
   * it stands among its alternatives. The session's channel, once `session.chat()` has opened it,
   * drops what was hidden as soon as this returns, from `created.rewound_to`, so the old and the
   * new answer are never shown together.
   *
   * Refused with a `RouterError` whose code is `branch_limit_reached` once the conversation has
   * used its 100 edit and regenerate points, and with a 400 for a conversation that cannot have
   * branches: shared, in a thread, answered by a dispatch worker, on a call, or a response from
   * before branches or typed during a call.
   */
  regenerate(response: ResponseRef, options: BranchResponseOptions = {}): Promise<AgentResponse> {
    return this.replace(response, "", options);
  }

  /**
   * Asks `text` in place of the question of an earlier response, as a new question beside it.
   * The question, its reply and everything after them are hidden in Chat and kept as a branch.
   * Otherwise as `regenerate`.
   */
  async edit(response: ResponseRef, text: string, options: BranchResponseOptions = {}): Promise<AgentResponse> {
    if (!text) {
      throw new ConfigurationError("an edit asks text in place of the question: regenerate answers it again as it was");
    }
    return this.replace(response, text, options);
  }

  /**
   * A page of the alternatives to a response, oldest first, itself among them: with `of: "reply"`
   * (the default) the answers to its question, its regenerations; with `of: "question"` the
   * questions asked in its place, its edits, each named by the first response that answered it.
   * Each has its `index`, the i of ‹ i/n ›, and whether it is `active`, on the conversation now.
   * `signal` abandons the read.
   */
  async siblings(
    response: ResponseRef,
    options: { of?: "reply" | "question"; limit?: number; cursor?: string; signal?: AbortSignal } = {},
  ): Promise<Schemas["ResponseSiblingPage"]> {
    return this.client.get("/v1/agents/sessions/{id}/responses/{response_id}/siblings", {
      path: { id: this.sessionId, response_id: recorded(response, "listed the siblings of") },
      query: { of: options.of, limit: options.limit, cursor: options.cursor },
      ...(options.signal ? { signal: options.signal } : {}),
    });
  }

  /** Every alternative to a response, oldest first, fetching a page at a time. See `siblings`. */
  async *unwindSiblings(
    response: ResponseRef,
    options: { of?: "reply" | "question"; limit?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<Schemas["ResponseSibling"]> {
    let cursor: string | undefined;
    while (true) {
      const page = await this.siblings(response, { ...options, ...(cursor ? { cursor } : {}) });
      yield* page.items;
      if (!page.has_more || !page.next_cursor) {
        return;
      }
      cursor = page.next_cursor;
    }
  }

  /**
   * A page of the messages a hidden branch shows, oldest first, from where it leaves the
   * conversation down to the newest response under `response`, shaped as Stream Chat messages
   * so they render in place of the channel's own. Nothing is written to Chat and the model is
   * not told: this is what a ‹ i/n › control shows. A response on the conversation answers
   * `active: true` and no items, since the channel already shows it. A branch whose copies were
   * deleted is a `RouterError` with code `branch_unavailable`. `signal` abandons the read.
   */
  async branch(
    response: ResponseRef,
    options: { limit?: number; cursor?: string; signal?: AbortSignal } = {},
  ): Promise<Schemas["BranchMessagePage"]> {
    return this.client.get("/v1/agents/sessions/{id}/responses/{response_id}/branch", {
      path: { id: this.sessionId, response_id: recorded(response, "read the branch of") },
      query: { limit: options.limit, cursor: options.cursor },
      ...(options.signal ? { signal: options.signal } : {}),
    });
  }

  /** Every message of a hidden branch, oldest first, fetching a page at a time. See `branch`. */
  async *unwindBranch(
    response: ResponseRef,
    options: { limit?: number; signal?: AbortSignal } = {},
  ): AsyncGenerator<Schemas["BranchMessage"]> {
    let cursor: string | undefined;
    while (true) {
      const page = await this.branch(response, { ...options, ...(cursor ? { cursor } : {}) });
      yield* page.items;
      if (page.active || !page.has_more || !page.next_cursor) {
        return;
      }
      cursor = page.next_cursor;
    }
  }

  /**
   * Makes the hidden branch of `response` the conversation again: call it before asking
   * anything while a hidden branch is shown. The branch on the conversation from where the two
   * part is hidden and kept, and this one is sent again, ending at the newest response under
   * `response`, as new messages with new ids that carry `custom.resent_from` and fire
   * `message.new` again. A branch already on the conversation changes nothing
   * (`already_active`), so a retry is safe. The session's channel drops what was hidden as
   * `regenerate` does.
   *
   * At most 100 messages are sent again; a longer branch is a `RouterError` with code
   * `branch_too_long`, and a conversation with no room for another hidden range
   * `branch_limit_reached`.
   */
  async activate(response: ResponseRef): Promise<Schemas["ActivatedBranch"]> {
    const responseId = recorded(response, "activated");
    const rewound = this.branching?.();
    const activated = await this.client.post("/v1/agents/sessions/{id}/responses/{response_id}/activate", {
      path: { id: this.sessionId, response_id: responseId },
      // Activating what is already active changes nothing, so asking again is safe.
      idempotent: true,
    });
    rewound?.(activated.rewound_to);
    return activated;
  }

  /** A regenerate without text, an edit with it. */
  private async replace(response: ResponseRef, text: string, options: BranchResponseOptions): Promise<AgentResponse> {
    const replaces = recorded(response, text ? "edited" : "regenerated");
    const rewound = this.branching?.();
    const created = await createResponse(this.client, this.sessionId, text, { ...options, replaces });
    rewound?.(created.created.rewound_to);
    return created;
  }
}

/** The id of the response `ref` names, or the empty string for one that was never recorded. */
function responseIdOf(ref: ResponseRef): string {
  return typeof ref === "string" ? ref : "response_id" in ref ? ref.response_id : ref.id;
}

/** The id of a response that has one, or the error saying a response with none cannot be `done`. */
function recorded(ref: ResponseRef, done: string): string {
  const id = responseIdOf(ref);
  if (!id) {
    throw new ConfigurationError(
      `a response that was never recorded cannot be ${done}: a session that records nothing hands back no id`,
    );
  }
  return id;
}

/**
 * Creates a response under `options.requestId`, or under a fresh one.
 *
 * Not exported from the package: a dispatch worker answering an inbound message calls it with
 * that message's own request id, so the answer lands on it.
 */
export async function createResponse(
  client: Client,
  sessionId: string,
  text: string,
  options: CreateResponseOptions & { replaces?: string } = {},
): Promise<AgentResponse> {
  const id = options.requestId || crypto.randomUUID();
  let resent = false;
  const created = await client.post("/v1/agents/sessions/{id}/responses", {
    path: { id: sessionId },
    // The request id is what the router answers once by, so sending it again is safe.
    idempotent: true,
    ...onRetry(() => {
      resent = true;
    }),
    body: {
      text,
      ...(options.images?.length ? { images: options.images } : {}),
      ...(options.maxOutputTokens !== undefined ? { max_output_tokens: options.maxOutputTokens } : {}),
      ...(options.reasoningEffort ? { reasoning_effort: options.reasoningEffort } : {}),
      ...(options.uiAction && Object.keys(options.uiAction).length ? { ui_action: options.uiAction } : {}),
      ...(options.replaces ? { replaces: options.replaces } : {}),
      request_id: id,
    },
  });
  // A regenerate asks nothing to look its turn up by, and an edit's text may have been asked
  // before, so a lost answer to either is handed back as the router gave it, with no id.
  if (created.id || !resent || options.replaces) {
    return new AgentResponse(client, created);
  }
  // The router answers a question it already took with no id: the first send reached it and
  // only its answer was lost. The turn that send started is the latest one asking this.
  return new AgentResponse(client, (await startedBy(client, sessionId, text)) ?? created);
}

/**
 * The latest turn on the session that asked `text`, or nothing when there is none or the
 * turns could not be read: the question was asked either way, so failing to name its turn
 * does not fail the asking.
 */
async function startedBy(
  client: Client,
  sessionId: string,
  text: string,
): Promise<Schemas["AgentResponse"] | undefined> {
  let latest: Schemas["AgentResponse"] | undefined;
  let cursor: string | undefined;
  try {
    do {
      const page = await client.get("/v1/agents/sessions/{id}/responses", {
        path: { id: sessionId },
        query: { limit: RESPONSE_PAGE, cursor },
      });
      for (const turn of page.items) {
        if (turn.said === text) {
          latest = turn;
        }
      }
      cursor = page.has_more ? page.next_cursor : undefined;
    } while (cursor);
  } catch {
    return undefined;
  }
  return latest;
}
