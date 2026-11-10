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

  /**
   * `asked` hears each question as it goes, for a transcript that shows it before the router
   * does, and returns what takes it back when the question could not be asked.
   */
  constructor(client: Client, sessionId: string, asked?: (text: string) => () => void) {
    this.client = client;
    this.sessionId = sessionId;
    this.asked = asked;
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
   * rewind to. A conversation kept in Stream Chat cannot be rewound, because the channel
   * would still hold the later turns: fork it at the response instead.
   */
  async rewind(
    to: AgentResponse | Schemas["AgentResponse"] | Schemas["AgentResponseItem"] | string,
  ): Promise<void> {
    const responseId =
      typeof to === "string" ? to : "response_id" in to ? to.response_id : to.id;
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
  options: CreateResponseOptions = {},
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
      request_id: id,
    },
  });
  if (created.id || !resent) {
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
