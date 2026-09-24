import type { Client, Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** How many items are read per request while unwinding. */
const ITEM_PAGE = 200;

/** An image handed to the model alongside the question. */
export type ImageSource = Schemas["ImageSource"];

export interface CreateResponseOptions {
  images?: ImageSource[];
}

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
    const page = Math.min(options.limit ?? ITEM_PAGE, 1000);
    let offset = 0;
    while (true) {
      const items = await this.list({ limit: page, offset });
      for (const item of items) {
        yield item;
      }
      // A short page is the last page. Asking again to see an empty one would double the
      // requests for every conversation that happens to be a multiple of the page size,
      // which is not worth avoiding one extra round trip in the rare exact-fit case.
      if (items.length < page) {
        return;
      }
      offset += items.length;
    }
  }

  /** One page of items, for a caller doing its own paging. */
  list(options: { limit?: number; offset?: number } = {}): Promise<readonly Schemas["AgentResponseItem"][]> {
    return this.client.get("/v1/agents/sessions/{id}/responses/items", {
      path: { id: this.sessionId },
      query: {
        ...(this.responseId ? { response_id: this.responseId } : {}),
        limit: options.limit,
        offset: options.offset,
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

  constructor(client: Client, sessionId: string) {
    this.client = client;
    this.sessionId = sessionId;
    this.items = new Items(client, sessionId);
  }

  /** Asks the agent something and names the turn it answers as. */
  async create(text: string, options: CreateResponseOptions = {}): Promise<AgentResponse> {
    const created = await this.client.post("/v1/agents/sessions/{id}/responses", {
      path: { id: this.sessionId },
      body: { text, ...(options.images?.length ? { images: options.images } : {}) },
    });
    return new AgentResponse(this.client, created);
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

  /** The turns so far, oldest first. */
  list(options: { limit?: number; offset?: number } = {}): Promise<readonly Schemas["AgentResponse"][]> {
    return this.client.get("/v1/agents/sessions/{id}/responses", {
      path: { id: this.sessionId },
      query: { limit: options.limit, offset: options.offset },
    });
  }
}
