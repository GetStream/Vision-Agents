import type { Client, Schemas } from "./client.js";
import { checkTeam } from "./backend.js";
import { ConfigurationError } from "./errors.js";
import { Responses } from "./responses.js";
import { Session, type ForkOptions, type SessionOptions } from "./session.js";

/**
 * What a session can be opened with, beyond which agent is holding it.
 *
 * Everything the wire takes is here: the labels a person finds the conversation by again,
 * the models this one conversation overrules, and `incognito` for one that is not to be
 * found again at all. The names are the wire's names, with the two-word ones spelled the
 * way the rest of the SDK spells them.
 */
export interface SessionSpec
  extends Omit<
    Schemas["CreateSessionRequest"],
    "agent" | "config_id" | "model_overwrites" | "tools" | "timezone"
  > {
  /**
   * The app's own page a connector's sign-in goes back to, as `returnURL` and `returnUrl` are on
   * iOS and Android: https, or http on localhost, of the page that opens the sign-in. The same as
   * the wire's `return_url`, which this wins over when both are given.
   */
  returnUrl?: string;
  /** What to change about the models for this conversation alone. */
  modelOverwrites?: Schemas["ModelOverwrites"];
  /**
   * The person's IANA zone, such as `Europe/Lisbon`. The agent tells the date and the hour in
   * it, and for a user the caller may name it is remembered as theirs for the scheduled tasks
   * and sessions that follow. A zone the router does not know is refused with a 400
   * `invalid_timezone`, and the session is not opened.
   *
   * Left out in a page, it is the zone the browser reports for this device, sent only when it
   * reads as a zone name: `UTC`, or one with a slash in it. Left out anywhere else, such as on
   * a server, a React Native app or a web worker, none is sent, because the zone of the
   * machine running the code is not known to be the caller's; an app there that knows the
   * person's zone passes it. An empty string sends none anywhere, which leaves the user's
   * stored zone, then the agent config's default, to decide.
   */
  timezone?: string;
  /**
   * The team the conversation's work is counted under, its team spend limits included, for a
   * backend opening one for a team: 1 to 59 letters, digits, _ . and -. Left out, it is the
   * team the caller's token names, else the agent's `tags: team`. A token signed for another
   * team is refused with a 400. Server side only: a device sending one is refused with a 403.
   */
  team?: string;
}

/** Opening a session, plus how it is watched. */
export interface CreateSessionOptions extends SessionSpec, SessionOptions {}

/**
 * Which of an agent's old conversations to list, as the wire spells it: the keys of one object
 * are ANDed, and `$and` and `$or` each take 1 to 10 filters and nest up to 3 deep. These are
 * all the fields there are; the router refuses any other key with a 400.
 *
 * ```ts
 * const filter: SessionFilter = {
 *   $or: [{ project_id: "health" }, { project_id: "billing" }],
 * };
 * ```
 *
 * One package serves a page and a server, so `review_score` and `user_id` are typed here for
 * both; only a server-side caller may use them.
 */
export interface SessionFilter {
  /** Every one of these filters holds. */
  $and?: readonly SessionFilter[];
  /**
   * At least one of these filters holds. It chooses among the sessions the caller may see and
   * never adds to them. `text` and `user_id` are refused inside it.
   */
  $or?: readonly SessionFilter[];
  /**
   * The agent name the session was opened against. A query through `agent.sessions` always
   * sets the top-level one to its own agent, whatever is put here.
   */
  agent?: string | { readonly $eq: string };
  /** The project the session belongs to: one, as a string or `$eq`, or any of up to 50 as `$in`. */
  project_id?: string | { readonly $eq: string } | { readonly $in: readonly string[] };
  /**
   * Server side only: the score from 1 to 5 the review gave the conversation, as a number or
   * as `$eq`, `$gt`, `$gte`, `$lt` and `$lte`, every one of which has to hold. A session nobody
   * reviewed matches none. A device asking for it is refused.
   */
  review_score?:
    | number
    | {
        readonly $eq?: number;
        readonly $gt?: number;
        readonly $gte?: number;
        readonly $lt?: number;
        readonly $lte?: number;
      };
  /**
   * Full text over the title, description, project and agent name, which `sessions.search`
   * sets. Once per filter, at its top level or in a top-level `$and`, and never inside `$or`.
   */
  text?: { readonly $q: string };
  /**
   * Server side only: whose sessions to list. An end user's token is already narrowed to their
   * own. Only at the top level of the filter or in a top-level `$and`.
   */
  user_id?: string | { readonly $eq: string };
}

/**
 * What a query can be ordered by. `last_message_at` is when the agent last answered, or when
 * the session began. `pinned` lists the pinned sessions first, descending only. `relevance` is
 * the order of a text search and nothing else. `review_score` is server side only.
 */
export type SessionSortField = Schemas["SessionSortField"];

/**
 * One step of a query's order. `direction` is -1 for descending, which it is when left out,
 * or 1 for ascending.
 *
 * The orders the router takes: `[{ field: "last_message_at" }]`; `[{ field: "pinned" }]`,
 * alone or followed by `last_message_at`; `[{ field: "review_score" }]`, server side only;
 * and `[{ field: "relevance" }]`, only with a `text` filter.
 */
export interface SessionSort {
  field: SessionSortField;
  direction?: -1 | 1;
}

/** Which of an agent's old conversations to list, in what order, a page at a time. */
export interface SessionQuery {
  /** Which sessions. Omitted is every one of the agent's the caller may see. */
  filter?: SessionFilter;
  /** The order. Omitted is `last_message_at`, newest first. */
  sort?: readonly SessionSort[];
  /** Up to 200. Omitted is 25. */
  limit?: number;
  /** The `next_cursor` of the page before, sent with the same filter and sort. */
  cursor?: string;
}

/** A search's filter and paging. It takes no sort: a search is ordered by relevance. */
export type SessionSearch = Omit<SessionQuery, "sort">;

/**
 * An agent's conversations: the one being held and the ones that were.
 *
 * `query` filters, sorts and pages; `search` reads the words a caller titled and described
 * their conversations with, best match first, narrowed by the same filter.
 */
export class Sessions {
  private readonly client: Client;
  /** The agent every call here is about, which is what makes this the agent's sessions. */
  private readonly agent: string;

  constructor(client: Client, agent: string) {
    this.client = client;
    this.agent = agent;
  }

  /**
   * Opens a conversation and starts watching it.
   *
   * It resolves once the backend is holding the conversation, so a session that has opened
   * is one that is already listening. Without `start_voice` it is held in writing, which is
   * what the resource surface is mostly for: a conversation somebody comes back to.
   *
   * Opened from a page, the session is in the device's own zone unless `timezone` says
   * otherwise.
   */
  async create(options: CreateSessionOptions = {}): Promise<Session> {
    const { tools, interim, decisions, watch, modelOverwrites, timezone, memory, returnUrl, team, ...rest } =
      options;
    const zone = timezone ?? deviceTimezone();
    const request: Schemas["CreateSessionRequest"] = {
      agent: this.agent,
      ...rest,
      ...(modelOverwrites ? { model_overwrites: modelOverwrites } : {}),
      ...(zone ? { timezone: zone } : {}),
      ...(memory ? { memory: this.memoryOf(memory) } : {}),
      ...(returnUrl ? { return_url: returnUrl } : {}),
      ...(team ? { team: checkTeam(team) } : {}),
    };
    return Session.open(this.client, request, watching({ tools, interim, decisions, watch }));
  }

  /**
   * Carries on a conversation held in writing, by the id of the session it was held in, and
   * starts watching it. One that ended is reopened with what was said in it.
   */
  async resume(id: string, options: SessionOptions = {}): Promise<Session> {
    return Session.watching(this.client, await this.get(id), watching(options));
  }

  /**
   * The agent's conversations, the ones that ended included: by default the one the agent
   * answered last comes first.
   *
   * ```ts
   * const page = await agent.sessions.query({
   *   filter: { $or: [{ project_id: "health" }, { project_id: "billing" }] },
   *   sort: [{ field: "pinned" }, { field: "last_message_at" }],
   * });
   * ```
   *
   * What comes back are the rows rather than live handles: reading a conversation back is
   * not the same as holding one, and most of these are over. `responses` reads the turns of
   * one, and `resume` carries one on. A `cursor` is only good with the sort it came from.
   */
  query(query: SessionQuery = {}): Promise<Schemas["SessionPage"]> {
    return this.client.post("/v1/agents/sessions/query", { body: this.queryOf(query) });
  }

  /**
   * Finds a conversation by what it was called, best match first.
   *
   * It reads the title, the description, the project and the agent name. The filter narrows
   * it as it narrows a query, and must not carry a `text` of its own. Nothing about an
   * incognito session is searchable, because nothing about it was written down.
   */
  async search(text: string, search: SessionSearch = {}): Promise<Schemas["SessionPage"]> {
    if (search.filter?.text !== undefined) {
      throw new ConfigurationError("search sets the filter's text itself: leave text out of the filter");
    }
    if ("sort" in search && (search as SessionQuery).sort !== undefined) {
      throw new ConfigurationError("search takes no sort: a text search is ordered by relevance");
    }
    const { filter, limit, cursor } = search;
    return this.client.post("/v1/agents/sessions/query", {
      body: this.queryOf({
        filter: { ...filter, text: { $q: text } },
        ...(limit === undefined ? {} : { limit }),
        ...(cursor === undefined ? {} : { cursor }),
      }),
    });
  }

  /**
   * The turns of a conversation this process is not holding, and what each was made of.
   *
   * Which is most of them: a page rendering a conversation from last week has its id and no
   * session. The same thing a held session offers as `session.responses`, so what renders a
   * live conversation renders an old one.
   */
  responses(id: string): Responses {
    return new Responses(this.client, id);
  }

  /** One conversation, whether or not it is still being held. */
  get(id: string): Promise<Schemas["Session"]> {
    return this.client.get("/v1/agents/sessions/{id}", { path: { id } });
  }

  /**
   * Renames, relabels, pins or unpins a conversation this process is not holding, running or
   * ended, and answers it as it now is. A field left out is left as it is; `custom` replaces
   * the labels whole; `pinned` sets or clears its `pinned_at`. `session.update` does the same
   * for one that is held.
   */
  update(id: string, changes: Schemas["UpdateSessionRequest"]): Promise<Schemas["Session"]> {
    return this.client.patch("/v1/agents/sessions/{id}", { path: { id }, body: changes });
  }

  /**
   * Pins a conversation, which a query sorted by `pinned` lists first, and answers it with
   * its `pinned_at`. It moves nothing else, and one already pinned keeps the time it was first
   * pinned. A person may pin their own; an incognito session cannot be pinned.
   */
  pin(id: string): Promise<Schemas["Session"]> {
    return this.update(id, { pinned: true });
  }

  /** Unpins a conversation, and answers it without a `pinned_at`. */
  unpin(id: string): Promise<Schemas["Session"]> {
    return this.update(id, { pinned: false });
  }

  /**
   * Has the agent join a conversation's call, `agent:<session id>`, and carry it on there.
   * Starting voice that is already on does nothing.
   */
  startVoice(id: string): Promise<Schemas["Session"]> {
    return this.client.post("/v1/agents/sessions/{id}/voice", { path: { id } });
  }

  /**
   * Takes the agent off a conversation's call and carries it on in writing. Stopping voice
   * that is off does nothing.
   */
  stopVoice(id: string): Promise<Schemas["Session"]> {
    return this.client.delete("/v1/agents/sessions/{id}/voice", { path: { id } });
  }

  /**
   * Stops a conversation, which is how the agent leaves. What it recorded and remembered is
   * kept; `delete` takes it away.
   */
  async close(id: string): Promise<void> {
    await this.client.post("/v1/agents/sessions/{id}/stop", { path: { id } });
  }

  /**
   * Continues a conversation as a new one, leaving the parent as it was, and starts watching
   * the fork. `session.fork` does the same for one that is held, and passes its tools on.
   */
  async fork(id: string, options: ForkOptions = {}): Promise<Session> {
    const { tools, interim, decisions, watch, modelOverwrites, ...rest } = options;
    const forked = await this.client.post("/v1/agents/sessions/{id}/fork", {
      path: { id },
      body: { ...rest, ...(modelOverwrites ? { model_overwrites: modelOverwrites } : {}) },
    });
    return Session.startWatching(this.client, forked, watching({ tools, interim, decisions, watch }));
  }

  /**
   * Deletes a conversation, running or ended: it is stopped, and its turns and what it
   * remembered are deleted with it. The user's other memories are kept.
   */
  delete(id: string): Promise<void> {
    return this.client.delete("/v1/agents/sessions/{id}", { path: { id } });
  }

  /**
   * Deletes what one conversation remembered, running or ended, and leaves the rest of the
   * user's memories alone. Server side only.
   */
  deleteMemories(id: string): Promise<void> {
    return this.client.delete("/v1/agents/sessions/{id}/memories", { path: { id } });
  }

  private memoryOf(memory: Schemas["SessionMemory"]): Schemas["SessionMemory"] {
    return memoryFor(this.client, memory);
  }

  /**
   * The query as it is sent, with the agent's own name always the filter's top-level `agent`.
   * The caller's filter is copied rather than wrapped in an `$and`, because `text` and
   * `user_id` have to stay at the top level.
   */
  private queryOf(query: SessionQuery): Schemas["SessionQuery"] {
    return {
      filter: { ...query.filter, agent: this.agent },
      ...(query.sort ? { sort: query.sort } : {}),
      ...(query.limit ? { limit: query.limit } : {}),
      ...(query.cursor ? { cursor: query.cursor } : {}),
    };
  }
}

/**
 * The memory a session is opened with, for whom it is remembered.
 *
 * A device remembers for the user `setUser` named and nobody else, whatever `user_id` it was
 * handed, since the router refuses a device naming another user. A backend may name any of its
 * users, and names the one it acts for when it names none.
 */
function memoryFor(
  client: Client,
  memory: Schemas["SessionMemory"],
): Schemas["SessionMemory"] {
  const backend = client.backend;
  const user = backend.serverSide ? memory.user_id || backend.actsFor || backend.userId : backend.userId;
  if (!user) {
    throw new ConfigurationError(
      "memory is remembered for a user: call setUser first, or on a server name user_id",
    );
  }
  return { ...memory, user_id: user };
}

/**
 * The zone this device is in, or the empty string when there is none to trust.
 *
 * Only a page has one: there the code runs on the person's own device, so the zone the browser
 * reports is theirs. Anywhere else it is the zone of whatever machine the code runs on, often
 * a server's UTC, and sending it for a verified user would store it as theirs.
 *
 * What the browser reports is sent only when it reads as a zone name, `UTC` or one with a
 * slash in it, and never `Etc/Unknown`, which is what ICU says when it could not tell. A
 * zone the router does not know fails the session, and a default the app never chose must
 * not be what fails it.
 */
function deviceTimezone(): string {
  if ((globalThis as { document?: unknown }).document === undefined) {
    return "";
  }
  let zone: unknown;
  try {
    zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  } catch {
    return "";
  }
  if (typeof zone !== "string" || zone === "Etc/Unknown") {
    return "";
  }
  return zone === "UTC" || zone.includes("/") ? zone : "";
}

/** How a session is watched, with what was left out left out. */
function watching(options: {
  tools?: SessionOptions["tools"] | undefined;
  interim?: boolean | undefined;
  decisions?: boolean | undefined;
  watch?: boolean | undefined;
}): SessionOptions {
  const { tools, interim, decisions, watch } = options;
  return {
    ...(tools ? { tools } : {}),
    ...(interim === undefined ? {} : { interim }),
    ...(decisions === undefined ? {} : { decisions }),
    ...(watch === undefined ? {} : { watch }),
  };
}
