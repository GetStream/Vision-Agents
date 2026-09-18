import type { Client, Schemas } from "./client.js";
import { Responses } from "./responses.js";
import { Session, type SessionOptions } from "./session.js";

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
    "agent" | "config_id" | "model_overwrites" | "tools"
  > {
  /** What to change about the models for this conversation alone. */
  modelOverwrites?: Schemas["ModelOverwrites"];
}

/** Opening a session, plus how it is watched. */
export interface CreateSessionOptions extends SessionSpec, SessionOptions {}

/** Which of an agent's old conversations to list. */
export interface SessionQuery {
  /** Only this project's. */
  project?: string;
  /** Only this user's, which only a backend may ask for: a token is already narrowed. */
  userId?: string;
  /** `running` or `closed`. Omitted is both. */
  state?: "running" | "closed";
  /** Labels a session must carry, all of them. */
  custom?: Record<string, string | number | boolean>;
  createdAfter?: Date | string;
  createdBefore?: Date | string;
  /** Up to 200. Omitted is 25. */
  limit?: number;
  offset?: number;
}

/**
 * An agent's conversations: the one being held and the ones that were.
 *
 * `query` filters and pages; `search` reads the words a caller titled and described their
 * conversations with. Two methods rather than one that does both, because a filter and a
 * phrase combine by narrowing and there is no sensible ranking of an empty phrase.
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
   * is one that is already listening. Without a `call_id` it is held in writing, which is
   * what the resource surface is mostly for: a conversation somebody comes back to.
   */
  create(options: CreateSessionOptions = {}): Promise<Session> {
    const { tools, interim, decisions, watch, modelOverwrites, ...rest } = options;
    const request: Schemas["CreateSessionRequest"] = {
      agent: this.agent,
      // Held in writing unless a call was named. A session resource is a conversation, and
      // a caller who wants one on a call says which call.
      ...(rest.call_id ? {} : { text: true }),
      ...rest,
      ...(modelOverwrites ? { model_overwrites: modelOverwrites } : {}),
    };
    return Session.open(this.client, request, {
      ...(tools ? { tools } : {}),
      ...(interim === undefined ? {} : { interim }),
      ...(decisions === undefined ? {} : { decisions }),
      ...(watch === undefined ? {} : { watch }),
    });
  }

  /**
   * The agent's conversations, newest first, the ones that ended included.
   *
   * What comes back are the rows rather than live handles: reading a conversation back is
   * not the same as holding one, and most of these are over. `responses` reads the turns of
   * one, and `create({ conversation_id })` opens a new conversation on its transcript.
   */
  query(query: SessionQuery = {}): Promise<readonly Schemas["Session"][]> {
    return this.client.get("/v1/agents/sessions", { query: this.filter(query) });
  }

  /**
   * Finds a conversation by what it was called.
   *
   * It reads the title, the description and the opening question, which is what a person
   * remembers a conversation by. Nothing about an incognito session is searchable, because
   * nothing about it was written down.
   */
  search(text: string, query: SessionQuery = {}): Promise<readonly Schemas["Session"][]> {
    return this.client.get("/v1/agents/sessions/search", {
      query: { ...this.filter(query), q: text },
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

  /** The query as the wire spells it, with the agent's own name always in it. */
  private filter(query: SessionQuery): Record<string, string | number | undefined> {
    return {
      agent: this.agent,
      project: query.project,
      user_id: query.userId,
      state: query.state,
      custom: query.custom ? JSON.stringify(query.custom) : undefined,
      created_after: timestamp(query.createdAfter),
      created_before: timestamp(query.createdBefore),
      limit: query.limit,
      offset: query.offset,
    };
  }
}

/** A moment as the wire takes it, so a caller can pass a Date or the string itself. */
export function timestamp(at: Date | string | undefined): string | undefined {
  if (at === undefined) {
    return undefined;
  }
  return at instanceof Date ? at.toISOString() : at;
}
