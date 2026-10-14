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
  /** Only this project's. A search covers every project, so the router refuses it on one. */
  projectId?: string;
  /** Only this user's, which only a backend may ask for: a token is already narrowed. */
  userId?: string;
  /** How the user took part. Omitted is every way. */
  modality?: Schemas["SessionModality"];
  /** `live` or `ended`. Omitted is both. */
  state?: Schemas["SessionState"];
  /** Only the ones created with this agent id, which names their transcript channel. */
  agentId?: string;
  /** Up to 200. Omitted is 25. */
  limit?: number;
  /** The `next_cursor` of the page before, sent with the same filters. */
  cursor?: string;
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
   * is one that is already listening. Without `start_voice` it is held in writing, which is
   * what the resource surface is mostly for: a conversation somebody comes back to.
   */
  create(options: CreateSessionOptions = {}): Promise<Session> {
    const { tools, interim, decisions, watch, modelOverwrites, ...rest } = options;
    const request: Schemas["CreateSessionRequest"] = {
      agent: this.agent,
      ...rest,
      ...(modelOverwrites ? { model_overwrites: modelOverwrites } : {}),
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
   * The agent's conversations, newest first, the ones that ended included.
   *
   * What comes back are the rows rather than live handles: reading a conversation back is
   * not the same as holding one, and most of these are over. `responses` reads the turns of
   * one, and `resume` carries one on.
   */
  query(query: SessionQuery = {}): Promise<Schemas["SessionPage"]> {
    return this.client.post("/v1/agents/sessions/query", { body: this.queryOf("", query) });
  }

  /**
   * Finds a conversation by what it was called, best match first.
   *
   * It reads the title, the description and the opening question, which is what a person
   * remembers a conversation by. Nothing about an incognito session is searchable, because
   * nothing about it was written down.
   */
  search(text: string, query: SessionQuery = {}): Promise<Schemas["SessionPage"]> {
    return this.client.post("/v1/agents/sessions/query", { body: this.queryOf(text, query) });
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

  /** The query as the wire spells it, with the agent's own name always in it. */
  private queryOf(text: string, query: SessionQuery): Schemas["SessionQuery"] {
    return {
      filter: {
        agent: this.agent,
        ...(query.projectId ? { project_id: query.projectId } : {}),
        ...(query.userId ? { user_id: query.userId } : {}),
        ...(query.modality ? { modality: query.modality } : {}),
        ...(query.state ? { state: query.state } : {}),
        ...(query.agentId ? { agent_id: query.agentId } : {}),
        ...(text ? { text: { $q: text } } : {}),
      },
      ...(query.limit ? { limit: query.limit } : {}),
      ...(query.cursor ? { cursor: query.cursor } : {}),
    };
  }
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
