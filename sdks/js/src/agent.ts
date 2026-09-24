import type { BackendOptions } from "./backend.js";
import { Client, type Schemas } from "./client.js";
import { DEFAULT_CALL_TYPE, Edge, type Call } from "./edge.js";
import { ConfigurationError, RouterError } from "./errors.js";
import { Session, type SessionOptions } from "./session.js";
import { Tools } from "./tools.js";
import type { InboundCall, InboundMessage } from "./inbound.js";

/** The memory filter key naming who the memories are about. Everything else narrows recall. */
export const USER_KEY = "user_id";

/** Where code the agent writes gets run. */
export interface Sandbox {
  provider: Schemas["Sandbox"];
}

/** A Daytona sandbox. The backend needs `DAYTONA_API_KEY` for it to do anything. */
export function daytona(): Sandbox {
  return { provider: "daytona" };
}

/**
 * A kind of work worth handing to the slower model.
 *
 * There is nothing behind a skill but a better model and more time. What it declares is the
 * description the fast model chooses by, and the instructions the slow one answers under.
 */
export interface Skill {
  name: string;
  /** The one line the fast model sees. */
  description: string;
  /** The full prompt, which only the subagent sees. */
  instructions: string;
  captureVideo?: boolean;
  /** How long the work may run before it is abandoned. Zero leaves the backend's default. */
  deadlineMs?: number;
}

/**
 * What stands between what a caller said and the model that answers them.
 *
 * The loop runs in the backend, so this is configuration rather than behaviour: it is
 * serialized into the session and the decisions are taken there.
 */
export interface Harness {
  /** Offer the backend's built-in skills. Naming skills of your own replaces them. */
  useSkills?: boolean;
  /** Model targets for the work handed over. The one under `default` runs the skills. */
  subagents?: Record<string, string>;
  /** Where delegated code runs. */
  vm?: Sandbox;
  /** Skills of your own, replacing the built-in set. */
  skills?: Skill[];
  /** How much delegated work may run at once. */
  tasks?: number;
}

/**
 * Which models hold the conversation.
 *
 * Every target is a provider/model name or a capability shortcut such as `llm-fast`;
 * leaving one out takes the backend's default for that modality.
 */
export interface Pipeline {
  /** A stored agent config to start from, named either by its id or by its name. */
  config?: string;
  llm?: string;
  stt?: string;
  tts?: string;
  /** A speech-to-speech target. Naming one means no transcriber or voice is opened. */
  sts?: string;
  /** The model that does the thinking a harness delegates. */
  subagent?: string;
  /** A provider-specific voice id. */
  voice?: string;
  /** A language hint, which narrows the candidates in every modality. */
  language?: string;
  /** Said on joining without going through the model. Empty waits to be spoken to. */
  greeting?: string;
  /** Murmur while a caller is still talking, the way a person does. */
  backchannel?: boolean;
  maxTokens?: number;
  toolTimeoutMs?: number;
  video?: Schemas["SessionVideo"];
}

/**
 * What an agent directory's agent.yaml declares: what the agent is called and what it runs
 * on. A setting left out leaves whatever the stored config has.
 */
export interface Declaration {
  name?: string;
  description?: string;
  mode?: Schemas["AgentMode"];
  stt?: string;
  tts?: string;
  /** An empty string turns speech-to-speech off, which is different from saying nothing. */
  sts?: string;
  voice?: string;
  llm?: string;
  subagent?: string;
  search?: string;
  greeting?: string;
  sandbox?: Schemas["Sandbox"];
  plugins?: string[];
  keyterms?: string[];
  tags?: Record<string, string>;
  video?: Schemas["SessionVideo"];
}

/**
 * Where a directory records the fingerprint it was last synced under.
 *
 * Handed over by whatever read the directory, so writing `.agent_sync` stays with the code
 * that can reach a filesystem and `sync` stays runnable wherever `fetch` is.
 */
export interface SyncStamp {
  /** The fingerprint last synced, or empty when there is none. */
  read(): Promise<string>;
  write(hash: string): Promise<void>;
}

/** An agent directory, as `loadFolder` from `@stream-io/vision-agents/node` reads one. */
export interface Folder {
  path: string;
  name: string;
  /** What agent.yaml declares. */
  settings?: Declaration;
  instructions: string;
  guardrail: string;
  skills: Skill[];
  knowledge: { source: string; text: string }[];
  knowledgeURLs: { url: string; title?: string; description?: string }[];
  /** `.agent_sync`, so a sync of a directory nothing has touched asks the router nothing. */
  stamp?: SyncStamp;
}

export interface AgentOptions {
  /** What the agent is called. It names the stored config and is who it appears as. */
  name?: string;
  /** The system prompt. */
  instructions?: string;
  /**
   * A guardrail.md: frontmatter saying how a turn is screened, then the policy in prose.
   *
   * It is enforced in the backend, so a turn the policy refuses never reaches the model.
   */
  guardrail?: string;
  /** An agent directory, whose contents fill in whatever is left out here. */
  folder?: Folder;
  /** Which models hold the conversation. */
  pipeline?: Pipeline;
  harness?: Harness;
  /** Labels every request the session makes, so spend can be attributed. */
  costTracking?: Record<string, string>;
  /** Who the memories are about, under `user_id`, and what narrows recall. */
  memoryFilter?: Record<string, string>;
  /** Who the agent joins a call as. Empty is derived from the name. */
  userId?: string;
  /** The caller's own functions. One is created when none is passed. */
  tools?: Tools;
  /** The router this agent talks to. */
  client?: Client | BackendOptions;
  /** Creates the Stream calls the backend joins. Built from the environment when needed. */
  edge?: Edge;
}

/** How a text conversation is held. */
export interface ChatOptions extends SessionOptions {
  /** Keep the conversation in Stream Chat, creating a channel when no id is given. */
  persist?: boolean;
  conversationId?: string;
  /**
   * The conversation being answered, which names the channel replies are written into.
   *
   * A worker answering several conversations has to set it. Left empty they all join under
   * the agent's user id, and a message arriving for one reaches whichever started last.
   */
  agentId?: string;
}

/**
 * A configured agent, before and between the calls it holds.
 *
 * An agent here is configuration and function calling. The conversation itself — joining
 * the call, hearing the caller, answering and speaking — happens in the backend, and what
 * arrives here are the events saying so.
 */
export class Agent {
  readonly name: string;
  readonly instructions: string;
  readonly userId: string;
  readonly tools: Tools;
  readonly client: Client;

  private readonly options: AgentOptions;
  private readonly folder: Folder | undefined;
  private readonly harness: Harness | undefined;
  private edgeImpl: Edge | undefined;
  /** `pipeline.config` resolved to the id the backend wants, looked up once. */
  private configId: string | undefined;

  constructor(options: AgentOptions = {}) {
    this.options = options;
    this.folder = options.folder;

    this.name = options.name || this.folder?.name || "";
    if (!this.name) {
      throw new ConfigurationError("an agent needs a name");
    }
    this.instructions = options.instructions || this.folder?.instructions || "";

    // A directory's skills are folded in here, so everything downstream reads one harness.
    const declared = options.harness;
    const fromFolder = this.folder?.skills ?? [];
    if (declared && (declared.skills?.length ?? 0) === 0 && fromFolder.length > 0) {
      this.harness = { ...declared, skills: fromFolder };
    } else if (!declared && fromFolder.length > 0) {
      this.harness = { useSkills: true, skills: fromFolder };
    } else {
      this.harness = declared;
    }
    validate(this.harness);

    this.userId = options.userId || userIdOf(this.name);
    this.tools = options.tools ?? new Tools();
    this.client =
      options.client instanceof Client ? options.client : new Client(options.client ?? {});
    this.edgeImpl = options.edge;
  }

  /** The guardrail policy the backend screens turns against. */
  get guardrail(): string {
    return this.options.guardrail || this.folder?.guardrail || "";
  }

  /** Creates the Stream calls the backend joins, built from the environment when needed. */
  get edge(): Edge {
    this.edgeImpl ??= new Edge();
    return this.edgeImpl;
  }

  /**
   * Has the backend join a call and hold a conversation on it.
   *
   * An empty call id creates one named after a random string, which is what a one-off
   * conversation wants. It resolves once the backend is in the call, so an agent that has
   * joined is one that is already listening.
   */
  async join(call: Partial<Call> = {}, options: SessionOptions = {}): Promise<Session> {
    const created = await this.edge.createCall(call, { id: this.userId, name: this.name });
    return this.open({ call_id: created.id, call_type: created.type }, options);
  }

  /**
   * A link a person can open to join this session's call from a browser and hear the agent.
   *
   * They join as a listener of their own rather than as the agent, so opening it twice puts
   * two people in the call instead of taking the first one's place.
   */
  monitorURL(session: Session): Promise<string> {
    return this.edge.monitorURLFor(
      {
        id: session.created.call_id ?? "",
        type: session.created.call_type ?? DEFAULT_CALL_TYPE,
      },
      { id: `monitor-${session.id}`, name: "Monitor" },
    );
  }

  /**
   * Holds the conversation in writing rather than on a call.
   *
   * No call is joined, nothing is transcribed and nothing is spoken. Everything between
   * hearing a question and answering it is unchanged: the same instructions, the same
   * skills handed to the same slower model, the same knowledge base.
   */
  chat(options: ChatOptions = {}): Promise<Session> {
    const { persist, conversationId, agentId, ...rest } = options;
    return this.open(
      {
        text: true,
        ...(persist === undefined ? {} : { persist_conversation: persist }),
        ...(conversationId ? { conversation_id: conversationId } : {}),
        ...(agentId ? { agent_id: agentId } : {}),
      },
      rest,
    );
  }

  /**
   * Answers a call that arrived on the dispatch socket.
   *
   * The call already exists — somebody rang a number and the router put them in it — so
   * nothing is created here. The number they reached is carried into the session, which is
   * what lets the agent transfer them.
   */
  answer(call: InboundCall, options: SessionOptions = {}): Promise<Session> {
    return this.open(
      {
        call_id: call.callId,
        call_type: call.callType,
        ...(call.calledNumber ? { phone: { number: call.calledNumber } } : {}),
      },
      options,
    );
  }

  /**
   * Answers a message written to an agent that is not running.
   *
   * The reply is written into the channel the message came from, so whoever wrote it is
   * already reading the answer as it is generated.
   */
  reply(message: InboundMessage, options: SessionOptions = {}): Promise<Session> {
    return this.chat({
      ...options,
      persist: true,
      conversationId: `${message.channelType}:${message.channelId}`,
      agentId: message.agentId,
    });
  }

  /**
   * Rings somebody and holds the conversation when they answer.
   *
   * The agent placed this call, so it is told it is navigating: recordings are let finish
   * and menus are answered rather than talked over.
   */
  async startCall(from: string, to: string, options: SessionOptions = {}): Promise<Session> {
    if (!from || !to) {
      throw new ConfigurationError("a call needs a number to ring from and one to ring");
    }

    const call = await this.edge.createCall({}, { id: this.userId, name: this.name });
    // Placing the call makes its own routing rule pinned to the call named here, so the
    // answered leg arrives in the call this agent is about to join. Attaching the number
    // first would be a second rule for the same number.
    const placed = await this.client.post("/v1/phone/calls", {
      body: {
        from,
        to,
        call_id: call.id,
        call_type: call.type,
        ...(this.options.costTracking ? { tags: this.options.costTracking } : {}),
      },
    });

    return this.open(
      {
        call_id: call.id,
        call_type: call.type,
        navigating: true,
        phone: { number: from, vendor_call_id: placed.vendor_call_id },
      },
      options,
    );
  }

  /**
   * Answers the next call to a number.
   *
   * The number is pointed at a fresh Stream call, the agent joins it, and this waits until
   * somebody rings and says something. The session it returns is the conversation with
   * whoever that was, from their second sentence: the one that unblocked this is read here
   * and does not arrive again on `events`.
   *
   * For more than one call at a time, wait on the dispatch socket instead.
   */
  async waitForCall(number: string, options: SessionOptions = {}): Promise<Session> {
    if (!number) {
      throw new ConfigurationError("there is no number to answer on");
    }

    const call = await this.edge.createCall({}, { id: this.userId, name: this.name });
    await this.client.post("/v1/phone/numbers/{e164}/attach", {
      path: { e164: number },
      body: { call_id: call.id, call_type: call.type },
    });

    const session = await this.open(
      { call_id: call.id, call_type: call.type, phone: { number } },
      options,
    );
    for await (const event of session.events()) {
      if (event.kind === "heard") {
        return session;
      }
    }
    await session.close();
    throw new RouterError(0, "waitForCall", "the call ended before anybody rang");
  }

  /**
   * Stores the agent in the backend: its instructions, guardrail, skills and knowledge,
   * and the models it was declared with.
   *
   * A config is what a session can be created from by name, so the things worth deciding
   * once are decided once. It is one request, and it carries a fingerprint of everything
   * in it, so syncing on every startup does nothing when nothing has changed. A setting
   * left out leaves whatever is stored, so a model chosen in the dashboard survives a sync
   * that says nothing about it.
   *
   * What the directory's agent.yaml declares goes with it, under whatever the code set.
   * A directory read by `loadFolder` records the fingerprint in `.agent_sync`, and a sync of
   * one nothing has touched since only reads the stored config back.
   *
   * Server side only: how an agent is configured is not a browser's to rewrite.
   */
  async sync(): Promise<Schemas["SyncAgentResult"]> {
    const pipeline = this.options.pipeline ?? {};
    const declared = this.folder?.settings ?? {};
    const skills = this.harness?.skills ?? this.folder?.skills ?? [];
    const subagent =
      this.harness?.subagents?.["default"] ?? (pipeline.subagent || declared.subagent);
    const sandbox = this.harness?.vm?.provider ?? declared.sandbox;
    const tags = { ...declared.tags, ...this.options.costTracking };
    const pages = this.folder?.knowledgeURLs ?? [];

    const body: Omit<Schemas["SyncAgentRequest"], "hash"> = {
      name: this.name,
      ...(this.instructions ? { instructions: this.instructions } : {}),
      ...(this.guardrail ? { guardrail: this.guardrail } : {}),
      // config_id is the config these belong to, and the config is written by this same
      // request, so the router fills it in from what it just stored.
      ...(skills.length > 0
        ? { skills: skills.map((skill) => ({ ...skillRequest(skill), config_id: "" })) }
        : {}),
      ...(this.folder?.knowledge.length ? { knowledge: this.folder.knowledge } : {}),
      ...(pages.length > 0 ? { knowledge_urls: pages } : {}),
      ...declaredRequest(declared),
      ...(pipeline.llm ? { llm: pipeline.llm } : {}),
      ...(pipeline.stt ? { stt: pipeline.stt } : {}),
      ...(pipeline.tts ? { tts: pipeline.tts } : {}),
      ...(pipeline.sts ? { sts: pipeline.sts } : {}),
      ...(pipeline.voice ? { voice: pipeline.voice } : {}),
      ...(pipeline.greeting ? { greeting: pipeline.greeting } : {}),
      ...(pipeline.video ? { video: pipeline.video } : {}),
      ...(subagent ? { subagent } : {}),
      ...(sandbox ? { sandbox } : {}),
      ...(Object.keys(tags).length > 0 ? { tags } : {}),
    };

    const hash = await fingerprint(body);
    const stamp = this.folder?.stamp;
    if (stamp && (await stamp.read()) === hash) {
      const [stored] = await this.client.get("/v1/agents/configs", {
        query: { name: this.name },
      });
      // A config deleted since the stamp was written is synced again rather than trusted.
      if (stored?.name === this.name) {
        return { unchanged: true, config: stored };
      }
    }

    const result = await this.client.post("/v1/agents/sync", { body: { ...body, hash } });
    await stamp?.write(hash);
    return result;
  }

  /** Renders the agent's configuration into a session and opens it. */
  private async open(
    call: Schemas["CreateSessionRequest"],
    options: SessionOptions,
  ): Promise<Session> {
    const pipeline = this.options.pipeline ?? {};
    const request: Schemas["CreateSessionRequest"] = {
      user_id: this.userId,
      user_name: this.name,
      agent_id: this.userId,
      ...(this.instructions ? { instructions: this.instructions } : {}),
      ...(await this.pipelineRequest(pipeline)),
      ...(this.harnessRequest()),
      ...(this.options.costTracking ? { tags: this.options.costTracking } : {}),
      ...(this.memoryRequest()),
      ...call,
    };

    return Session.open(this.client, request, { tools: this.tools, ...options });
  }

  private async pipelineRequest(
    pipeline: Pipeline,
  ): Promise<Partial<Schemas["CreateSessionRequest"]>> {
    return {
      ...(pipeline.llm ? { llm: pipeline.llm } : {}),
      ...(pipeline.stt ? { stt: pipeline.stt } : {}),
      ...(pipeline.tts ? { tts: pipeline.tts } : {}),
      ...(pipeline.sts ? { sts: pipeline.sts } : {}),
      ...(pipeline.voice ? { voice: pipeline.voice } : {}),
      ...(pipeline.greeting ? { greeting: pipeline.greeting } : {}),
      ...(pipeline.language ? { languages: [pipeline.language] } : {}),
      ...(pipeline.backchannel === undefined ? {} : { backchannel: pipeline.backchannel }),
      ...(pipeline.maxTokens ? { max_tokens: pipeline.maxTokens } : {}),
      ...(pipeline.toolTimeoutMs ? { tool_timeout_ms: pipeline.toolTimeoutMs } : {}),
      ...(pipeline.video ? { video: pipeline.video } : {}),
      ...(pipeline.config ? { config_id: await this.resolveConfig(pipeline.config) } : {}),
    };
  }

  private harnessRequest(): Partial<Schemas["CreateSessionRequest"]> {
    const harness = this.harness;
    const pipelineSubagent = this.options.pipeline?.subagent;
    if (!harness) {
      return pipelineSubagent ? { subagent: pipelineSubagent } : {};
    }

    // An absent skill list and an empty one mean different things: one leaves the built-in
    // set alone, the other turns delegation off.
    const replaces = (harness.skills?.length ?? 0) > 0 || harness.useSkills === false;
    const subagent = harness.subagents?.["default"] ?? pipelineSubagent;

    return {
      ...(subagent ? { subagent } : {}),
      ...(harness.tasks ? { tasks: harness.tasks } : {}),
      ...(harness.vm ? { sandbox: harness.vm.provider } : {}),
      ...(replaces ? { skills: (harness.skills ?? []).map(skillRequest) } : {}),
    };
  }

  /** Splits the filter into who the memories are about and what narrows them. */
  private memoryRequest(): Partial<Schemas["CreateSessionRequest"]> {
    const filter = this.options.memoryFilter;
    if (!filter || Object.keys(filter).length === 0) {
      return {};
    }

    const narrowing: Record<string, string> = {};
    let userId = "";
    for (const [key, value] of Object.entries(filter)) {
      if (key === USER_KEY) {
        userId = value;
        continue;
      }
      narrowing[key] = value;
    }
    return {
      memory: {
        ...(userId ? { user_id: userId } : {}),
        ...(Object.keys(narrowing).length > 0 ? { filter: narrowing } : {}),
      },
    };
  }

  /**
   * Turns a config name into the id the backend looks one up by.
   *
   * A name that matches nothing stored is passed through untouched: it is then either an
   * id, or a mistake the backend is better placed to report than a guess here would be.
   */
  private async resolveConfig(named: string): Promise<string> {
    if (this.configId) {
      return this.configId;
    }
    const stored = await this.client.get("/v1/agents/configs");
    this.configId = stored.find((config) => config.name === named)?.id ?? named;
    return this.configId;
  }

}

/**
 * Renders a skill the way both the session spec and the sync request take it.
 *
 * The two shapes are the same but for the config a stored skill belongs to, which is not
 * something a session has.
 */
function skillRequest(skill: Skill): Schemas["SessionSkill"] {
  return {
    name: skill.name,
    description: skill.description,
    instructions: skill.instructions,
    ...(skill.captureVideo === undefined ? {} : { capture_video: skill.captureVideo }),
    ...(skill.deadlineMs ? { deadline_ms: skill.deadlineMs } : {}),
  };
}

/**
 * Renders what agent.yaml declared into a sync request. Only what it names is sent, so the
 * router leaves whatever is stored for the rest.
 */
function declaredRequest(declared: Declaration): Partial<Schemas["SyncAgentRequest"]> {
  return {
    ...(declared.mode ? { mode: declared.mode } : {}),
    ...(declared.stt ? { stt: declared.stt } : {}),
    ...(declared.tts ? { tts: declared.tts } : {}),
    ...(declared.sts === undefined ? {} : { sts: declared.sts }),
    ...(declared.voice ? { voice: declared.voice } : {}),
    ...(declared.llm ? { llm: declared.llm } : {}),
    ...(declared.search ? { search: declared.search } : {}),
    ...(declared.greeting ? { greeting: declared.greeting } : {}),
    ...(declared.plugins?.length ? { plugins: declared.plugins } : {}),
    ...(declared.keyterms?.length ? { keyterms: declared.keyterms } : {}),
    ...(declared.video ? { video: declared.video } : {}),
  };
}

/**
 * A fingerprint of everything a sync would write.
 *
 * The router does nothing when it has seen the same one, so what goes into it has to be
 * everything that could change: keys are sorted so the same directory hashes the same way
 * twice regardless of the order the request was built in.
 */
async function fingerprint(body: unknown): Promise<string> {
  const canonical = JSON.stringify(body, (_, value: unknown) =>
    value && typeof value === "object" && !Array.isArray(value)
      ? Object.fromEntries(Object.entries(value).sort(([one], [other]) => one.localeCompare(other)))
      : value,
  );
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonical));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Refuses a harness that would mean something different on every run. */
function validate(harness: Harness | undefined): void {
  if (!harness) {
    return;
  }
  if ((harness.tasks ?? 0) < 0) {
    throw new ConfigurationError("tasks cannot be negative");
  }
  for (const skill of harness.skills ?? []) {
    if (!skill.name) {
      throw new ConfigurationError("a skill needs a name");
    }
    if (!skill.description) {
      throw new ConfigurationError(
        `${skill.name} needs a description, since it is all the fast model sees`,
      );
    }
    if (!skill.instructions) {
      throw new ConfigurationError(
        `${skill.name} needs instructions, since they are what the subagent answers under`,
      );
    }
  }
}

/** Turns a name into something a call can be joined under. */
export function userIdOf(name: string): string {
  const id = name
    .toLowerCase()
    .replace(/[^a-z0-9_-]/g, "-")
    .replace(/^-+|-+$/g, "");
  return id || "vision-agent";
}
