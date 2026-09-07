import type { components } from "./api";

type Schemas = components["schemas"];

export type Call = Schemas["Call"];
export type CallEvent = Schemas["CallEvent"];
export type DecisionKind = Schemas["DecisionKind"];
export type TimelineEntry = Schemas["TimelineEntry"];
export type TranscriptMessage = Schemas["TranscriptMessage"];
export type CallToken = Schemas["CallToken"];
export type CallTokenRequest = Schemas["CallTokenRequest"];
export type StatsBucket = Schemas["StatsBucket"];
export type TurnStatsBucket = Schemas["TurnStatsBucket"];
export type AgentConfig = Schemas["AgentConfig"];
export type AgentConfigRequest = Schemas["AgentConfigRequest"];
export type AgentMode = Schemas["AgentMode"];
export type Skill = Schemas["Skill"];
export type SkillRequest = Schemas["SkillRequest"];
export type KnowledgeUrl = Schemas["KnowledgeUrl"];
export type KnowledgeDocument = Schemas["KnowledgeDocument"];
export type IngestedKnowledge = Schemas["IngestedKnowledge"];
export type ChatToken = Schemas["ChatToken"];
export type ChatTokenRequest = Schemas["ChatTokenRequest"];
export type Session = Schemas["Session"];
export type CreateSessionRequest = Schemas["CreateSessionRequest"];
export type Provider = Schemas["Provider"];
export type Voice = Schemas["Voice"];
export type VoiceRequest = Schemas["VoiceRequest"];
export type VoiceSampleRequest = Schemas["VoiceSampleRequest"];
export type PhoneNumber = Schemas["PhoneNumber"];
export type AvailableNumber = Schemas["AvailableNumber"];
export type NumberSearchResult = Schemas["NumberSearchResult"];
export type AttachedNumber = Schemas["AttachedNumber"];
export type AttachNumberRequest = Schemas["AttachNumberRequest"];
export type BuyNumberRequest = Schemas["BuyNumberRequest"];
export type Simulation = Schemas["Simulation"];
export type SimulationRequest = Schemas["SimulationRequest"];
export type SimulationRun = Schemas["SimulationRun"];
export type SimulationCase = Schemas["SimulationCase"];
export type SimulationLine = Schemas["SimulationLine"];
export type Plugin = Schemas["Plugin"];
export type PluginConnection = Schemas["PluginConnection"];
export type AuthorizePluginRequest = Schemas["AuthorizePluginRequest"];
export type PluginAuthorization = Schemas["PluginAuthorization"];

/**
 * Where the router is and who we are talking to it as.
 *
 * Both are public, because both reach the browser either way: the dashboard calls the
 * router directly rather than through a server of its own, so a secret kept here would
 * only be a secret shipped to everyone who opened the page.
 */
export const ROUTER_URL = (
  process.env.NEXT_PUBLIC_ROUTER_URL ?? "http://localhost:8080"
).replace(/\/$/, "");

export const CUSTOMER_ID = process.env.NEXT_PUBLIC_CUSTOMER_ID ?? "";

export class RouterError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "RouterError";
  }
}

type Query = Record<string, string | number | boolean | undefined | null>;

function url(path: string, query?: Query): string {
  const built = new URL(ROUTER_URL + path);
  for (const [key, value] of Object.entries(query ?? {})) {
    if (value !== undefined && value !== null && value !== "") {
      built.searchParams.set(key, String(value));
    }
  }
  return built.toString();
}

async function send<T>(
  method: string,
  path: string,
  options: { query?: Query; body?: unknown } = {},
): Promise<T> {
  const response = await fetch(url(path, options.query), {
    method,
    headers: {
      "X-Customer-Id": CUSTOMER_ID,
      ...(options.body === undefined
        ? {}
        : { "Content-Type": "application/json" }),
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });

  if (!response.ok) {
    // The router answers every failure with an Error object, but a proxy in front of it
    // may not, so the status is what this falls back to rather than a parse failure.
    const detail = await response.text();
    let message = detail || response.statusText;
    try {
      message = (JSON.parse(detail) as { error?: string }).error ?? message;
    } catch {
      // The body was not the router's own error shape, so the text stands.
    }
    throw new RouterError(response.status, message);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const router = {
  calls: (query?: { limit?: number; running?: boolean; agent_id?: string }) =>
    send<Call[]>("GET", "/v1/agents/calls", { query }),
  call: (id: string) => send<Call>("GET", `/v1/agents/calls/${id}`),
  callEvents: (id: string, limit?: number) =>
    send<CallEvent[]>("GET", `/v1/agents/calls/${id}/events`, {
      query: { limit },
    }),
  callTimeline: (id: string) =>
    send<TimelineEntry[]>("GET", `/v1/agents/calls/${id}/timeline`),
  callTranscript: (id: string) =>
    send<TranscriptMessage[]>("GET", `/v1/agents/calls/${id}/transcript`),
  callToken: (id: string, body: CallTokenRequest = {}) =>
    send<CallToken>("POST", `/v1/agents/calls/${id}/token`, { body }),

  stats: (modality: string, from: Date, to: Date) =>
    send<StatsBucket[]>("GET", `/v1/${modality}/stats`, {
      query: {
        from: from.toISOString(),
        to: to.toISOString(),
        granularity: "daily",
      },
    }),
  turnStats: (from: Date, to: Date) =>
    send<TurnStatsBucket[]>("GET", "/v1/turns/stats", {
      query: {
        from: from.toISOString(),
        to: to.toISOString(),
        granularity: "daily",
      },
    }),

  providers: (modality: string) =>
    send<Provider[]>("GET", `/v1/${modality}/providers`),

  configs: () => send<AgentConfig[]>("GET", "/v1/agents/configs"),
  config: (id: string) => send<AgentConfig>("GET", `/v1/agents/configs/${id}`),
  createConfig: (body: AgentConfigRequest) =>
    send<AgentConfig>("POST", "/v1/agents/configs", { body }),
  updateConfig: (id: string, body: AgentConfigRequest) =>
    send<AgentConfig>("PUT", `/v1/agents/configs/${id}`, { body }),
  deleteConfig: (id: string) =>
    send<void>("DELETE", `/v1/agents/configs/${id}`),

  simulations: () => send<Simulation[]>("GET", "/v1/agents/simulations"),
  simulation: (id: string) =>
    send<Simulation>("GET", `/v1/agents/simulations/${id}`),
  createSimulation: (body: SimulationRequest) =>
    send<Simulation>("POST", "/v1/agents/simulations", { body }),
  updateSimulation: (id: string, body: SimulationRequest) =>
    send<Simulation>("PUT", `/v1/agents/simulations/${id}`, { body }),
  deleteSimulation: (id: string) =>
    send<void>("DELETE", `/v1/agents/simulations/${id}`),
  runSimulation: (id: string) =>
    send<SimulationRun>("POST", `/v1/agents/simulations/${id}/run`),

  simulationRuns: (query: { simulationID?: string; limit?: number } = {}) =>
    send<SimulationRun[]>("GET", "/v1/agents/simulation-runs", {
      query: { simulation_id: query.simulationID, limit: query.limit },
    }),
  simulationRun: (id: string) =>
    send<SimulationRun>("GET", `/v1/agents/simulation-runs/${id}`),
  cancelSimulationRun: (id: string) =>
    send<SimulationRun>("POST", `/v1/agents/simulation-runs/${id}/cancel`),

  skills: (configID?: string) =>
    send<Skill[]>("GET", "/v1/agents/skills", {
      query: { config_id: configID },
    }),
  createSkill: (body: SkillRequest) =>
    send<Skill>("POST", "/v1/agents/skills", { body }),
  updateSkill: (id: string, body: SkillRequest) =>
    send<Skill>("PUT", `/v1/agents/skills/${id}`, { body }),
  deleteSkill: (id: string) => send<void>("DELETE", `/v1/agents/skills/${id}`),

  plugins: (query?: string) =>
    send<Plugin[]>("GET", "/v1/agents/plugins", { query: { q: query } }),
  pluginConnections: (configID: string) =>
    send<PluginConnection[]>("GET", `/v1/agents/configs/${configID}/plugins`),
  authorizePlugin: (configID: string, pluginID: string, instanceURL?: string) =>
    send<PluginAuthorization>(
      "POST",
      `/v1/agents/configs/${configID}/plugins/${pluginID}/authorize`,
      { body: instanceURL ? { instance_url: instanceURL } : {} },
    ),
  disconnectPlugin: (configID: string, pluginID: string) =>
    send<void>("DELETE", `/v1/agents/configs/${configID}/plugins/${pluginID}`),

  knowledgeUrls: (namespace?: string) =>
    send<KnowledgeUrl[]>("GET", "/v1/agents/knowledge/urls", {
      query: { namespace },
    }),
  addKnowledgeUrl: (namespace: string, url: string) =>
    send<KnowledgeUrl>("POST", "/v1/agents/knowledge/urls", {
      body: { namespace, url },
    }),
  indexKnowledgeUrl: (id: string) =>
    send<KnowledgeUrl>("POST", `/v1/agents/knowledge/urls/${id}/index`),
  deleteKnowledgeUrl: (id: string) =>
    send<void>("DELETE", `/v1/agents/knowledge/urls/${id}`),
  ingestKnowledge: (namespace: string, documents: KnowledgeDocument[]) =>
    send<IngestedKnowledge>("POST", "/v1/agents/knowledge", {
      body: { namespace, documents },
    }),

  chatToken: (body: ChatTokenRequest) =>
    send<ChatToken>("POST", "/v1/agents/chat-token", { body }),

  // The generated request type marks everything with a schema default as required, which
  // a caller leaving it to the router is not, so what is actually optional is optional.
  createSession: (body: Partial<CreateSessionRequest>) =>
    send<Session>("POST", "/v1/agents/sessions", { body }),
  respondSession: (id: string, text: string) =>
    send<void>("POST", `/v1/agents/sessions/${id}/respond`, { body: { text } }),
  closeSession: (id: string) => send<void>("DELETE", `/v1/agents/sessions/${id}`),

  voices: () => send<Voice[]>("GET", "/v1/agents/voices"),
  createVoice: (body: VoiceRequest) =>
    send<Voice>("POST", "/v1/agents/voices", { body }),
  updateVoice: (id: string, body: VoiceRequest) =>
    send<Voice>("PUT", `/v1/agents/voices/${id}`, { body }),
  deleteVoice: (id: string) => send<void>("DELETE", `/v1/agents/voices/${id}`),
  addVoiceSample: (id: string, body: VoiceSampleRequest) =>
    send<Voice>("POST", `/v1/agents/voices/${id}/samples`, { body }),
  prepareVoice: (id: string, providers: string[]) =>
    send<Voice>("POST", `/v1/agents/voices/${id}/prepare`, {
      body: { providers },
    }),

  numbers: () => send<PhoneNumber[]>("GET", "/v1/phone/numbers"),
  searchNumbers: (query: {
    country: string;
    vendor?: string;
    area_code?: string;
    contains?: string;
  }) =>
    send<NumberSearchResult>("GET", "/v1/phone/numbers/available", { query }),
  buyNumber: (body: BuyNumberRequest) =>
    send<PhoneNumber>("POST", "/v1/phone/numbers", { body }),
  releaseNumber: (e164: string) =>
    send<void>("DELETE", `/v1/phone/numbers/${encodeURIComponent(e164)}`),
  attachNumber: (e164: string, body: AttachNumberRequest) =>
    send<AttachedNumber>(
      "POST",
      `/v1/phone/numbers/${encodeURIComponent(e164)}/attach`,
      { body },
    ),
};

/**
 * sessionSocket is the live feed for a call.
 *
 * A call's id is its session's id, so watching one that is still running needs no
 * endpoint of its own. The customer travels in the query rather than a header because the
 * browser WebSocket API cannot set one.
 */
export function sessionSocket(callID: string): string {
  const socket = new URL(
    `${ROUTER_URL}/v1/agents/sessions/${callID}/events`.replace(
      /^http/,
      "ws",
    ),
  );
  socket.searchParams.set("customer_id", CUSTOMER_ID);
  socket.searchParams.set("interim", "true");
  return socket.toString();
}
