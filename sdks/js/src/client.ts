import {
  Backend,
  type BackendOptions,
  type StreamUser,
  type TokenSource,
  type UserTokenOptions,
} from "./backend.js";
import { ConfigurationError, errorOf, RouterError } from "./errors.js";
import type { components, paths } from "./generated/api.js";
import {
  ANONYMOUS_GUEST_REFUSED,
  claimGuestUser,
  forgetGuest,
  guestUser,
  type Guest,
  type GuestStore,
  type GuestUserOptions,
} from "./guests.js";
import { AgentHandle } from "./handle.js";
import { Memories } from "./memories.js";
import { Router, type RouterOptions } from "./router.js";
import { ScheduledTasks } from "./scheduled-tasks.js";
import { Simulations } from "./simulations.js";
import { SpendLimits } from "./spend-limits.js";
import { StreamPeers, type StreamClients } from "./stream.js";
import { Teams } from "./teams.js";

/** The schemas from the spec, so callers can name a request or a response they build. */
export type Schemas = components["schemas"];

/** The methods the router serves. */
export type Method = "get" | "post" | "put" | "patch" | "delete";

type Operation<P extends keyof paths, M extends Method> = M extends keyof paths[P]
  ? NonNullable<paths[P][M]>
  : never;

/** The paths that answer to one method, which is what keeps `get` off a POST-only path. */
export type PathsWith<M extends Method> = {
  [P in keyof paths]: [Operation<P, M>] extends [never] ? never : P;
}[keyof paths];

type ParametersOf<Op> = Op extends { readonly parameters: infer P } ? P : unknown;

/**
 * The parameters an operation takes, with a query value allowed to be undefined.
 *
 * A caller passing one through from somewhere that may not have it should not have to
 * build the object conditionally: an undefined value is left off the query string rather
 * than sent as the word.
 */
type QueryOf<P> = {
  [K in keyof P]: K extends "query"
    ? { [Q in keyof NonNullable<P[K]>]: NonNullable<P[K]>[Q] | undefined }
    : P[K];
};

/**
 * The JSON body an operation takes, required exactly when the spec requires it.
 *
 * Three cases rather than two, because an optional body and no body at all are different:
 * attaching a number takes one and does not need one, and a caller should be able to leave
 * it out without being able to invent one where the spec has none.
 */
type BodyOf<Op> = Op extends {
  readonly requestBody: { readonly content: { readonly "application/json": infer B } };
}
  ? { readonly body: B }
  : Op extends {
        readonly requestBody?: { readonly content: { readonly "application/json": infer B } };
      }
    ? { readonly body?: B }
    : { readonly body?: never };

/**
 * Everything one call needs, taken from the operation.
 *
 * The generated `parameters` already says which of `path` and `query` an operation has and
 * which are required, so it is reused rather than restated: an endpoint that grows a
 * required parameter starts failing to compile here.
 */
export type RequestOptions<Op> = QueryOf<Omit<ParametersOf<Op>, "header" | "cookie">> &
  BodyOf<Op> & {
    readonly signal?: AbortSignal;
    /** Sent alongside the credentials, for a header a deployment of its own wants. */
    readonly headers?: Readonly<Record<string, string>>;
    /**
     * Whether sending the request twice does what sending it once does, so a failure in
     * between may be retried. GET, PUT and DELETE are; a POST or PATCH is only when it says
     * so, such as one carrying a request id the router answers once.
     */
    readonly idempotent?: boolean;
  };

type ResponsesOf<Op> = Op extends { readonly responses: infer R } ? R : never;

/**
 * What a call answers with.
 *
 * 204 has no body and becomes void, which is most of the ways a session is acted on.
 */
export type Result<Op> = {
  [S in Extract<keyof ResponsesOf<Op>, 200 | 201 | 202 | 204>]: ResponsesOf<Op>[S] extends {
    content: { readonly "application/json": infer B };
  }
    ? B
    : ResponsesOf<Op>[S] extends { content: { readonly "text/event-stream": string } }
      ? ReadableStream<Uint8Array>
      : ResponsesOf<Op>[S] extends { content: { readonly "text/plain": string } }
        ? string
        : void;
}[Extract<keyof ResponsesOf<Op>, 200 | 201 | 202 | 204>];

/** Whether the options can be left out entirely, which they can when nothing is required. */
type Arguments<Op> = Record<string, never> extends RequestOptions<Op>
  ? [options?: RequestOptions<Op>]
  : [options: RequestOptions<Op>];

/**
 * The router's API, as resources: `api.simulations`, `api.memories`, `api.agent(name)`.
 *
 * ```ts
 * const api = new Client();
 * const simulation = await api.simulations.create({ name, config_id, scenario, assertion });
 * const session = await api.agent("docs").sessions.create();
 * ```
 *
 * Underneath, `get`, `post`, `put`, `patch` and `delete` take any path in the spec, typed from
 * it. They are what the resources are built on, and the way to reach an endpoint no resource
 * covers yet; code a customer reads uses the resource.
 */
export class Client {
  readonly backend: Backend;
  /** What the app's agents remember about its users. */
  readonly memories: Memories;
  /** Conversations to test an agent with, and their runs. */
  readonly simulations: Simulations;
  /** The reminders and later work agents hold for the user `setUser` named. */
  readonly scheduledTasks: ScheduledTasks;
  /** The customer's spend limits. Server side only. */
  readonly spendLimits: SpendLimits;
  /** The organization's teams and their spend limits. Server side only. */
  readonly teams: Teams;
  /** @internal The chat and video clients this client's sessions share. */
  readonly peers: StreamPeers;

  constructor(backend: Backend | BackendOptions = {}) {
    this.backend = backend instanceof Backend ? backend : new Backend(backend);
    this.memories = new Memories(this);
    this.simulations = new Simulations(this);
    this.scheduledTasks = new ScheduledTasks(this);
    this.spendLimits = new SpendLimits(this);
    this.teams = new Teams(this, this.spendLimits);
    this.peers = new StreamPeers(this.backend);
  }

  /**
   * Says who this client is acting for, and hands over the token that proves it.
   *
   * ```ts
   * const api = new Client({ url: accelerate, apiKey });
   * await api.setUser({ id: "jlahey", name: "Jim Lahey" }, userToken);
   * ```
   *
   * From here on the conversations this client opens belong to that user, and the ones it
   * lists are that user's own. A browser gets the token from the app's own backend, which is
   * the thing that authenticated them; nothing here mints it, because minting it needs the
   * secret and a secret in a page is a secret published.
   */
  setUser(user: StreamUser | string, token: TokenSource): Promise<void>;
  /** Acts as a guest from `guestUser`, with the token it was minted with. */
  setUser(guest: Guest): Promise<void>;
  setUser(user: StreamUser | string | Guest, token?: TokenSource): Promise<void> {
    if (isGuest(user)) {
      // Only who they are goes on, never the token they carry, which would otherwise be sent to
      // Stream Chat and Video as a field of the user.
      return this.backend.setUser({ id: user.id, ...(user.name ? { name: user.name } : {}) }, token ?? user.token);
    }
    return this.backend.setUser(user, token as TokenSource);
  }

  /**
   * The token a backend hands to one user's device, naming the user's team and usage
   * categories, fixed when it is signed so the device cannot move its work to another team.
   * Server side only: it is signed with the secret.
   *
   * ```ts
   * const token = await api.createUserToken("user-42", { team: "finance" });
   * ```
   */
  createUserToken(userId: string, options: UserTokenOptions = {}): Promise<string> {
    return this.backend.createUserToken(userId, options);
  }

  /**
   * Forgets the user, which is what signing out is. The Stream clients `session.chat()` and
   * `session.video()` built stay connected as them until `disconnect`.
   */
  clearUser(): void {
    this.backend.clearUser();
  }

  /**
   * Hands over the app's own `StreamChat` and `StreamVideoClient`, so every session opens on
   * them rather than on second ones. Connecting them, as the user `setUser` named, stays the
   * app's job, and `disconnect` leaves them alone.
   *
   * ```ts
   * api.use({ chat: streamChat, video: videoClient });
   * ```
   */
  use(clients: StreamClients): void {
    this.peers.use(clients);
  }

  /**
   * Disconnects the Stream chat and video clients `session.chat()` and `session.video()`
   * built. Closing a session leaves them connected, since other sessions share them.
   */
  disconnect(): Promise<void> {
    return this.peers.disconnect();
  }

  /**
   * Search, the one routed modality a device may reach, under a stored router config and with
   * cost labels when given. No request is made until a call.
   *
   * ```ts
   * const answer = await api.router({ config: "healthcare" }).search("what changed in the pricing page");
   * ```
   */
  router(options: RouterOptions = {}): Router {
    return new Router(this, options);
  }

  /**
   * An agent, by the name it is configured under.
   *
   * ```ts
   * const agent = api.agent("docs");
   * ```
   *
   * No request is made: this is the name in a wrapper, and a name that matches nothing is
   * refused when a session is opened rather than here. Which means addressing an agent costs
   * nothing, and is the same line whether the config was written this morning or last year.
   */
  agent(name: string): AgentHandle {
    return new AgentHandle(this, name);
  }

  /**
   * Gets or creates a guest, so somebody can talk to an agent before they sign up.
   *
   * Remembered in a cookie or in localStorage where there is one, so reloading the page is
   * the same guest with the same history rather than a stranger. Nothing is remembered on a
   * server, where there is no "this person" to remember.
   */
  guestUser(options: GuestUserOptions = {}, store?: GuestStore): Promise<Guest> {
    return store === undefined
      ? guestUser(this, options)
      : guestUser(this, options, store);
  }

  /** Forgets the remembered guest, which is what signing in has to do. */
  forgetGuest(store?: GuestStore): void {
    return store === undefined ? forgetGuest() : forgetGuest(store);
  }

  /**
   * Moves a guest's conversations onto the account they turned out to be.
   *
   * Server side only: only the backend that just authenticated the account knows which guest
   * it was, and a page able to ask could claim anybody's conversations by guessing an id.
   */
  claimGuestUser(
    guest: Guest | string,
    real: StreamUser | string,
  ): Promise<Schemas["ClaimGuestResult"]> {
    return claimGuestUser(this, guest, real);
  }

  get<P extends PathsWith<"get">>(
    path: P,
    ...options: Arguments<Operation<P, "get">>
  ): Promise<Result<Operation<P, "get">>> {
    return this.send("get", path, options[0]);
  }

  post<P extends PathsWith<"post">>(
    path: P,
    ...options: Arguments<Operation<P, "post">>
  ): Promise<Result<Operation<P, "post">>> {
    return this.send("post", path, options[0]);
  }

  put<P extends PathsWith<"put">>(
    path: P,
    ...options: Arguments<Operation<P, "put">>
  ): Promise<Result<Operation<P, "put">>> {
    return this.send("put", path, options[0]);
  }

  patch<P extends PathsWith<"patch">>(
    path: P,
    ...options: Arguments<Operation<P, "patch">>
  ): Promise<Result<Operation<P, "patch">>> {
    return this.send("patch", path, options[0]);
  }

  delete<P extends PathsWith<"delete">>(
    path: P,
    ...options: Arguments<Operation<P, "delete">>
  ): Promise<Result<Operation<P, "delete">>> {
    return this.send("delete", path, options[0]);
  }

  private async send<T>(method: Method, template: string, options: unknown): Promise<T> {
    const { path, query, body, signal, headers, idempotent } = (options ?? {}) as {
      path?: Record<string, string | number>;
      query?: Record<string, unknown>;
      body?: unknown;
      signal?: AbortSignal;
      headers?: Record<string, string>;
      idempotent?: boolean;
    };

    const operation = `${method.toUpperCase()} ${template}`;
    const anonymously = operation === "POST /v1/agents/guests" && this.backend.anonymous;
    const retryable = idempotent ?? (method === "get" || method === "put" || method === "delete");
    const url = new URL(this.backend.url + fill(template, path));
    for (const [name, value] of Object.entries({ ...this.backend.query(), ...query })) {
      if (value === undefined || value === null) {
        continue;
      }
      for (const one of Array.isArray(value) ? value : [value]) {
        url.searchParams.append(name, String(one));
      }
    }

    // The token the last attempt carried, so a 401 forgets that one and not a newer one.
    let sent: string | undefined;
    const attempt = async (): Promise<Response> => {
      const credentials = await this.backend.headers(anonymously);
      sent = credentials["Authorization"]?.replace(/^Bearer /, "");
      const request: RequestInit = {
        method: method.toUpperCase(),
        headers: { ...credentials, ...headers },
      };
      if (body !== undefined) {
        request.headers = { ...request.headers, "Content-Type": "application/json" };
        request.body = JSON.stringify(body);
      }
      if (signal) {
        request.signal = signal;
      }
      try {
        return await this.backend.request(url.toString(), request);
      } catch (cause) {
        // The caller's own abort is theirs to see as it is: reported as a router that could
        // not be reached, a caller retrying those would send again what it just cancelled.
        if (signal?.aborted) {
          throw signal.reason ?? cause;
        }
        // A request that never arrived is reported as such rather than as a status, because
        // a caller retrying on a network failure and one retrying on a 500 are different.
        throw new RouterError(0, operation, `could not reach the router: ${String(cause)}`, { cause });
      }
    };

    let retries = 0;
    let refreshed = false;
    while (true) {
      let response: Response;
      try {
        response = await attempt();
      } catch (cause) {
        if (!(cause instanceof RouterError) || !retryable || retries >= this.backend.maxRetries) {
          throw cause;
        }
        await pause(backoff(retries), signal);
        retries += 1;
        continue;
      }

      // A 401 is the token having expired more often than anything else, so a token function
      // is asked for a fresh one and the request sent once more. Once: a second refusal is the
      // token being wrong rather than old, and asking again would only ask the app's backend
      // the same question in a loop.
      if (response.status === 401 && !refreshed && !anonymously && this.backend.expireToken(sent)) {
        refreshed = true;
        await response.body?.cancel();
        continue;
      }

      // A router that is busy or being replaced says so with these, and answers the same
      // request a moment later. Only one that is safe to send twice is, and only a few times:
      // past that the failure is the caller's to see rather than this client's to hide.
      if (retryable && TRANSIENT.has(response.status) && retries < this.backend.maxRetries) {
        const after = retryAfterMs(response);
        if (after === undefined || after <= MAX_RETRY_AFTER_MS) {
          await response.body?.cancel();
          await pause(after ?? backoff(retries), signal);
          retries += 1;
          continue;
        }
      }

      if (anonymously && response.status === 401) {
        throw new ConfigurationError(ANONYMOUS_GUEST_REFUSED);
      }
      if (!response.ok) {
        throw await errorOf(response, operation);
      }
      const mediaType = response.headers.get("Content-Type")?.split(";")[0]?.trim().toLowerCase();
      if (
        response.status === 204 ||
        (response.headers.get("Content-Length") === "0" &&
          mediaType !== "text/plain" && mediaType !== "text/event-stream")
      ) {
        return undefined as T;
      }
      if (mediaType === "text/event-stream") {
        if (!response.body) {
          throw new RouterError(response.status, operation, "the router returned no event stream");
        }
        return response.body as T;
      }
      let content: string;
      try {
        content = await response.text();
      } catch (cause) {
        if (signal?.aborted) {
          throw signal.reason ?? cause;
        }
        if (retryable && retries < this.backend.maxRetries) {
          await pause(backoff(retries), signal);
          retries += 1;
          continue;
        }
        throw new RouterError(0, operation, "the connection ended before the router's response was read", {
          cause,
          requestId: response.headers.get("X-Request-Id") || undefined,
        });
      }
      if (mediaType === "text/plain") {
        return content as T;
      }
      try {
        return JSON.parse(content) as T;
      } catch (cause) {
        throw new RouterError(response.status, operation, "the router returned invalid JSON", {
          cause,
          requestId: response.headers.get("X-Request-Id") || undefined,
        });
      }
    }
  }
}

/** The answers that say the router could not take the request just now, rather than that it was wrong. */
const TRANSIENT = new Set([429, 502, 503, 504]);

/** The first wait before a retry; each one after doubles it, up to `MAX_BACKOFF_MS`. */
const BASE_BACKOFF_MS = 250;
const MAX_BACKOFF_MS = 4_000;

/** The longest `Retry-After` waited out; a router asking for longer is reported instead. */
const MAX_RETRY_AFTER_MS = 30_000;

/**
 * How long to wait before retry `retries` (from 0): doubling, with half of it random so a
 * burst of clients refused together does not come back together.
 */
function backoff(retries: number): number {
  const ceiling = Math.min(MAX_BACKOFF_MS, BASE_BACKOFF_MS * 2 ** retries);
  return ceiling / 2 + Math.random() * (ceiling / 2);
}

/** What the response's `Retry-After` asks for, in milliseconds, as seconds or as a date. */
function retryAfterMs(response: Response): number | undefined {
  const header = response.headers.get("Retry-After");
  if (!header) {
    return undefined;
  }
  const seconds = Number(header);
  if (Number.isFinite(seconds)) {
    return Math.max(0, seconds * 1000);
  }
  const at = Date.parse(header);
  return Number.isNaN(at) ? undefined : Math.max(0, at - Date.now());
}

/** Waits, or rejects with the signal's reason as soon as it aborts. */
function pause(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }
    const aborted = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", aborted);
      resolve();
    }, ms);
    signal?.addEventListener("abort", aborted, { once: true });
  });
}

/** Puts the path parameters into the template, so `{id}` is never sent literally. */
function fill(template: string, path: Record<string, string | number> | undefined): string {
  return template.replace(/\{(\w+)\}/g, (_, name: string) => {
    const value = path?.[name];
    if (value === undefined) {
      // Nothing was sent: this is a mistake in the call, which no retry would fix.
      throw new ConfigurationError(`${template} needs a ${name}`);
    }
    return encodeURIComponent(String(value));
  });
}

/** Whether `setUser` was handed a guest, which carries the token it was minted with. */
function isGuest(user: StreamUser | string | Guest): user is Guest {
  return typeof user === "object" && typeof user["token"] === "string";
}
