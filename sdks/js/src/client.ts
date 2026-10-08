import { Backend, type BackendOptions, type StreamUser, type TokenSource } from "./backend.js";
import { errorOf, RouterError } from "./errors.js";
import type { components, paths } from "./generated/api.js";
import {
  claimGuestUser,
  forgetGuest,
  guestUser,
  type Guest,
  type GuestStore,
  type GuestUserOptions,
} from "./guests.js";
import { AgentHandle } from "./handle.js";
import { Memories } from "./memories.js";
import { Simulations } from "./simulations.js";

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

  constructor(backend: Backend | BackendOptions = {}) {
    this.backend = backend instanceof Backend ? backend : new Backend(backend);
    this.memories = new Memories(this);
    this.simulations = new Simulations(this);
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
  setUser(user: StreamUser | string, token: TokenSource): Promise<void> {
    return this.backend.setUser(user, token);
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
    const { path, query, body, signal, headers } = (options ?? {}) as {
      path?: Record<string, string | number>;
      query?: Record<string, unknown>;
      body?: unknown;
      signal?: AbortSignal;
      headers?: Record<string, string>;
    };

    const operation = `${method.toUpperCase()} ${template}`;
    const url = new URL(this.backend.url + fill(template, path));
    for (const [name, value] of Object.entries({ ...this.backend.query(), ...query })) {
      if (value === undefined || value === null) {
        continue;
      }
      for (const one of Array.isArray(value) ? value : [value]) {
        url.searchParams.append(name, String(one));
      }
    }

    const request: RequestInit = {
      method: method.toUpperCase(),
      headers: { ...(await this.backend.headers()), ...headers },
    };
    if (body !== undefined) {
      request.headers = { ...request.headers, "Content-Type": "application/json" };
      request.body = JSON.stringify(body);
    }
    if (signal) {
      request.signal = signal;
    }

    let response: Response;
    try {
      response = await this.backend.request(url.toString(), request);
    } catch (cause) {
      // A request that never arrived is reported as such rather than as a status, because
      // a caller retrying on a network failure and one retrying on a 500 are different.
      throw new RouterError(0, operation, `could not reach the router: ${String(cause)}`);
    }

    if (!response.ok) {
      throw await errorOf(response, operation);
    }
    if (response.status === 204 || response.headers.get("Content-Length") === "0") {
      return undefined as T;
    }
    return (await response.json()) as T;
  }
}

/** Puts the path parameters into the template, so `{id}` is never sent literally. */
function fill(template: string, path: Record<string, string | number> | undefined): string {
  return template.replace(/\{(\w+)\}/g, (_, name: string) => {
    const value = path?.[name];
    if (value === undefined) {
      throw new RouterError(0, template, `${template} needs a ${name}`);
    }
    return encodeURIComponent(String(value));
  });
}
