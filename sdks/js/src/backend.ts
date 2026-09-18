import { ConfigurationError } from "./errors.js";

/** Where the router is when nothing says otherwise. */
export const DEFAULT_URL = "http://localhost:8080";

/** The environment the same variables are read from as in the Go and Python clients. */
export const URL_ENV = "STREAM_ACCELERATION_URL";
export const CUSTOMER_ENV = "STREAM_ACCELERATION_CUSTOMER_ID";
export const API_KEY_ENV = "STREAM_API_KEY";
export const API_SECRET_ENV = "STREAM_API_SECRET";
export const AUTHENTICATE_ENV = "STREAM_ACCELERATION_AUTHENTICATE";

/** How long a token minted here lasts. Short, because it is minted per request. */
const TOKEN_VALIDITY_SECONDS = 60 * 60;

/** A token the SDK was handed, or a function that fetches a fresh one. */
export type TokenSource = string | (() => string | Promise<string>);

/**
 * Somebody a client is acting for, as Stream knows them.
 *
 * Only the id is needed here, because it is the token that says who they are and the app's
 * own backend that minted it. The name and the rest are carried so chat and video have
 * something to show without a second lookup.
 */
export interface StreamUser {
  id: string;
  name?: string;
  image?: string;
  [custom: string]: unknown;
}

/** What Stream's own chat and video clients connect with. */
export interface StreamCredentials {
  apiKey: string;
  user: StreamUser;
  token: string;
}

/** The subset of the WebSocket constructor this SDK uses. */
export type WebSocketLike = Pick<
  WebSocket,
  "send" | "close" | "addEventListener" | "removeEventListener" | "readyState"
> & { binaryType: string };

export type WebSocketConstructor = new (url: string) => WebSocketLike;

export interface BackendOptions {
  /** The router's base URL. Falls back to `STREAM_ACCELERATION_URL`, then localhost. */
  url?: string;
  /**
   * Who the work is billed to, taken at face value.
   *
   * This is what a router running without keys in front of it reads, which is a laptop or
   * a deployment behind a proxy that has already decided who the caller is. Falls back to
   * `STREAM_ACCELERATION_CUSTOMER_ID`.
   */
  customerId?: string;
  /** The public half of a Stream credential. Falls back to `STREAM_API_KEY`. */
  apiKey?: string;
  /**
   * The secret belonging to that key, which mints a server-side token.
   *
   * Server side only. A secret in a browser is a secret published, and every caller
   * holding one can rewrite every agent in the app, so `token` is what a browser passes.
   * Falls back to `STREAM_API_SECRET`.
   */
  apiSecret?: string;
  /**
   * A token an app's own backend minted for this end user to hold.
   *
   * This is the browser's credential. What it names is what the router lets the caller
   * reach: the conversations that user owns, and nothing an agent is configured with.
   */
  token?: TokenSource;
  /**
   * The end user this SDK is acting for.
   *
   * With `apiSecret` it is sent as a header, which is how a backend says which of its
   * users it is acting for; the sessions it opens then belong to that user, so the user's
   * own device can reach them afterwards. With `token` it is already in the token and
   * this is ignored.
   */
  userId?: string;
  /**
   * Whether the router is reached through Stream's authenticating proxy, which is what
   * every hosted deployment sits behind.
   *
   * The proxy wants the credential spelled Stream's own way: an `api_key` header rather
   * than the `X-Api-Key` of the spec, and `stream-auth-type: jwt` whoever the token is
   * for. Both spellings cannot be sent at once, which is why this is a switch rather than
   * a belt-and-braces header set: `Stream-Auth-Type: server` is what a router reached
   * directly needs before it admits a backend, and it is the one thing the proxy refuses
   * outright.
   *
   * Opt-in rather than inferred from holding a credential, because a Stream key and secret
   * are in the environment for plenty of reasons that have nothing to do with this router.
   * Falls back to `STREAM_ACCELERATION_AUTHENTICATE`.
   */
  authenticate?: boolean;
  /** Used for every request. Defaults to the global fetch. */
  fetch?: typeof fetch;
  /** Used for every socket. Defaults to the global WebSocket. */
  webSocket?: WebSocketConstructor;
}

/**
 * Where the router is and who is calling it.
 *
 * Three ways to say who that is, and which one a deployment takes is a property of the
 * deployment rather than a choice: a customer id for a router with nothing in front of it,
 * a key and secret for a process the customer runs, and a key and a token for a browser.
 */
export class Backend {
  readonly url: string;
  readonly customerId: string;
  readonly apiKey: string;
  /** Whether the credential is spelled for Stream's proxy rather than for the router. */
  readonly authenticate: boolean;
  private readonly apiSecret: string;
  /**
   * Who this is acting for and what it holds for them, which `setUser` replaces.
   *
   * Mutable because a browser has no credential when the page loads: the client is built
   * against a URL and a key, and the token arrives once the app knows who is looking at it.
   */
  private userIdValue: string;
  private token: TokenSource | undefined;
  private user: StreamUser | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly webSocketImpl: WebSocketConstructor | undefined;

  constructor(options: BackendOptions = {}) {
    this.url = (options.url ?? env(URL_ENV) ?? DEFAULT_URL).replace(/\/+$/, "");
    this.customerId = options.customerId ?? env(CUSTOMER_ENV) ?? "";
    // The same reasoning as the secret below: naming a customer is choosing the way a
    // router with nothing in front of it is reached, and a key that happens to be in the
    // environment does not overrule the choice. Otherwise pointing a client at a local
    // router from a shell that has a Stream key in it sends the credential that router
    // does not read, and it answers that the customer header is missing.
    this.apiKey = options.apiKey ?? (options.customerId ? "" : env(API_KEY_ENV) ?? "");
    // A token that was handed in is the caller's answer to who they are, so an ambient
    // secret does not overrule it. Without this a client built the way a browser builds
    // one turns into a backend as soon as it runs in a process that happens to have the
    // secret in its environment, which is most of them and every test.
    this.apiSecret = options.apiSecret ?? (options.token ? "" : env(API_SECRET_ENV) ?? "");
    this.token = options.token;
    this.userIdValue = options.userId ?? "";
    this.authenticate = options.authenticate ?? boolean(env(AUTHENTICATE_ENV));
    this.webSocketImpl = options.webSocket ?? globalWebSocket();

    const chosen = options.fetch ?? globalThis.fetch;
    if (!chosen) {
      throw new ConfigurationError(
        "there is no fetch here; pass one as the fetch option or run on Node 22 or newer",
      );
    }
    this.fetchImpl = chosen;

    // Falling back to the customer header would send a request the proxy refuses, and
    // report it as whatever the proxy says rather than as what it is. This one is checked
    // here rather than at first use because it is a contradiction in what was passed, not a
    // credential that has yet to arrive.
    if (this.authenticate && !this.apiKey) {
      throw new ConfigurationError(
        `a router behind the proxy is reached with a credential; pass apiKey, or ${API_KEY_ENV}`,
      );
    }
  }

  /** Who this is acting for, which is empty until a token or a user id says. */
  get userId(): string {
    return this.userIdValue;
  }

  /** The user `setUser` was given, for a caller that wants their name back. */
  get identity(): StreamUser | undefined {
    return this.user;
  }

  /**
   * Says who this client is acting for, and hands over the token that proves it.
   *
   * A browser has no credential when the page loads: the app knows who is looking at it
   * only after its own backend has said so, which is also where the token comes from. So
   * the client is built against a URL and a key, and this arrives afterwards.
   *
   * It is async because it is what a caller awaits before making a request — nothing is
   * fetched here, but `await client.setUser(...)` is the line that reads as "from here on
   * this is Jim", and making it synchronous would invite requests that raced it.
   */
  async setUser(user: StreamUser | string, token: TokenSource): Promise<void> {
    const named = typeof user === "string" ? { id: user } : user;
    if (!named.id) {
      throw new ConfigurationError("a user needs an id");
    }
    if (!token) {
      throw new ConfigurationError(`there is no token for ${named.id} to hold`);
    }
    this.user = named;
    this.userIdValue = named.id;
    this.token = token;
  }

  /**
   * Refuses a request there is no way to authenticate.
   *
   * Checked here rather than in the constructor because the requested shape supplies the
   * credential afterwards: `new Client({ url, apiKey })` is a client waiting for a
   * `setUser`, and throwing on that line would make the shape impossible to write.
   */
  private assertCredentialed(): void {
    if (!this.apiKey && !this.customerId) {
      throw new ConfigurationError(
        `who is calling is not set; pass customerId or ${CUSTOMER_ENV} for a router that ` +
          `trusts one, or apiKey with either apiSecret or token`,
      );
    }
    if (this.apiKey && !this.apiSecret && !this.token) {
      throw new ConfigurationError(
        "apiKey needs the secret it belongs to, a token minted with it, or a setUser call",
      );
    }
  }

  /**
   * Whether this backend speaks for a process the customer runs rather than for a device.
   *
   * Only a server-side caller reaches the operations the spec does not mark client
   * accessible, which is everything about how an agent is configured. It is worth asking
   * before a call rather than reading a 403 afterwards.
   */
  get serverSide(): boolean {
    return Boolean(this.apiSecret) || (!this.apiKey && Boolean(this.customerId));
  }

  /**
   * What every request to the router carries.
   *
   * Minted per call, so a client left idle longer than a token lasts does not wake up
   * holding an expired one.
   */
  async headers(): Promise<Record<string, string>> {
    this.assertCredentialed();
    if (!this.apiKey) {
      return { "X-Customer-Id": this.customerId };
    }

    if (this.authenticate) {
      // `jwt` whoever the token is for. The proxy works out the caller from the token it
      // verified and rewrites who the router is told is calling, so saying `server` here
      // would be claiming what the proxy is there to decide — and it refuses it.
      return {
        api_key: this.apiKey,
        "stream-auth-type": "jwt",
        Authorization: `Bearer ${await this.proxyToken()}`,
      };
    }

    const headers: Record<string, string> = { "X-Api-Key": this.apiKey };
    if (this.apiSecret) {
      headers["Authorization"] = `Bearer ${await this.serverToken()}`;
      headers["Stream-Auth-Type"] = "server";
      if (this.userId) {
        headers["X-Stream-User-Id"] = this.userId;
      }
      return headers;
    }

    headers["Authorization"] = `Bearer ${await this.userToken()}`;
    headers["Stream-Auth-Type"] = "jwt";
    return headers;
  }

  /**
   * The WebSocket URL for a path on the router, credentials included.
   *
   * They go in the query string because a browser WebSocket carries no headers of its own.
   * `Stream-Auth-Type` has no query counterpart on purpose, which is why a socket opened
   * from a browser cannot claim to be a backend.
   */
  async socketURL(path: string, query: Record<string, string> = {}): Promise<string> {
    this.assertCredentialed();
    const url = new URL(this.url.replace(/^http/, "ws") + path);
    for (const [name, value] of Object.entries(query)) {
      url.searchParams.set(name, value);
    }

    if (this.apiKey) {
      url.searchParams.set("api_key", this.apiKey);
      url.searchParams.set(
        "token",
        this.authenticate
          ? await this.proxyToken()
          : this.apiSecret
            ? await this.serverToken()
            : await this.userToken(),
      );
      if (this.userId) {
        url.searchParams.set("user_id", this.userId);
      }
    } else {
      url.searchParams.set("customer_id", this.customerId);
    }
    return url.toString();
  }

  /** Opens a socket with the runtime's WebSocket, or the one that was passed in. */
  openSocket(url: string): WebSocketLike {
    if (!this.webSocketImpl) {
      throw new ConfigurationError(
        "there is no WebSocket here; pass one as the webSocket option or run on Node 22 or newer",
      );
    }
    return new this.webSocketImpl(url);
  }

  /** Sends one request. Exposed so the client and the sockets share one fetch. */
  request(url: string, init: RequestInit): Promise<Response> {
    return this.fetchImpl(url, init);
  }

  /**
   * What Stream's own chat and video clients need to connect, or undefined.
   *
   * Undefined for a backend reached by customer id: that is this router's own way of
   * trusting a caller and means nothing to Stream, so there is no credential to pass on. A
   * server-side backend gets a token minted for the user it is acting for rather than its
   * own server token, because a chat client connects as somebody.
   */
  async streamCredentials(): Promise<StreamCredentials | undefined> {
    if (!this.apiKey) {
      return undefined;
    }
    const user = this.user ?? (this.userIdValue ? { id: this.userIdValue } : undefined);
    if (!user) {
      return undefined;
    }
    if (this.token) {
      return { apiKey: this.apiKey, user, token: await this.userToken() };
    }
    if (!this.apiSecret) {
      return undefined;
    }
    return {
      apiKey: this.apiKey,
      user,
      token: await signToken({ user_id: user.id }, this.apiSecret),
    };
  }

  /**
   * A token that speaks for the app itself, which is Stream's `server: true`.
   *
   * It names no user by definition: one that named a user would be a token minted for that
   * user to hold. Which of its users a backend is acting for goes in a header instead.
   */
  private serverToken(): Promise<string> {
    return signToken({ server: true }, this.apiSecret);
  }

  /**
   * The token the proxy is given, which names a user where there is one.
   *
   * A token handed in is used as it is. Otherwise it is minted here, and `userId` is what
   * decides whose it is: the proxy has no header to read a backend's choice of user from,
   * so with a user named the token has to be that user's.
   */
  private proxyToken(): Promise<string> {
    if (this.token) {
      return this.userToken();
    }
    return this.userId
      ? signToken({ user_id: this.userId }, this.apiSecret)
      : this.serverToken();
  }

  private async userToken(): Promise<string> {
    const source = this.token;
    if (typeof source === "string") {
      return source;
    }
    if (!source) {
      throw new ConfigurationError("there is no token to authenticate with");
    }
    return source();
  }
}

/**
 * Signs a Stream token.
 *
 * HS256 through Web Crypto rather than a JWT library, because Web Crypto is in every
 * runtime this package supports and a dependency here would be one in every browser bundle
 * that imports the package for something else.
 */
export async function signToken(
  claims: Record<string, unknown>,
  secret: string,
  validitySeconds = TOKEN_VALIDITY_SECONDS,
): Promise<string> {
  if (!secret) {
    throw new ConfigurationError("a token cannot be signed without a secret");
  }

  const issued = Math.floor(Date.now() / 1000);
  const payload = { iat: issued, exp: issued + validitySeconds, ...claims };
  const signing = `${base64url(JSON.stringify({ alg: "HS256", typ: "JWT" }))}.${base64url(
    JSON.stringify(payload),
  )}`;

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signing));
  return `${signing}.${base64urlBytes(new Uint8Array(mac))}`;
}

function base64url(text: string): string {
  return base64urlBytes(new TextEncoder().encode(text));
}

function base64urlBytes(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/**
 * Reads an environment variable where there is an environment to read.
 *
 * A browser has none, and reaching for `process` there would throw rather than fall back,
 * so the read goes through the global object.
 */
export function env(name: string): string | undefined {
  const values = (globalThis as { process?: { env?: Record<string, string | undefined> } })
    .process?.env;
  const value = values?.[name];
  return value ? value : undefined;
}

function globalWebSocket(): WebSocketConstructor | undefined {
  return (globalThis as { WebSocket?: WebSocketConstructor }).WebSocket;
}

/** Reads a variable written as a flag, the way the Go client reads the same one. */
function boolean(value: string | undefined): boolean {
  return value === "1" || value?.toLowerCase() === "true";
}
