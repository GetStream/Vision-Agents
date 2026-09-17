import { API_KEY_ENV, API_SECRET_ENV, env, signToken } from "./backend.js";
import { ConfigurationError, RouterError } from "./errors.js";

/** Where Stream's API is. The video paths are served from the same host as chat's. */
const STREAM_API = "https://chat.stream-io-api.com";

/** The Stream call type used when none is named. */
export const DEFAULT_CALL_TYPE = "agent";

/** Stream's hosted video demo, the same page the Python examples open. */
export const DEFAULT_MONITOR_URL = "https://getstream.io/video/demos";

/** How long a browser's token lasts. A call outliving it is a call nobody is still on. */
const MONITOR_TOKEN_VALIDITY = 60 * 60;

/** Somebody a call is created by or watched as. */
export interface User {
  id: string;
  name?: string;
}

/** One Stream call, named the way the backend needs it named. */
export interface Call {
  id: string;
  type: string;
}

export interface EdgeOptions {
  /** The Stream app calls are created in. Falls back to `STREAM_API_KEY`. */
  apiKey?: string;
  /** Its secret, which signs the tokens. Falls back to `STREAM_API_SECRET`. */
  apiSecret?: string;
  /** Points the monitoring link at another deployment of the demo. */
  monitorURL?: string;
  fetch?: typeof fetch;
}

/**
 * The Stream call an agent is asked to join.
 *
 * The acceleration backend joins a call that already exists, so what is needed here is
 * creating one and a link a person can open to be on the other end of it. No media crosses
 * this class: the conversation happens in the backend.
 *
 * Server side only, because it holds the app secret.
 */
export class Edge {
  private readonly apiKey: string;
  private readonly apiSecret: string;
  private readonly monitorURL: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: EdgeOptions = {}) {
    this.apiKey = options.apiKey ?? env(API_KEY_ENV) ?? "";
    this.apiSecret = options.apiSecret ?? env(API_SECRET_ENV) ?? "";
    this.monitorURL = (options.monitorURL ?? env("EXAMPLE_BASE_URL") ?? DEFAULT_MONITOR_URL)
      .replace(/\/+$/, "");
    this.fetchImpl = options.fetch ?? globalThis.fetch;

    if (!this.apiKey || !this.apiSecret) {
      throw new ConfigurationError(
        `${API_KEY_ENV} and ${API_SECRET_ENV} are required to create a call`,
      );
    }
  }

  /**
   * Creates the call the backend will join, or returns the one already under that id.
   *
   * An empty id names a new call after a random one, which is what a one-off conversation
   * wants.
   */
  async createCall(call: Partial<Call>, createdBy: User): Promise<Call> {
    if (!createdBy.id) {
      throw new ConfigurationError("a call needs somebody to have created it");
    }

    const named: Call = {
      id: call.id || randomID(),
      type: call.type || DEFAULT_CALL_TYPE,
    };

    const url = new URL(
      `${STREAM_API}/api/v2/video/call/${encodeURIComponent(named.type)}/${encodeURIComponent(named.id)}`,
    );
    url.searchParams.set("api_key", this.apiKey);

    const response = await this.fetchImpl(url.toString(), {
      method: "POST",
      headers: {
        Authorization: await signToken({ server: true }, this.apiSecret),
        "Stream-Auth-Type": "jwt",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data: { created_by_id: createdBy.id } }),
    });
    if (!response.ok) {
      const said = await response.text().catch(() => "");
      throw new RouterError(
        response.status,
        `POST /video/call/${named.type}/${named.id}`,
        said.trim() || `Stream answered ${response.status}`,
      );
    }
    return named;
  }

  /**
   * Mints a token for somebody to join a call as.
   *
   * Signed here rather than fetched, so this makes no network calls: the coordinator
   * registers the browser as a user when it connects, the same way it does for the agent.
   */
  token(user: User, validitySeconds = MONITOR_TOKEN_VALIDITY): Promise<string> {
    if (!user.id) {
      throw new ConfigurationError("a token needs a user to name");
    }
    return signToken({ user_id: user.id }, this.apiSecret, validitySeconds);
  }

  /** A link a person can open to join a call from a browser and hear the agent. */
  async monitorURLFor(call: Call, user: User): Promise<string> {
    if (!call.id) {
      throw new ConfigurationError("there is no call to watch");
    }

    const query = new URLSearchParams({
      api_key: this.apiKey,
      token: await this.token(user),
      skip_lobby: "true",
      user_name: user.name || user.id,
    });
    return `${this.monitorURL}/join/${encodeURIComponent(call.id)}?${query.toString()}`;
  }
}

function randomID(): string {
  const raw = crypto.getRandomValues(new Uint8Array(8));
  return [...raw].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
