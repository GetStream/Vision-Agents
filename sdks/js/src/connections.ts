import { Client, type Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** One account the app or one of its users holds at a connector, such as a user's Linear. */
export type Connection = Schemas["Connection"];
/** A consent started for a connection: where to send the browser, and until when. */
export type Authorization = Schemas["Authorization"];

/** Whose connection a call is about. */
export interface ConnectionUser {
  /**
   * The end user whose connection it is, sent as the owner and as `X-Stream-User-Id`. Left out or
   * empty, it is the app's own connection, and the call adds no user (a client built with a
   * `userId` of its own still sends that one, as on its other calls).
   */
  readonly userId?: string;
}

/** How a connection is created, and whose it is. */
export interface ConnectionOptions extends ConnectionUser {
  /** Tells the owner's connections to the same connector apart. */
  readonly label?: string;
}

/** Where a consent sends the browser back to, and whose connection it is. */
export interface AuthorizationOptions extends ConnectionUser {
  /**
   * The app's own page the browser goes back to when the consent ends, with `connection_id` and
   * `status` added to its query. Left out, it goes back to the dashboard.
   */
  readonly returnUrl?: string;
}

/** Which connections `list` reads. */
export interface ConnectionFilter extends ConnectionUser {
  /** Keeps one connector's, such as `linear`. Left out, every connector's. */
  readonly connectorId?: string;
}

/** The page size `list` reads in, the most the router answers. */
const PAGE = 200;

/**
 * The accounts the app and its users hold at connectors, such as a user's Linear: what an
 * agent's tools reach that account through. Server side only: the router refuses a user's
 * token with a 403 `server_side_only`.
 *
 * The user is a parameter of each call, never a client of its own. With `userId` the call is
 * about that user's connections: the user goes to the router as the owner and as
 * `X-Stream-User-Id` alike, on reads too, so the two never differ. Without it the call is about
 * the app's own connections. An app's connection is reached whatever `userId` a call names; a
 * user's only with that user's id, and for anybody else it is not found, the same as one that
 * does not exist.
 *
 * ```ts
 * const connection = await api.connections.create("linear", { userId: "alice" });
 * const consent = await api.connections.authorize(connection.id, { returnUrl, userId: "alice" });
 * ```
 */
export class Connections {
  private readonly client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /**
   * Stores a pending connection to `connectorId`, a built-in such as `linear` or one of the app's
   * own connectors: the user's when `userId` names one, else the app's.
   */
  async create(connectorId: string, options: ConnectionOptions = {}): Promise<Connection> {
    const owner: Schemas["ConnectionOwner"] = options.userId
      ? { type: "user", user_id: options.userId }
      : { type: "app" };
    return this.as(options.userId).post("/v1/agents/connections", {
      body: { connector_id: connectorId, ...(options.label ? { label: options.label } : {}), owner },
    });
  }

  /**
   * Starts the provider's consent for a connection: a first one for a pending connection, a
   * reconnect for one connected before. Open `launch_url` in a popup from a page of `returnUrl`'s
   * origin and post it `handoff_token`; the attempt ends at `expires_at`.
   */
  async authorize(id: string, options: AuthorizationOptions = {}): Promise<Authorization> {
    return this.as(options.userId).post("/v1/agents/connections/{id}/authorizations", {
      path: { id: named(id) },
      ...(options.returnUrl ? { body: { return_url: options.returnUrl } } : {}),
    });
  }

  /** One connection. Its `status` is `pending` until the owner consents. */
  async get(id: string, options: ConnectionUser = {}): Promise<Connection> {
    return this.as(options.userId).get("/v1/agents/connections/{id}", { path: { id: named(id) } });
  }

  /**
   * Every connection of the user `userId` names (only theirs, not the app's), or the app's own
   * without it, newest first, read page by page.
   */
  async list(filter: ConnectionFilter = {}): Promise<readonly Connection[]> {
    const client = this.as(filter.userId);
    const connections: Connection[] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await client.get("/v1/agents/connections", {
        query: {
          owner_type: filter.userId ? "user" : "app",
          connector_id: filter.connectorId || undefined,
          limit: PAGE,
          cursor,
        },
      });
      connections.push(...(page.items ?? []));
      if (!page.has_more || !page.next_cursor) {
        return connections;
      }
      cursor = page.next_cursor;
    }
  }

  /**
   * Disconnects a connection and drops its credentials. One an agent config binds as its fixed
   * connection is refused with a 409.
   */
  async delete(id: string, options: ConnectionUser = {}): Promise<void> {
    await this.as(options.userId).delete("/v1/agents/connections/{id}", { path: { id: named(id) } });
  }

  /** The client a call goes out on: this one, or this backend's credential speaking for the user. */
  private as(userId: string | undefined): Client {
    return userId ? new Client(this.client.backend.speakingFor(userId)) : this.client;
  }
}

/**
 * The connection id a call names. An empty one is refused before anything is sent: it would ask
 * for `/v1/agents/connections/`, which names no connection.
 */
function named(id: string): string {
  if (!id) {
    throw new ConfigurationError("a connection needs its id");
  }
  return id;
}
