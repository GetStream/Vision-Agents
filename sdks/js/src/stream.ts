import type { Backend, StreamCredentials, StreamUser } from "./backend.js";
import { ConfigurationError } from "./errors.js";

/** The parts of a `StreamChat` this package touches. */
export interface ChatClient {
  /** Who it is connected as, in stream-chat 10. */
  readonly userId?: string | undefined;
  /** Who it is connected as, in stream-chat 9 (a deprecated alias of `userId` in 10). */
  readonly userID?: string | undefined;
  channel(type: string, id: string): unknown;
  connectUser(user: { id: string }, token: string | (() => Promise<string>)): Promise<unknown>;
  disconnectUser(): Promise<unknown>;
}

/** The parts of a `StreamVideoClient` this package touches. */
export interface VideoClient {
  readonly streamClient: { readonly userID?: string };
  call(type: string, id: string): unknown;
  disconnectUser(): Promise<unknown>;
}

/** Stream clients the app already holds, handed over with `client.use`. */
export interface StreamClients {
  chat?: ChatClient;
  video?: VideoClient;
}

/** `StreamChat`, typed as far as this package uses it. */
export type ChatConstructor = new (apiKey: string) => ChatClient;

/** `StreamVideoClient`, typed as far as this package uses it. */
export type VideoConstructor = new (options: {
  apiKey: string;
  user: { id: string };
  tokenProvider: () => Promise<string>;
}) => VideoClient;

/**
 * The Stream classes `@stream-io/calypso/chat` and `/video` registered on import.
 *
 * Those entry points are what import the optional peers, statically, so a bundler resolves
 * them for an app that asked and never sees them in one that did not. This package imports
 * neither: it has no dependencies, and two large ones would be paid for by everybody.
 */
const registered: { chat?: ChatConstructor; video?: VideoConstructor } = {};

/** Called by `@stream-io/calypso/chat`. */
export function registerChat(StreamChat: ChatConstructor): void {
  registered.chat = StreamChat;
}

/** Called by `@stream-io/calypso/video`. */
export function registerVideo(StreamVideoClient: VideoConstructor): void {
  registered.video = StreamVideoClient;
}

/**
 * Stream's own chat and video clients: the app's, or else built from the identity `setUser`
 * gave.
 *
 * One of each per key and user, however many sessions ask: a second connection as the same
 * person is a second socket for nothing, and on a phone the SDKs refuse one outright. Each
 * built one is handed a token function rather than a token, so it outlives the token it
 * started with.
 */
export class StreamPeers {
  private readonly backend: Backend;
  private readonly given: StreamClients = {};
  private readonly chats = new Map<string, Promise<ChatClient>>();
  private readonly videos = new Map<string, Promise<VideoClient>>();

  constructor(backend: Backend) {
    this.backend = backend;
  }

  /** Uses the app's own clients from now on, whoever signs in, rather than building others. */
  use(clients: StreamClients): void {
    if (clients.chat) {
      this.given.chat = clients.chat;
    }
    if (clients.video) {
      this.given.video = clients.video;
    }
  }

  /** The connected `StreamChat` for the current user. */
  async chat(): Promise<ChatClient> {
    if (this.given.chat) {
      return this.sameUser("chat", this.given.chat, this.given.chat.userId ?? this.given.chat.userID);
    }
    return this.shared(this.chats, async () => {
      const credentials = await this.credentials("chat");
      const StreamChat = installed(registered.chat, "chat", "stream-chat");
      const client = new StreamChat(credentials.apiKey);
      await client.connectUser(chatUser(client, credentials.user), streamTokens(this.backend, credentials, "chat"));
      return client;
    });
  }

  /** The `StreamVideoClient` for the current user. */
  async video(): Promise<VideoClient> {
    if (this.given.video) {
      return this.sameUser("video", this.given.video, this.given.video.streamClient.userID);
    }
    return this.shared(this.videos, async () => {
      const credentials = await this.credentials("video");
      const StreamVideoClient = installed(registered.video, "video", "@stream-io/video-client");
      return new StreamVideoClient({
        apiKey: credentials.apiKey,
        user: credentials.user,
        tokenProvider: streamTokens(this.backend, credentials, "video"),
      });
    });
  }

  /** Disconnects every client built here. The app's own are left connected. */
  async disconnect(): Promise<void> {
    const opened = [...this.chats.values(), ...this.videos.values()];
    this.chats.clear();
    this.videos.clear();
    await Promise.all(opened.map(async (client) => (await client).disconnectUser()));
  }

  /** The app's client, unless it is connected as somebody other than the user `setUser` named. */
  private sameUser<T>(what: string, client: T, connected: string | undefined): T {
    const user = this.backend.userId;
    if (connected && user && connected !== user) {
      throw new ConfigurationError(
        `the ${what} client is connected as ${connected} but setUser named ${user}: connect it ` +
          `as the same user, or the conversation is somebody else's`,
      );
    }
    return client;
  }

  private async credentials(what: string, refresh = false): Promise<StreamCredentials> {
    const credentials = await this.backend.streamCredentials(refresh);
    if (!credentials) {
      throw new ConfigurationError(
        `${what} connects to Stream rather than to this router, so it needs a Stream key and ` +
          `a user: pass apiKey and call setUser, or pass apiKey with apiSecret and userId`,
      );
    }
    return credentials;
  }

  /**
   * The client cached for the current key and user, or the one `open` builds. Keyed before any
   * token is asked for, so a session reusing a connection mints nothing. A failed one is not kept.
   */
  private shared<T>(cache: Map<string, Promise<T>>, open: () => Promise<T>): Promise<T> {
    const key = `${this.backend.apiKey}:${this.backend.userId}`;
    let opening = cache.get(key);
    if (!opening) {
      opening = open();
      cache.set(key, opening);
      opening.catch(() => cache.delete(key));
    }
    return opening;
  }
}

/** @internal Refreshes a connection's token only while its original user is still signed in. */
export function streamTokens(
  backend: Backend,
  credentials: StreamCredentials,
  what: string,
): () => Promise<string> {
  let unused: string | undefined = credentials.token;
  return async () => {
    if (backend.userId !== credentials.user.id) {
      throw new ConfigurationError(
        `the ${what} connection belongs to ${credentials.user.id}; disconnect it after changing users`,
      );
    }
    const token = unused ?? (await backend.streamCredentials(true))?.token;
    if (!token) {
      throw new ConfigurationError(`there is no token for the ${what} connection`);
    }
    unused = undefined;
    return token;
  };
}

/** The class an entry point registered, or the error saying which one to import. */
function installed<T>(registered: T | undefined, what: string, peer: string): T {
  if (!registered) {
    throw new ConfigurationError(
      `${what} needs ${peer}: install it with npm install ${peer}, then import ` +
        `"@stream-io/calypso/${what}" once, beside the import of this package`,
    );
  }
  return registered;
}

/** What stream-chat 10 takes of a user top level when connecting; the rest goes under `custom`. */
const CONNECT_FIELDS = new Set(["id", "name", "image", "language", "invisible", "privacy_settings"]);

/**
 * The user `setUser` named, as the installed stream-chat takes it.
 *
 * `StreamUser` keeps the app's own fields top level, which is how stream-chat 9 takes them.
 * 10 takes them only under `custom`, and a client with `userId` is 10's: 9 has only `userID`,
 * set once connected. What the app already put under `custom` is kept there.
 */
function chatUser(client: ChatClient, user: StreamUser): StreamUser {
  if (!("userId" in client)) {
    return user;
  }
  const top: StreamUser = { id: user.id };
  const custom: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(user)) {
    if (CONNECT_FIELDS.has(key)) {
      top[key] = value;
    } else if (key !== "custom") {
      custom[key] = value;
    }
  }
  const given = user["custom"];
  if (typeof given === "object" && given !== null && !Array.isArray(given)) {
    Object.assign(custom, given);
  } else if (given !== undefined) {
    custom["custom"] = given;
  }
  return Object.keys(custom).length > 0 ? { ...top, custom } : top;
}
