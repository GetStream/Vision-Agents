import type { Backend, WebSocketLike } from "./backend.js";
import { SocketClosedError } from "./errors.js";

/**
 * One JSON message on a router socket.
 *
 * Every frame carries a `type` and the fields of that event. It is loosely typed because
 * OpenAPI stops at the upgrade, so unlike the REST shapes these are not generated: a
 * deployment that has learned a new event should reach a caller reading `frame.type`
 * rather than be dropped by this package.
 */
export type Frame = { readonly type?: string } & Readonly<Record<string, unknown>>;

/** What arrives on a socket: a JSON frame, or audio. */
export type Message = Frame | Uint8Array;

/**
 * One WebSocket to the router.
 *
 * The rest of the client is generated from the OpenAPI spec, but OpenAPI stops at the
 * upgrade, so the sockets are written by hand. This is the whole of it: JSON one way, JSON
 * and audio the other.
 */
export class Socket {
  private readonly socket: WebSocketLike;
  private readonly arrived: Message[] = [];
  private readonly waiting: ((next: IteratorResult<Message>) => void)[] = [];
  private ended = false;
  /** Set by `close()`, so a socket this side shut is never taken for one that failed. */
  private closing = false;
  private failure: SocketClosedError | undefined;

  private constructor(socket: WebSocketLike) {
    this.socket = socket;
    this.socket.binaryType = "arraybuffer";
    this.socket.addEventListener("message", this.receive);
    this.socket.addEventListener("close", this.finish);
    this.socket.addEventListener("error", this.finish);
  }

  /**
   * Opens a socket on a path of the router, credentials included, and tries once more with a
   * fresh token if it is refused.
   *
   * A browser is not told why an upgrade was refused, so a token the router no longer takes
   * looks like any other refusal: when a token function is what the socket carried, it is
   * asked for a new one and the socket dialled again. Once, for the same reason a request is
   * retried once: a second refusal is not the token being old.
   */
  static async dial(backend: Backend, path: string, query: Record<string, string> = {}): Promise<Socket> {
    const url = await backend.socketURL(path, query);
    try {
      return await Socket.open(backend, url);
    } catch (cause) {
      const carried = new URL(url).searchParams.get("token") ?? undefined;
      if (!backend.expireToken(carried)) {
        throw cause;
      }
      return Socket.open(backend, await backend.socketURL(path, query));
    }
  }

  /** Dials the router and resolves once the upgrade is through. */
  static open(backend: Backend, url: string): Promise<Socket> {
    return new Promise((resolve, reject) => {
      let raw: WebSocketLike;
      try {
        raw = backend.openSocket(url);
      } catch (cause) {
        reject(cause);
        return;
      }

      const opened = () => {
        raw.removeEventListener("open", opened);
        raw.removeEventListener("error", refused);
        raw.removeEventListener("close", refused);
        resolve(new Socket(raw));
      };
      // A browser is told nothing about why an upgrade was refused, so the reason the
      // router gave is not available to report. Where it was is, and is usually the answer;
      // the query is left out, because it carries the credentials and errors end up in logs.
      const refused = () => {
        raw.removeEventListener("open", opened);
        raw.removeEventListener("error", refused);
        raw.removeEventListener("close", refused);
        reject(new SocketClosedError(`the router refused the socket at ${withoutQuery(url)}`));
      };

      raw.addEventListener("open", opened);
      raw.addEventListener("error", refused);
      raw.addEventListener("close", refused);
    });
  }

  /**
   * Why the socket stopped, when it did not end normally: an error, or the router closing it
   * with anything but a normal close. Undefined while it is open, after `close()`, and after a
   * normal close. `messages()` ends the same way either way, so this is how a reader tells.
   */
  get closedBy(): SocketClosedError | undefined {
    return this.failure;
  }

  /** Whether the socket can still carry a message. */
  get open(): boolean {
    return !this.ended && this.socket.readyState === 1;
  }

  /** Writes one JSON frame. */
  send(frame: Frame): void {
    if (!this.open) {
      throw new SocketClosedError();
    }
    this.socket.send(JSON.stringify(frame));
  }

  /** Writes one binary frame, which on a modality socket is audio. */
  sendAudio(audio: Uint8Array): void {
    if (!this.open) {
      throw new SocketClosedError();
    }
    this.socket.send(audio);
  }

  /**
   * Yields what arrives until the socket closes.
   *
   * There is one reader: a second `for await` over the same socket would take half the
   * frames each. A caller that needs two views of a conversation should read once and fan
   * out from there.
   */
  async *messages(): AsyncGenerator<Message> {
    while (true) {
      const next = this.arrived.shift();
      if (next !== undefined) {
        yield next;
        continue;
      }
      if (this.ended) {
        return;
      }
      const waited = await new Promise<IteratorResult<Message>>((resolve) => {
        this.waiting.push(resolve);
      });
      if (waited.done) {
        return;
      }
      yield waited.value;
    }
  }

  /** Shuts the socket. Safe to call twice. */
  close(): void {
    this.closing = true;
    if (this.socket.readyState <= 1) {
      this.socket.close();
    }
    this.finish();
  }

  private readonly receive = (event: Event) => {
    const { data } = event as MessageEvent<unknown>;

    if (typeof data === "string") {
      let frame: Frame;
      try {
        frame = JSON.parse(data) as Frame;
      } catch {
        // A text frame that is not JSON can only be a bug on the far side, and ending the
        // stream over it would lose everything said after it.
        return;
      }
      this.deliver(frame);
      return;
    }

    if (data instanceof ArrayBuffer) {
      this.deliver(new Uint8Array(data));
    }
  };

  private readonly finish = (event?: Event) => {
    if (this.ended) {
      // A browser reports a dropped socket as an error and then the close, and only the close
      // says which code and why.
      if (event?.type === "close" && !this.closing) {
        this.failure = failureOf(event) ?? this.failure;
      }
      this.socket.removeEventListener("close", this.finish);
      return;
    }
    this.ended = true;
    if (!this.closing && event) {
      this.failure = failureOf(event);
    }
    this.socket.removeEventListener("message", this.receive);
    this.socket.removeEventListener("error", this.finish);
    if (event?.type !== "error") {
      this.socket.removeEventListener("close", this.finish);
    }
    for (const resolve of this.waiting.splice(0)) {
      resolve({ done: true, value: undefined });
    }
  };

  private deliver(message: Message): void {
    const resolve = this.waiting.shift();
    if (resolve) {
      resolve({ done: false, value: message });
      return;
    }
    this.arrived.push(message);
  }
}

/** A socket URL with its query, which holds the credentials, taken off. */
function withoutQuery(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.origin}${parsed.pathname}`;
  } catch {
    return url.split("?")[0] ?? "";
  }
}

/**
 * What a close or error event says went wrong, or undefined for a normal close: 1000, or 1005
 * when the far side sent no code at all.
 */
function failureOf(event: Event): SocketClosedError | undefined {
  if (event.type === "error") {
    return new SocketClosedError("the socket to the router failed");
  }
  const { code, reason } = event as Event & { code?: number; reason?: string };
  if (code === undefined || code === 1000 || code === 1005) {
    return undefined;
  }
  return new SocketClosedError(`the router closed the socket (${code}${reason ? `: ${reason}` : ""})`);
}

/** Reads a string field, or the empty string if it is absent or another type. */
export function text(frame: Frame, key: string): string {
  const value = frame[key];
  return typeof value === "string" ? value : "";
}

/** Reads a numeric field, or zero if it is absent or another type. */
export function number(frame: Frame, key: string): number {
  const value = frame[key];
  return typeof value === "number" ? value : 0;
}

/** Reads a boolean field, false if it is absent or another type. */
export function flag(frame: Frame, key: string): boolean {
  return frame[key] === true;
}

/** Reads a nested object, or undefined if it is absent or another type. */
export function nested(frame: Frame, key: string): Frame | undefined {
  const value = frame[key];
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Frame)
    : undefined;
}
