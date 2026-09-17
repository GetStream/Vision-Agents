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

  private constructor(socket: WebSocketLike) {
    this.socket = socket;
    this.socket.binaryType = "arraybuffer";
    this.socket.addEventListener("message", this.receive);
    this.socket.addEventListener("close", this.finish);
    this.socket.addEventListener("error", this.finish);
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
      // router gave is not available to report. The URL is, and is usually the answer.
      const refused = () => {
        raw.removeEventListener("open", opened);
        raw.removeEventListener("error", refused);
        raw.removeEventListener("close", refused);
        reject(new SocketClosedError(`the router refused the socket at ${url}`));
      };

      raw.addEventListener("open", opened);
      raw.addEventListener("error", refused);
      raw.addEventListener("close", refused);
    });
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

  private readonly finish = () => {
    if (this.ended) {
      return;
    }
    this.ended = true;
    this.socket.removeEventListener("message", this.receive);
    this.socket.removeEventListener("close", this.finish);
    this.socket.removeEventListener("error", this.finish);
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
