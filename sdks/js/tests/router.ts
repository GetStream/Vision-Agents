import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { WebSocketServer, type WebSocket } from "ws";

/** One request the SDK made, as the router saw it. */
export interface Received {
  method: string;
  path: string;
  query: URLSearchParams;
  headers: Record<string, string>;
  body: unknown;
}

/** What to answer one route with. */
export interface Reply {
  status?: number;
  body?: unknown;
}

/** One socket a test is holding the other end of. */
export interface Connection {
  socket: WebSocket;
  path: string;
  query: URLSearchParams;
  /** Sends one JSON frame down to the SDK. */
  send: (frame: Record<string, unknown>) => void;
  /** Resolves with the next frame the SDK sends up. */
  next: () => Promise<Record<string, unknown>>;
}

/**
 * A router a test can hold the other end of.
 *
 * It is a real HTTP server and a real WebSocket upgrade rather than a stubbed fetch, so
 * what the tests exercise is the request this SDK actually puts on the wire.
 */
export class TestRouter {
  readonly received: Received[] = [];

  private readonly server: Server;
  private readonly sockets = new WebSocketServer({ noServer: true });
  private readonly routes = new Map<string, Reply>();
  private readonly waiting: ((connection: Connection) => void)[] = [];
  private readonly connected: Connection[] = [];
  private address = "";

  private constructor() {
    this.server = createServer((request, response) => {
      const url = new URL(request.url ?? "/", "http://router");
      const chunks: Buffer[] = [];
      request.on("data", (chunk: Buffer) => chunks.push(chunk));
      request.on("end", () => {
        const raw = Buffer.concat(chunks).toString();
        this.received.push({
          method: request.method ?? "",
          path: url.pathname,
          query: url.searchParams,
          headers: headersOf(request),
          body: raw ? JSON.parse(raw) : undefined,
        });

        const reply = this.routes.get(`${request.method} ${url.pathname}`);
        if (!reply) {
          response.writeHead(404, { "Content-Type": "application/json" });
          response.end(JSON.stringify({ error: `nothing serves ${url.pathname}` }));
          return;
        }
        const status = reply.status ?? 200;
        if (status === 204 || reply.body === undefined) {
          response.writeHead(status);
          response.end();
          return;
        }
        response.writeHead(status, { "Content-Type": "application/json" });
        response.end(JSON.stringify(reply.body));
      });
    });

    this.server.on("upgrade", (request, socket, head) => {
      const url = new URL(request.url ?? "/", "http://router");
      this.sockets.handleUpgrade(request, socket, head, (raw) => {
        const connection = connectionOf(raw, url);
        const waiting = this.waiting.shift();
        if (waiting) {
          waiting(connection);
        } else {
          this.connected.push(connection);
        }
      });
    });
  }

  /** Starts one on a port the operating system picks. */
  static async start(): Promise<TestRouter> {
    const router = new TestRouter();
    await new Promise<void>((resolve) => router.server.listen(0, "127.0.0.1", resolve));
    const { port } = router.server.address() as AddressInfo;
    router.address = `http://127.0.0.1:${port}`;
    return router;
  }

  get url(): string {
    return this.address;
  }

  /** Answers one method and path with this. */
  serve(method: string, path: string, reply: Reply): this {
    this.routes.set(`${method} ${path}`, reply);
    return this;
  }

  /** Resolves with the next socket the SDK opens. */
  socket(): Promise<Connection> {
    const open = this.connected.shift();
    if (open) {
      return Promise.resolve(open);
    }
    return new Promise((resolve) => this.waiting.push(resolve));
  }

  /** The last request the SDK made, which is what most assertions are about. */
  get last(): Received {
    const request = this.received.at(-1);
    if (!request) {
      throw new Error("the SDK made no requests");
    }
    return request;
  }

  async stop(): Promise<void> {
    for (const connection of this.connected) {
      connection.socket.close();
    }
    this.sockets.close();
    await new Promise<void>((resolve) => this.server.close(() => resolve()));
  }
}

function connectionOf(socket: WebSocket, url: URL): Connection {
  const arrived: Record<string, unknown>[] = [];
  const waiting: ((frame: Record<string, unknown>) => void)[] = [];

  socket.on("message", (data) => {
    const frame = JSON.parse(data.toString()) as Record<string, unknown>;
    const next = waiting.shift();
    if (next) {
      next(frame);
    } else {
      arrived.push(frame);
    }
  });

  return {
    socket,
    path: url.pathname,
    query: url.searchParams,
    send: (frame) => socket.send(JSON.stringify(frame)),
    next: () => {
      const frame = arrived.shift();
      if (frame) {
        return Promise.resolve(frame);
      }
      return new Promise((resolve) => waiting.push(resolve));
    },
  };
}

function headersOf(request: IncomingMessage): Record<string, string> {
  const read: Record<string, string> = {};
  for (const [name, value] of Object.entries(request.headers)) {
    read[name.toLowerCase()] = Array.isArray(value) ? value.join(", ") : (value ?? "");
  }
  return read;
}

/** Reads a JWT's claims without verifying it, which is all a test needs to assert on. */
export function claimsOf(token: string): Record<string, unknown> {
  const payload = token.split(".")[1];
  if (!payload) {
    throw new Error(`${token} is not a JWT`);
  }
  return JSON.parse(Buffer.from(payload, "base64url").toString()) as Record<string, unknown>;
}
