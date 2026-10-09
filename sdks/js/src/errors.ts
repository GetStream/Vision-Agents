import type { components } from "./generated/api.js";

/**
 * The kind of failure, which decides the status: one the spec names, or one a newer router
 * added. Typed open rather than refused, because a failure is worth more read than dropped.
 */
export type RouterErrorType = components["schemas"]["ErrorType"] | (string & {});

/** What the router said went wrong. */
export class RouterError extends Error {
  /** The HTTP status, or 0 when the request never got an answer. */
  readonly status: number;
  /** The method and path that was asked for, for a log line that says which call failed. */
  readonly operation: string;
  /** The kind of failure, or undefined when the body was not the router's. */
  readonly type: RouterErrorType | undefined;
  /**
   * What went wrong, for a program to branch on: `not_configured`, `validation_failed`,
   * `session_not_found` and more as the router learns them, so an unknown one is expected.
   */
  readonly code: string | undefined;
  /** Where the code is explained. */
  readonly docUrl: string | undefined;
  /**
   * The response's `X-Request-Id`, which is what to quote to support: a 500 says only
   * "something went wrong", and this is how the rest of it is found.
   */
  readonly requestId: string | undefined;

  constructor(
    status: number,
    operation: string,
    message: string,
    detail: {
      readonly type?: RouterErrorType | undefined;
      readonly code?: string | undefined;
      readonly docUrl?: string | undefined;
      readonly requestId?: string | undefined;
    } = {},
  ) {
    super(message);
    this.name = "RouterError";
    this.status = status;
    this.operation = operation;
    this.type = detail.type;
    this.code = detail.code;
    this.docUrl = detail.docUrl;
    this.requestId = detail.requestId;
  }
}

/** A socket that is no longer open was written to. */
export class SocketClosedError extends Error {
  constructor(message = "the socket is not open") {
    super(message);
    this.name = "SocketClosedError";
  }
}

/**
 * The router would not run a worker's hosted tools for an agent.
 *
 * Kept apart from RouterError because no request failed: the socket is fine, and asking
 * again would only be told the same thing.
 */
export class HostingRefusedError extends Error {
  /** The agent the tools were offered for. */
  readonly agentId: string;
  /** Why the router said no. */
  readonly reason: string;

  constructor(agentId: string, reason: string) {
    super(`the router refused to host tools for agent ${agentId}: ${reason}`);
    this.name = "HostingRefusedError";
    this.agentId = agentId;
    this.reason = reason;
  }
}

/**
 * The SDK was configured in a way that cannot work.
 *
 * Kept apart from RouterError because nothing was sent: there is no status, and the fix is
 * here rather than at the other end.
 */
export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationError";
  }
}

/**
 * Turns a response that was not a success into the error it reports.
 *
 * Every failure the router answers is `{"error": {"message", "type", "code", "doc_url"}}`,
 * but a 502 from something in front of it, an empty body or an older router's
 * `{"error": "..."}` is not, so those keep their text as the message and leave the rest
 * undefined rather than fail to parse: a JSON error in place of the router's would hide
 * what went wrong. The request id is read either way, since a proxy may still pass it on.
 */
export async function errorOf(
  response: Response,
  operation: string,
  answeredBy = "the router",
): Promise<RouterError> {
  const requestId = response.headers.get("X-Request-Id") || undefined;
  const text = await response.text().catch(() => "");
  const error = envelopeOf(text);
  const message = stringOf(error?.["message"]);
  if (!error || !message) {
    return new RouterError(
      response.status,
      operation,
      text.trim() || `${answeredBy} answered ${response.status}`,
      { requestId },
    );
  }
  return new RouterError(response.status, operation, message, {
    type: stringOf(error["type"]),
    code: stringOf(error["code"]),
    docUrl: stringOf(error["doc_url"]),
    requestId,
  });
}

/** The envelope's `error` object, or undefined when the body is anything else. */
function envelopeOf(text: string): Record<string, unknown> | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return undefined;
  }
  return objectOf(objectOf(parsed)?.["error"]);
}

function objectOf(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function stringOf(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}
