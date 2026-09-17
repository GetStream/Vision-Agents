/** What the router said went wrong. */
export class RouterError extends Error {
  /** The HTTP status, or 0 when the request never got an answer. */
  readonly status: number;
  /** The method and path that was asked for, for a log line that says which call failed. */
  readonly operation: string;

  constructor(status: number, operation: string, message: string) {
    super(message);
    this.name = "RouterError";
    this.status = status;
    this.operation = operation;
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
