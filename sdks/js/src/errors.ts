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
