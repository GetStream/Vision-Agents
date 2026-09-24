/**
 * The JavaScript client for the Stream acceleration backend.
 *
 * Everything here runs in a browser and on a server. What only a server may do is not a
 * matter of which file it is in but of what it is authenticated as: a `Backend` holding an
 * API secret reaches every operation, and one holding a user's token reaches the
 * conversations that user owns and nothing about how an agent is configured.
 *
 * Reading an agent directory needs a filesystem, so `loadFolder` lives in
 * `@stream-io/vision-agents/node`.
 */

export {
  API_KEY_ENV,
  API_SECRET_ENV,
  AUTHENTICATE_ENV,
  Backend,
  CUSTOMER_ENV,
  DEFAULT_URL,
  URL_ENV,
  signToken,
  type BackendOptions,
  type StreamCredentials,
  type StreamUser,
  type TokenSource,
  type WebSocketConstructor,
  type WebSocketLike,
} from "./backend.js";

export {
  Client,
  type Method,
  type PathsWith,
  type RequestOptions,
  type Result,
  type Schemas,
} from "./client.js";

export { AgentHandle } from "./handle.js";

export {
  Sessions,
  timestamp,
  type CreateSessionOptions,
  type SessionQuery,
  type SessionSpec,
} from "./sessions.js";

export {
  AgentResponse,
  Items,
  Responses,
  type CreateResponseOptions,
  type ImageSource,
} from "./responses.js";

export {
  GUEST_STORAGE_KEY,
  browserStore,
  claimGuestUser,
  forgetGuest,
  guestUser,
  type Guest,
  type GuestStore,
  type GuestUserOptions,
} from "./guests.js";

export { ConfigurationError, RouterError, SocketClosedError } from "./errors.js";

export type { components, operations, paths } from "./generated/api.js";

export {
  Socket,
  flag,
  nested,
  number,
  text,
  type Frame,
  type Message,
} from "./socket.js";

export { Tools, render, type ParameterSchema, type Tool } from "./tools.js";

export {
  Session,
  eventOf,
  type ForkOptions,
  type Participant,
  type SessionChat,
  type SessionEvent,
  type SessionOptions,
  type SessionVideo,
} from "./session.js";

export {
  Agent,
  USER_KEY,
  daytona,
  userIdOf,
  type AgentOptions,
  type ChatOptions,
  type Declaration,
  type Folder,
  type Harness,
  type Pipeline,
  type Sandbox,
  type Skill,
  type SyncStamp,
} from "./agent.js";

export {
  DEFAULT_CALL_TYPE,
  DEFAULT_MONITOR_URL,
  Edge,
  type Call,
  type EdgeOptions,
  type User,
} from "./edge.js";

export { callOf, messageOf, type InboundCall, type InboundMessage } from "./inbound.js";

export {
  Dispatch,
  type CallHandler,
  type DispatchOptions,
  type MessageHandler,
} from "./dispatch.js";
