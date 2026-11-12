/**
 * Stream Chat for `session.chat()`, opted into by importing this once:
 *
 * ```ts
 * import "@stream-io/stream-agents/chat";
 * ```
 *
 * Its own entry point so `stream-chat` is resolved only for an app that asked for it.
 */

import * as streamChat from "stream-chat";

import { registerChat, type ChatConstructor, type ChatVersion } from "./stream.js";

// `version` is exported from stream-chat 10.0.2 on, and not at all by 9. Read through a plain
// object rather than as a named import, which 9 would fail to link and a bundler would warn on.
const installed: { readonly StreamChat: ChatConstructor; readonly version?: ChatVersion } = streamChat;

registerChat(installed.StreamChat, installed.version);

export {
  AGENT_CHAT_PAGE_SIZE,
  AgentChat,
  AgentChatStore,
  type AgentActivity,
  type AgentChatMessage,
  type AgentChatOptions,
  type AgentChatSendOptions,
  type AgentChatState,
  type OpenAgentSession,
} from "./agent-chat.js";
export {
  AGENT_ACTIVITY_STATES,
  agentChatTransport,
  type AgentChatTransportOptions,
  type AgentTransport,
  type AgentTransportActivity,
  type AgentTransportFeed,
  type AgentTransportRequest,
} from "./agent-chat-transport.js";
