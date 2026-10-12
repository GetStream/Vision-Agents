/**
 * Stream Chat for `session.chat()`, opted into by importing this once:
 *
 * ```ts
 * import "@stream-io/vision-agents/chat";
 * ```
 *
 * Its own entry point so `stream-chat` is resolved only for an app that asked for it.
 */

import { StreamChat } from "stream-chat";

import { registerChat } from "./stream.js";

registerChat(StreamChat);
