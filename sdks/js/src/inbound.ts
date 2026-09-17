import { type Frame, text } from "./socket.js";

/** A call that arrived, as the router hands it to a worker. */
export interface InboundCall {
  /** The Stream call the caller is already in, which is the one to join. */
  callId: string;
  callType: string;
  /** The number they rang, which is what the agent acts from and can transfer on. */
  calledNumber: string;
  /** The number they rang from, where the vendor passed it on. */
  callerNumber: string;
  /** Whatever was put on the Stream call, for a deployment that routes on its own fields. */
  custom: Record<string, string>;
  /** When the call arrived, or undefined when the router sent no timestamp it could read. */
  at?: Date;
}

/**
 * A message written to an agent that is not running.
 *
 * One written to an agent that is already running never arrives here: the router answers it
 * from that session, because that agent is the one that knows what has been said so far.
 */
export interface InboundMessage {
  /** The channel it was written in, which is the conversation to answer in. */
  channelId: string;
  channelType: string;
  /** Who to answer as, which names the channel replies are written into. */
  agentId: string;
  /** The stored agent config the router matched, if it matched one. */
  configId: string;
  text: string;
  messageId: string;
  userId: string;
  userName: string;
  at?: Date;
}

/** Reads a call frame off the wire. */
export function callOf(frame: Frame): InboundCall {
  const at = whenOf(frame["at"]);
  return {
    callId: text(frame, "call_id"),
    callType: text(frame, "call_type") || "default",
    calledNumber: text(frame, "called_number"),
    callerNumber: text(frame, "caller_number"),
    custom: strings(frame["custom"]),
    ...(at ? { at } : {}),
  };
}

/** Reads a message frame off the wire. */
export function messageOf(frame: Frame): InboundMessage {
  const at = whenOf(frame["at"]);
  return {
    channelId: text(frame, "channel_id"),
    channelType: text(frame, "channel_type") || "agent",
    agentId: text(frame, "agent_id"),
    configId: text(frame, "config_id"),
    text: text(frame, "text"),
    messageId: text(frame, "message_id"),
    userId: text(frame, "user_id"),
    userName: text(frame, "user_name"),
    ...(at ? { at } : {}),
  };
}

function strings(value: unknown): Record<string, string> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return {};
  }
  const read: Record<string, string> = {};
  for (const [key, each] of Object.entries(value)) {
    read[key] = String(each);
  }
  return read;
}

/** Reads an RFC 3339 timestamp, tolerating the trailing Z Go writes. */
function whenOf(value: unknown): Date | undefined {
  if (typeof value !== "string") {
    return undefined;
  }
  const when = new Date(value);
  return Number.isNaN(when.getTime()) ? undefined : when;
}
