import { type Frame, text } from "./socket.js";

/** A call that arrived, as the router hands it to a worker. */
export interface InboundCall {
  /** The Stream call the caller is already in, named for `sessionId`. */
  callId: string;
  callType: string;
  /**
   * The session to open for the call, with voice: the call is named for it. Empty for a
   * number attached before calls were, which has to be attached again.
   */
  sessionId: string;
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
 * A message written to an agent that is not running, or to a running session whose agent
 * leaves text to dispatch.
 *
 * Otherwise one written to an agent that is already running never arrives here: the router
 * answers it from that session, because that agent is the one that knows what has been said
 * so far.
 */
export interface InboundMessage {
  /** The channel it was written in, which is the conversation to answer in. May be empty. */
  channelId: string;
  channelType: string;
  /** Who to answer as, which names the channel replies are written into. */
  agentId: string;
  /** The stored agent config the router matched, if it matched one. */
  configId: string;
  /**
   * The running session it was written to, set when its agent leaves text to dispatch.
   * Nothing has answered it: `dispatch.answer` has the model do so.
   */
  sessionId: string;
  /** The request it was sent as, which the answer is created under. May be empty. */
  requestId: string;
  /** What was written. Empty when a photo was sent without text. */
  text: string;
  /** The photos sent with it, by the URLs the channel keeps them at. Empty or absent when none was. */
  images?: InboundImage[];
  messageId: string;
  userId: string;
  userName: string;
  at?: Date;
}

/** A photo sent with a message. */
export interface InboundImage {
  /** Where it is kept, which the model fetches itself. */
  url: string;
  /** Its type, when the channel said. */
  mimeType?: string;
}

/** Reads a call frame off the wire. */
export function callOf(frame: Frame): InboundCall {
  const at = whenOf(frame["at"]);
  return {
    callId: text(frame, "call_id"),
    callType: text(frame, "call_type") || "default",
    sessionId: text(frame, "session_id"),
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
    sessionId: text(frame, "session_id"),
    requestId: text(frame, "request_id"),
    text: text(frame, "text"),
    images: imagesOf(frame["images"]),
    messageId: text(frame, "message_id"),
    userId: text(frame, "user_id"),
    userName: text(frame, "user_name"),
    ...(at ? { at } : {}),
  };
}

/**
 * Reads the photos a message frame carries. An entry that is not an object or has no URL is
 * dropped, since there is nothing to fetch.
 */
function imagesOf(value: unknown): InboundImage[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const images: InboundImage[] = [];
  for (const entry of value) {
    if (typeof entry !== "object" || entry === null) {
      continue;
    }
    const url = text(entry as Frame, "url");
    const mimeType = text(entry as Frame, "mime_type");
    if (url) {
      images.push({ url, ...(mimeType ? { mimeType } : {}) });
    }
  }
  return images;
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
