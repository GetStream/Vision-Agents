import type { Client, Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** Everything about a conversation except which one it is. */
export type ConversationRequest = Omit<
  Schemas["CreateSessionRequest"],
  "text" | "persist_conversation" | "agent_id" | "conversation_id"
>;

export interface ConversationOptions extends ConversationRequest {
  /**
   * Your own name for this conversation, stable across the sessions that hold it.
   *
   * It is what a running session is found by, both here and by the backend when a message
   * arrives for one, so two people's conversations must not share it.
   */
  id: string;
  /**
   * The `conversation_id` an earlier open returned, which resumes that channel.
   *
   * Left out, the backend names a channel of its own and hands it back. It cannot be
   * named up front: naming one is a resume, and a resume reads the channel without
   * creating it, so a name nothing has been held in yet is refused.
   */
  conversationId?: string;
}

/**
 * The session holding a conversation, opening one if none is.
 *
 * Text conversations are kept in Stream Chat, so what is said outlives the session that
 * heard it. Two things follow, and they are what this is for. The channel belongs to the
 * backend, which means the first open takes no channel and every later one takes the
 * channel the first was given. And a resume has to come back as the same `id` it left as,
 * because the backend checks a conversation is being reopened by whoever held it.
 *
 * ```ts
 * const session = await conversation(api, { id, conversationId: stored, config_id });
 * // Keep session.conversation_id: it is the channel, and the way back to what was said.
 * ```
 *
 * An anonymous caller that goes by no name cannot be found this way. The backend tells
 * such a caller about no sessions at all — listing them would hand one stranger another's
 * — so hold onto the session id and read it back with `getSession` instead.
 */
export async function conversation(
  client: Client,
  options: ConversationOptions,
): Promise<Schemas["Session"]> {
  const { id, conversationId, ...request } = options;
  if (!id) {
    throw new ConfigurationError("a conversation needs an id to be found by");
  }
  if (conversationId !== undefined && !conversationId.startsWith("agent:")) {
    throw new ConfigurationError(
      `${JSON.stringify(conversationId)} is not a conversation id; pass back the ` +
        "conversation_id an open returned, which the backend writes as agent:<channel>",
    );
  }

  const running = await client.get("/v1/agents/sessions");
  const held = running.find((session) => session.agent_id === id);
  if (held) {
    return held;
  }

  return client.post("/v1/agents/sessions", {
    body: {
      ...request,
      text: true,
      persist_conversation: true,
      agent_id: id,
      ...(conversationId ? { conversation_id: conversationId } : {}),
    },
  });
}
