import type { AgentActivity, AgentChat, AgentChatMessage } from "./agent-chat.js";
import type { ImageSource, UIAction } from "./responses.js";

/**
 * The shapes Stream's AI components take for a conversation's transport: `AIConversationController`
 * in `stream-chat-react-native` and `stream-chat-react`. Declared here rather than imported, so
 * this package depends on neither: a value of these shapes is one of theirs.
 */

/** What the agent says it is doing, with the state names of Stream Chat's AI indicator. */
export interface AgentTransportActivity {
  state: string;
  text?: string;
}

/** Where the transport tells the conversation what it holds. */
export interface AgentTransportFeed<Message> {
  readonly isConnected: boolean;
  updateMessages(messages: readonly Message[]): void;
  updateHasOlderMessages(hasOlderMessages: boolean): void;
  updateActivity(activity: AgentTransportActivity | undefined): void;
}

/** What the person asks of the agent. */
export interface AgentTransportRequest {
  /** `message`, `regenerate` or `edit`, or a kind of the app's own, sent like a message. */
  kind: string;
  /** An id for the person's message, which a retry keeps. */
  messageId: string;
  text: string;
  /** The files the person picked, on this device. */
  attachments: readonly unknown[];
  /** The option picked in the composer, such as thinking harder. */
  chatOption?: { id: string } | undefined;
  /** What the composer held besides. A generated-UI action travels as `custom.uiAction`. */
  custom?: Record<string, unknown> | undefined;
}

/** How the conversation reaches the agent. */
export interface AgentTransport<Message> {
  connect(feed: AgentTransportFeed<Message>): () => void;
  send(request: AgentTransportRequest): Promise<void>;
  stop(): Promise<void>;
  loadOlderMessages(): Promise<void>;
}

/** The ids of the states Stream Chat's AI indicator reports, which the AI components read. */
export const AGENT_ACTIVITY_STATES = {
  thinking: "AI_STATE_THINKING",
  checkingExternalSources: "AI_STATE_EXTERNAL_SOURCES",
  error: "AI_STATE_ERROR",
} as const;

export interface AgentChatTransportOptions<M extends AgentChatMessage, Message> {
  /**
   * Makes a Stream message the kit's message: `fromStreamMessage` in `stream-chat-react`,
   * `createFromStreamMessage()` in `stream-chat-react-native`. `reasoningText` is each reply's
   * whole thinking, from its live updates.
   */
  adapt(message: M, reasoningText: (messageId: string, stepId: string) => string | undefined): Message;
  /**
   * Reads the files the person picked as images for the model, such as data URIs of shrunk
   * photos. Without it, a request with attachments is refused.
   */
  images?(attachments: readonly unknown[]): Promise<ImageSource[]>;
  /** The reasoning effort a composer option asks for, such as `high` for thinking harder. */
  reasoningEffort?(chatOption: { id: string } | undefined): string | undefined;
  /** What the activity says while skills run. Defaults to "Working on trip planning" and the like. */
  workingText?(skills: readonly string[]): string | undefined;
  /**
   * What the conversation says when the agent couldn't write its reply, by the part that failed.
   * Undefined, the default, leaves the kit's own translated text: the router's own is a provider's
   * raw error, which isn't for people to read.
   */
  failedText?(context: string): string | undefined;
}

/**
 * The transport an `AIConversationController` reaches the agent through: the messages of the
 * channel `chat` reads, what the agent is doing from the session's socket, and asking through the
 * session, which records the question in the channel and answers it there.
 *
 * The router keeps a conversation in Chat and can't rewind it, so give the controller
 * `actions: { edit: false, regenerate: false }`. A `regenerate` then only comes from retrying a
 * question the agent couldn't answer, and is asked again as a new question; an `edit` is refused.
 *
 * ```ts
 * const conversation = new AIConversationController({
 *   transport: agentChatTransport(chat, { adapt: (message, reasoningText) => fromStreamMessage(message, { reasoningText }) }),
 *   actions: { edit: false, regenerate: false },
 * });
 * ```
 */
export function agentChatTransport<M extends AgentChatMessage, Message>(
  chat: AgentChat<M>,
  options: AgentChatTransportOptions<M, Message>,
): AgentTransport<Message> {
  const reasoningText = (messageId: string, stepId: string) => chat.reasoning.get(messageId, stepId);
  const workingText = options.workingText ?? defaultWorkingText;
  // What the latest question went out with, which asking it again keeps: its photos, what the
  // person did in a reply's UI, and how long the model may think.
  let lastAsked: { text: string; images: ImageSource[]; uiAction: UIAction | undefined; reasoningEffort: string | undefined } | undefined;
  return {
    connect(feed) {
      // The same message for each Stream message that didn't change, so only the messages an
      // update touched are made, and shown, afresh.
      let adapted = new WeakMap<M, Message>();
      const adapt = (message: M) => {
        let made = adapted.get(message);
        if (made === undefined) {
          made = options.adapt(message, reasoningText);
          adapted.set(message, made);
        }
        return made;
      };
      let previous:
        | { messages: readonly M[]; reasoningVersion: number; activity: AgentActivity | undefined; hasOlderMessages: boolean }
        | undefined;
      return chat.state.subscribe((state) => {
        if (!feed.isConnected) {
          return;
        }
        const thought = previous !== undefined && previous.reasoningVersion !== state.reasoningVersion;
        if (thought) {
          // More of a reply's thinking arrived: that reply alone is shown afresh.
          const reply = state.messages.find((message) => message.id === state.reasoningMessageId);
          if (reply) {
            adapted.delete(reply);
          } else {
            adapted = new WeakMap();
          }
        }
        if (previous?.messages !== state.messages || thought) {
          feed.updateMessages(state.messages.map(adapt));
        }
        if (previous?.hasOlderMessages !== state.hasOlderMessages) {
          feed.updateHasOlderMessages(state.hasOlderMessages);
        }
        if (previous === undefined || previous.activity !== state.activity) {
          feed.updateActivity(transportActivity(state.activity, workingText, options.failedText));
        }
        previous = {
          messages: state.messages,
          reasoningVersion: state.reasoningVersion,
          activity: state.activity,
          hasOlderMessages: state.hasOlderMessages,
        };
      });
    },
    async send(request) {
      if (request.kind === "edit") {
        throw new Error("The agent can't edit a message it was sent: give the conversation actions: { edit: false }");
      }
      if (request.kind === "regenerate") {
        // Asked again as a new question, under a new request id: under the original one, the
        // router would take it for the question it already answered. It keeps what the question
        // went out with, which a regenerate doesn't carry.
        const again = lastAsked?.text === request.text ? lastAsked : undefined;
        const reasoningEffort = options.reasoningEffort?.(request.chatOption) ?? again?.reasoningEffort;
        await chat.send(request.text, {
          ...(again && again.images.length > 0 ? { images: again.images } : {}),
          ...(reasoningEffort ? { reasoningEffort } : {}),
          ...(again?.uiAction ? { uiAction: again.uiAction } : {}),
        });
        return;
      }
      const images = await imagesOf(request.attachments, options.images);
      const reasoningEffort = options.reasoningEffort?.(request.chatOption);
      const uiAction = uiActionOf(request.custom);
      lastAsked = { text: request.text, images, uiAction, reasoningEffort };
      await chat.send(request.text, {
        ...(images.length > 0 ? { images } : {}),
        ...(reasoningEffort ? { reasoningEffort } : {}),
        ...(uiAction ? { uiAction } : {}),
        requestId: request.messageId,
      });
    },
    stop: () => chat.stop(),
    loadOlderMessages: () => chat.loadOlderMessages(),
  };
}

function defaultWorkingText(skills: readonly string[]): string | undefined {
  return skills.length > 0 ? `Working on ${skills.map((skill) => skill.replace(/_/g, " ")).join(" and ")}` : undefined;
}

function transportActivity(
  activity: AgentActivity | undefined,
  workingText: (skills: readonly string[]) => string | undefined,
  failedText: ((context: string) => string | undefined) | undefined,
): AgentTransportActivity | undefined {
  switch (activity?.kind) {
    case "thinking":
      return { state: AGENT_ACTIVITY_STATES.thinking };
    case "working":
      return withText(AGENT_ACTIVITY_STATES.checkingExternalSources, workingText(activity.skills));
    case "failed":
      return withText(AGENT_ACTIVITY_STATES.error, failedText?.(activity.context));
    default:
      return undefined;
  }
}

function withText(state: string, text: string | undefined): AgentTransportActivity {
  return text ? { state, text } : { state };
}

async function imagesOf(
  attachments: readonly unknown[],
  read: ((attachments: readonly unknown[]) => Promise<ImageSource[]>) | undefined,
): Promise<ImageSource[]> {
  if (attachments.length === 0) {
    return [];
  }
  if (!read) {
    throw new Error("Sending photos needs agentChatTransport's images option");
  }
  return read(attachments);
}

function uiActionOf(custom: Record<string, unknown> | undefined): UIAction | undefined {
  const action = custom?.["uiAction"];
  return typeof action === "object" && action !== null && !Array.isArray(action) ? (action as UIAction) : undefined;
}
