import type { AgentActivity, AgentChat, AgentChatMessage } from "./agent-chat.js";
import {
  branchErrorOf,
  onRewound,
  supportBranchOf,
  type BranchErrorCode,
  type RewoundTo,
  type SupportBranch,
} from "./branches.js";
import type { Schemas } from "./client.js";
import { RouterError } from "./errors.js";
import { messageField } from "./message-field.js";
import type { ImageSource, UIAction } from "./responses.js";
import type { Session } from "./session.js";

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
export interface AgentTransportRequest<Message = unknown> {
  /** `message`, `regenerate` or `edit`, or a kind of the app's own, sent like a message. */
  kind: string;
  /**
   * For a `regenerate`, the reply to answer again; for an `edit`, the person's message to ask
   * differently. A `regenerate` without one retries a question the agent couldn't answer.
   */
  message?: Message | undefined;
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
  send(request: AgentTransportRequest<Message>): Promise<void>;
  stop(): Promise<void>;
  loadOlderMessages(): Promise<void>;
  /**
   * Shows the alternative before or after `message`, the ‹ i/n › of its `branch`. Present, it
   * has the controller offer regenerate on every reply and edit on every person's message.
   * Once `signal` aborts, it shows nothing of that version.
   */
  switchBranch?: (message: Message, to: "previous" | "next", signal?: AbortSignal) => Promise<void>;
}

/**
 * Why a regenerate, an edit, a question or a ‹ i/n › switch couldn't go through, in words to
 * show the person. `cause` is the router's error, when it was one.
 */
export class AgentBranchError extends Error {
  /**
   * The router's code, or `not_branchable` for a message without a place among alternatives:
   * one from before branches, one typed during a call, or one somebody else wrote.
   */
  readonly code: BranchErrorCode | "not_branchable";

  constructor(code: BranchErrorCode | "not_branchable", cause?: unknown) {
    super(BRANCH_ERROR_TEXT[code], cause === undefined ? undefined : { cause });
    this.name = "AgentBranchError";
    this.code = code;
  }
}

const BRANCH_ERROR_TEXT: Record<BranchErrorCode | "not_branchable", string> = {
  not_branchable:
    "This message is from before replies could be regenerated or edited, or was written during a call, so it can't be branched",
  branch_limit_reached: "This conversation can't be regenerated or edited any more: continue in a new chat",
  branch_too_long: "This version of the conversation is too long to continue from: continue in a new chat",
  branch_unavailable: "This version of the conversation is no longer available",
};

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
 * It branches. A `regenerate` of a reply answers its question again and an `edit` of the
 * person's message asks it differently, both beside the earlier version, which Chat hides and
 * the router keeps; the hidden messages drop out of the feed as soon as the router answers.
 * `switchBranch` shows another version, ‹ i/n ›, from the router's copy without writing to
 * Chat; asking anything while one is shown, or regenerating or editing a message of it, first
 * makes it the conversation again (`responses.activate`). A `channel.rewound`, or the message the
 * versions part at leaving the channel, goes back to the channel's own.
 *
 * Only messages the router marked with `support_branch` can be branched: one from before
 * branches, or typed during a call, is refused with an `AgentBranchError` (`not_branchable`), as
 * are the router's `branch_limit_reached`, `branch_too_long` and `branch_unavailable`. A
 * `regenerate` with no message retries a question the agent couldn't answer, asked again.
 *
 * ```ts
 * const conversation = new AIConversationController({
 *   transport: agentChatTransport(chat, { adapt: (message, reasoningText) => fromStreamMessage(message, { reasoningText }) }),
 * });
 * ```
 */
export function agentChatTransport<M extends AgentChatMessage, Message>(
  chat: AgentChat<M>,
  options: AgentChatTransportOptions<M, Message>,
): Required<AgentTransport<Message>> {
  const reasoningText = (messageId: string, stepId: string) => chat.reasoning.get(messageId, stepId);
  const workingText = options.workingText ?? defaultWorkingText;
  // What the latest question went out with, which asking it again keeps: its photos, what the
  // person did in a reply's UI, and how long the model may think.
  let lastAsked: { text: string; images: ImageSource[]; uiAction: UIAction | undefined; reasoningEffort: string | undefined } | undefined;
  // The other version of the conversation shown in place of the channel's, from a message on.
  let overlay: Overlay<M> | undefined;
  // Bumped by each switch and each send, so a switch that finishes after either shows nothing.
  let switches = 0;
  // What a regenerate, an edit or an activation of this device hid. The channel drops them too,
  // but stream-chat 9 says nothing when it does, so the feed would show them until its next event.
  // A hidden message never comes back under its id: one brought back is sent again as a new one.
  const hidden = new Set<string>();
  // The response the latest regenerate or edit went through, by the message it targets, which
  // its retry targets again though the first attempt may have hidden it: the one replaced while
  // it is asked, then the one that took its place, which a retry under a fresh request id, after
  // the agent couldn't answer, regenerates or edits. Only the latest, as only it can be retried.
  let lastBranch: { readonly targetId: string; readonly responseId: string } | undefined;
  // Each connection's way back from the messages it fed to the Stream messages they were made of.
  const sources = new Set<(message: Message) => M | undefined>();
  // Each connection's feed, told again when the view changes outside the channel.
  const refreshes = new Set<() => void>();
  const refresh = () => {
    for (const again of [...refreshes]) {
      again();
    }
  };

  /** The channel's messages without what this device hid. */
  const visible = (): readonly M[] => {
    const { messages } = chat.state.getLatestValue();
    return hidden.size > 0 ? messages.filter((message) => !hidden.has(message.id)) : messages;
  };

  /** What the conversation shows: the channel's messages, or another version from where they part. */
  const shown = (messages: readonly M[]): readonly M[] => {
    const kept = hidden.size > 0 ? messages.filter((message) => !hidden.has(message.id)) : messages;
    if (!overlay) {
      return kept;
    }
    const fork = kept.findIndex((message) => message.id === overlay?.forkId);
    if (fork < 0) {
      // The message the versions part at left the channel: the conversation changed under it.
      overlay = undefined;
      return kept;
    }
    return [...kept.slice(0, fork), ...overlay.items];
  };

  /** The Stream message a message the controller holds was made of. */
  const sourceOf = (message: Message | undefined): M | undefined => {
    if (message === undefined) {
      return undefined;
    }
    for (const lookup of sources) {
      const source = lookup(message);
      if (source) {
        return source;
      }
    }
    // A message the app made itself, from the same id.
    const id = typeof message === "object" && message !== null ? (message as { id?: unknown }).id : undefined;
    return typeof id === "string" ? shown(chat.state.getLatestValue().messages).find((one) => one.id === id) : undefined;
  };

  /**
   * Runs a regenerate, an edit or an activation, and drops from the feed what it hid of the
   * messages shown before it went, at once, as the session drops them from the channel.
   */
  const rewinding = async (run: () => Promise<RewoundTo | undefined>): Promise<void> => {
    const before = visible();
    let to: RewoundTo | undefined;
    try {
      to = await run();
    } catch (error) {
      throw branchError(error);
    }
    const at = to ? before.findIndex((message) => message.id === to.message_id) : -1;
    if (to && at >= 0) {
      for (const message of before.slice(to.inclusive ? at : at + 1)) {
        hidden.add(message.id);
      }
    }
    refresh();
  };

  /**
   * Waits until the channel holds `count` messages sent again that `before` did not have. The
   * controller takes a finished message arriving after `send` resolved for the new reply, so the
   * copies an activation sends again have to be in the feed before it resolves.
   */
  const resentArrived = async (before: ReadonlySet<string>, count: number): Promise<void> => {
    const arrived = (messages: readonly M[]) =>
      messages.filter((message) => !before.has(message.id) && messageField(message, "resent_from") !== undefined).length >= count;
    let unsubscribe: (() => void) | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const done = await new Promise<boolean>((resolve) => {
      // The router has written the copies to Chat before it answers, so only the socket is left
      // to bring them: well under a second. Three bound a slow socket without holding the person's
      // question long when it dropped them; the read below brings them then.
      timer = setTimeout(() => resolve(false), RESENT_TIMEOUT_MS);
      unsubscribe = chat.state.subscribe((state) => {
        if (arrived(state.messages)) {
          resolve(true);
        }
      });
    });
    clearTimeout(timer);
    unsubscribe?.();
    if (!done) {
      let gaveUp: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          chat.readLatest(),
          new Promise<void>((resolve) => {
            gaveUp = setTimeout(resolve, READ_LATEST_TIMEOUT_MS);
          }),
        ]);
      } catch {
        // What is asked does not wait on the copies: they show with the channel's next read.
      } finally {
        clearTimeout(gaveUp);
      }
    }
  };

  /** Makes the version shown the conversation again, before anything is asked on it. */
  const activate = async (session: Session, viewed: Overlay<M>): Promise<void> => {
    const before = new Set(chat.state.getLatestValue().messages.map((message) => message.id));
    let resent = 0;
    await rewinding(async () => {
      const activated = await session.responses.activate(viewed.leafResponseId);
      resent = activated.resent;
      return activated.rewound_to;
    });
    if (resent > 0) {
      await resentArrived(before, resent);
    }
    if (overlay === viewed) {
      overlay = undefined;
    }
    refresh();
  };

  /** Leaves the version shown for the channel's own, before branching off a message of it. */
  const leaveOverlayFor = async (session: Session, source: M): Promise<void> => {
    const viewed = overlay;
    if (!viewed) {
      return;
    }
    if (viewed.items.includes(source)) {
      // The router refuses to branch off a response that is not on the conversation.
      await activate(session, viewed);
      return;
    }
    overlay = undefined;
    refresh();
  };

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
      let fed = new Map<Message, M>();
      const lookup = (message: Message) => fed.get(message);
      const render = (messages: readonly M[]) => {
        const next = new Map<Message, M>();
        const made = shown(messages).map((message) => {
          const one = adapt(message);
          next.set(one, message);
          return one;
        });
        fed = next;
        feed.updateMessages(made);
      };
      const again = () => {
        if (feed.isConnected) {
          render(chat.state.getLatestValue().messages);
        }
      };
      sources.add(lookup);
      refreshes.add(again);
      // A rewind of the channel, from any device, shows the channel's own version again.
      let followed: Session | undefined;
      let unfollow: (() => void) | undefined;
      let disconnected = false;
      const follow = (session: Session | undefined) => {
        if (session === followed) {
          return;
        }
        followed = session;
        unfollow?.();
        unfollow = undefined;
        session
          ?.chat()
          .then(({ client, channel }) => {
            const cid = (channel as { cid?: unknown } | null)?.cid;
            if (disconnected || followed !== session || typeof cid !== "string") {
              return;
            }
            unfollow = onRewound(client, cid, () => {
              if (overlay) {
                overlay = undefined;
                refresh();
              }
            });
          })
          // Without the channel there is nothing shown to rewind.
          .catch(() => undefined);
      };
      let previous:
        | { messages: readonly M[]; reasoningVersion: number; activity: AgentActivity | undefined; hasOlderMessages: boolean }
        | undefined;
      const stop = chat.state.subscribe((state) => {
        if (!feed.isConnected) {
          return;
        }
        follow(state.session);
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
          render(state.messages);
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
      return () => {
        disconnected = true;
        stop();
        unfollow?.();
        sources.delete(lookup);
        refreshes.delete(again);
      };
    },
    async send(request) {
      // A switch still reading would land its version over what is sent now: it shows nothing.
      // The version already shown stays until what is sent replaces it, and if it fails.
      switches++;
      if (request.kind === "regenerate" && request.message === undefined) {
        // A question the agent couldn't answer, asked again as a new question under a new request
        // id: under the original one, the router would take it for the question it already
        // answered. It keeps what the question went out with, which a regenerate doesn't carry.
        const again = lastAsked?.text === request.text ? lastAsked : undefined;
        const reasoningEffort = options.reasoningEffort?.(request.chatOption) ?? again?.reasoningEffort;
        await chat.send(request.text, {
          ...(again && again.images.length > 0 ? { images: again.images } : {}),
          ...(reasoningEffort ? { reasoningEffort } : {}),
          ...(again?.uiAction ? { uiAction: again.uiAction } : {}),
        });
        return;
      }
      if (request.kind === "regenerate" || request.kind === "edit") {
        const source = sourceOf(request.message);
        const place = source ? supportBranchOf(source) : undefined;
        const targetId = idOf(request.message);
        const retried = targetId !== undefined && lastBranch?.targetId === targetId ? lastBranch.responseId : undefined;
        let responseId: string;
        if (retried !== undefined) {
          responseId = retried;
        } else if (source && place) {
          responseId =
            request.kind === "edit" ? answeringOf(shown(chat.state.getLatestValue().messages), source, place) : place.responseId;
        } else {
          throw new AgentBranchError("not_branchable");
        }
        lastBranch = targetId === undefined ? undefined : { targetId, responseId };
        const remembered = lastBranch;
        const session = await chat.use();
        if (source) {
          await leaveOverlayFor(session, source);
        }
        const reasoningEffort = options.reasoningEffort?.(request.chatOption);
        const effort = reasoningEffort ? { reasoningEffort } : {};
        const replacing = rewinding(async () => {
          const created =
            request.kind === "edit"
              ? await session.responses.edit(responseId, request.text, {
                  ...effort,
                  // The new question is stored under it, so it replaces the one on its way. The
                  // edited message's own id is taken already.
                  ...(request.messageId && request.messageId !== source?.id ? { requestId: request.messageId } : {}),
                })
              : // The controller's id is new for each regenerate and kept by its retry, which the
                // router then answers as the one it already took.
                await session.responses.regenerate(responseId, {
                  ...effort,
                  ...(request.messageId ? { requestId: request.messageId } : {}),
                });
          if (created.created.id && remembered && lastBranch === remembered) {
            lastBranch = { targetId: remembered.targetId, responseId: created.created.id };
          }
          return created.created.rewound_to;
        });
        try {
          await replacing;
        } catch (error) {
          // The response is gone from the conversation, or it changed: the next attempt reads
          // which response to go through from the conversation again.
          if (remembered && lastBranch?.targetId === remembered.targetId && stale(error)) {
            lastBranch = undefined;
          }
          throw error;
        }
        return;
      }
      const images = await imagesOf(request.attachments, options.images);
      const reasoningEffort = options.reasoningEffort?.(request.chatOption);
      const uiAction = uiActionOf(request.custom);
      const viewed = overlay;
      if (viewed) {
        // Asked on the version shown, which the agent answers from only once it is the conversation.
        await activate(await chat.use(), viewed);
      }
      lastAsked = { text: request.text, images, uiAction, reasoningEffort };
      await chat.send(request.text, {
        ...(images.length > 0 ? { images } : {}),
        ...(reasoningEffort ? { reasoningEffort } : {}),
        ...(uiAction ? { uiAction } : {}),
        requestId: request.messageId,
      });
    },
    async switchBranch(message, to, signal) {
      const source = sourceOf(message);
      const place = source ? supportBranchOf(source) : undefined;
      if (!source || !place) {
        throw new AgentBranchError("not_branchable");
      }
      const switching = ++switches;
      // Whether a later switch or a send, or the controller through `signal`, took its place.
      const dropped = () => switching !== switches || signal?.aborted === true;
      const reading = signal ? { signal } : {};
      const session = await chat.session();
      const siblings: Schemas["ResponseSibling"][] = [];
      try {
        for await (const sibling of session.responses.unwindSiblings(place.responseId, { of: place.of, ...reading })) {
          siblings.push(sibling);
        }
      } catch (error) {
        if (dropped()) {
          return;
        }
        throw branchError(error);
      }
      // Where the message stands now, which its mark, written when it was, may not say.
      const self = place.of === "question" ? place.questionResponseId || place.responseId : place.responseId;
      const index = siblings.find((sibling) => sibling.id === self)?.index ?? place.index;
      const target = siblings.find((sibling) => sibling.index === index + (to === "next" ? 1 : -1));
      if (!target || dropped()) {
        return;
      }
      if (target.active) {
        overlay = undefined;
        refresh();
        return;
      }
      const read: Schemas["BranchMessage"][] = [];
      try {
        for await (const item of session.responses.unwindBranch(target.id, reading)) {
          read.push(item);
        }
      } catch (error) {
        if (dropped()) {
          return;
        }
        throw branchError(error);
      }
      if (dropped()) {
        return;
      }
      const first = read[0];
      if (!first) {
        // It became the conversation while it was read.
        overlay = undefined;
        refresh();
        return;
      }
      const channel = visible();
      const forkId = forkOf(channel, first.id, overlay, source);
      if (forkId === undefined) {
        throw new Error("The message isn't in the conversation shown");
      }
      const items = read.map((item) => streamMessageOf(item, channel[0]) as unknown as M);
      const leaf = items.findLast((item) => supportBranchOf(item) !== undefined);
      overlay = { forkId, items, leafResponseId: (leaf && supportBranchOf(leaf)?.responseId) || target.id };
      refresh();
    },
    stop: () => chat.stop(),
    loadOlderMessages: () => chat.loadOlderMessages(),
  };
}

/** How long an activation waits for the copies it sent again to arrive, before reading the channel. */
const RESENT_TIMEOUT_MS = 3000;

/**
 * How long that read may take before what is asked goes without it: stream-chat's own request
 * timeout by default, so only a read that would hang anyway, as one waiting on a socket that
 * never connects, is given up on. The read carries on, and shows what it brings when it lands.
 */
const READ_LATEST_TIMEOUT_MS = 3000;

/**
 * The response an edit of the person's message goes through: the one whose reply is shown after
 * it, answering the same question. The question keeps the mark of the response that first
 * answered it, which regenerating the reply hides, and the router edits only through a response
 * on the conversation. The question's own when no reply to it is shown.
 */
function answeringOf<M extends AgentChatMessage>(shown: readonly M[], question: M, place: SupportBranch): string {
  const asked = place.questionResponseId || place.responseId;
  const at = shown.indexOf(question);
  for (const message of at < 0 ? [] : shown.slice(at + 1)) {
    const mark = supportBranchOf(message);
    if (mark?.of === "question") {
      break;
    }
    if (mark?.of === "reply" && (mark.questionResponseId || mark.responseId) === asked) {
      return mark.responseId;
    }
  }
  return place.responseId;
}

/** Another version of the conversation, shown in place of the channel's from one message on. */
interface Overlay<M> {
  /** The channel's message the versions part at: it and everything after are not shown. */
  readonly forkId: string;
  /** The version's messages from there, as the router copied them, made Stream messages. */
  readonly items: readonly M[];
  /** Its newest response, which bringing it back ends at. */
  readonly leafResponseId: string;
}

/**
 * The channel's message a branch read back parts from the channel at. The router's copy starts
 * where the branch leaves the conversation, which can be at a message the channel still shows,
 * such as the question of a regenerated reply, under its id or a copy's. Otherwise it parts at
 * the message switched from, or where the version shown parts, for a message of that version.
 */
function forkOf<M extends AgentChatMessage>(
  channel: readonly M[],
  firstId: string,
  overlay: Overlay<M> | undefined,
  source: M,
): string | undefined {
  const shared = channel.find((message) => message.id === firstId || messageField(message, "resent_from") === firstId);
  if (shared) {
    return shared.id;
  }
  if (overlay?.items.includes(source)) {
    return overlay.forkId;
  }
  return channel.some((message) => message.id === source.id) ? source.id : undefined;
}

/**
 * A message of a branch read back, as the channel's own messages are: the app's fields top
 * level and times as Dates in stream-chat 9; under `custom` and in nanoseconds in 10, which
 * `like`, one of the channel's messages, shows by its `created_at`.
 */
function streamMessageOf(item: Schemas["BranchMessage"], like: object | undefined): AgentChatMessage & Record<string, unknown> {
  const at = Date.parse(item.created_at);
  const cid = (like as { cid?: unknown } | undefined)?.cid;
  const common = {
    id: item.id,
    text: item.text,
    type: "regular",
    status: "received",
    attachments: [...item.attachments],
    user: { id: item.user_id },
    ...(typeof cid === "string" ? { cid } : {}),
  };
  if (typeof (like as { created_at?: unknown } | undefined)?.created_at === "number") {
    return { ...common, custom: { ...item.custom }, created_at: at * 1e6, updated_at: at * 1e6 };
  }
  return { ...item.custom, ...common, created_at: new Date(at), updated_at: new Date(at) };
}

/** The id of a message the controller holds, or undefined for one without. */
function idOf(message: unknown): string | undefined {
  const id = typeof message === "object" && message !== null ? (message as { id?: unknown }).id : undefined;
  return typeof id === "string" && id ? id : undefined;
}

/**
 * Whether the router refused a regenerate or an edit for the response it went through: no
 * longer on the conversation, or the conversation changed (409 without a branch code), or not
 * known (404).
 */
function stale(error: unknown): boolean {
  return error instanceof RouterError && ((error.status === 409 && branchErrorOf(error) === undefined) || error.status === 404);
}

/** A router refusal of a branch in the person's words, or the error as it was. */
function branchError(error: unknown): unknown {
  const code = branchErrorOf(error);
  return code ? new AgentBranchError(code, error) : error;
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
