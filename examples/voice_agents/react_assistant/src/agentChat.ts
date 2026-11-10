import { LiveReasoning, type Session, type UIAction } from "@stream-io/stream-agents";
import type { Channel, LocalMessage } from "stream-chat";
import type { AIMessageRating } from "stream-chat-react/ai-components";

import { messageOf } from "./errors";
import { photoSources } from "./photos";
import { Store } from "./store";
import { troubleText, workingLabel } from "./voicePresence";

/** How many messages are read at a time. */
export const PAGE_SIZE = 30;

/** What the agent is doing while its reply shows nothing yet. */
export type Activity = { kind: "thinking" } | { kind: "working"; text: string };

export interface AgentChatSnapshot {
  /** The channel's messages, oldest first. */
  readonly messages: readonly LocalMessage[];
  /** Bumped whenever more of a reply's live thinking arrives, so the list redraws it. */
  readonly reasoningVersion: number;
  /** Whether the channel holds messages older than those loaded. */
  readonly hasOlderMessages: boolean;
  /** How the person rated replies here, by message. Kept for the session only: the router takes no ratings yet. */
  readonly ratings: Readonly<Record<string, AIMessageRating>>;
  /** Why asking failed, or undefined. */
  readonly failure: string | undefined;
  /** The newest message when something was last asked here, until it fails or is stopped. */
  readonly askedAfter: string | undefined;
}

/**
 * A written conversation as Stream's AI components want it: the channel it is kept in, the whole
 * of each reply's thinking, and what the agent is doing before it writes.
 *
 * The router writes everything here. Asking goes through the session, which records the question
 * in the channel and answers it there: the reply's text, and its steps (each round of reasoning
 * and each tool call) as attachments. The session's socket is what runs this page's tools and
 * carries the person's answers to the ones that ask first, on a call too.
 */
export class AgentChat extends Store<AgentChatSnapshot> {
  /** Each reply's thinking in full, from its live updates. A stored step only keeps its opening. */
  readonly reasoning = new LiveReasoning();

  private readonly stops: (() => void)[] = [];

  private constructor(
    readonly session: Session,
    private readonly channel: Channel,
  ) {
    super({
      messages: [],
      reasoningVersion: 0,
      hasOlderMessages: false,
      ratings: {},
      failure: undefined,
      askedAfter: undefined,
    });
  }

  /**
   * Opens the session's channel and starts following it. The chat owns the session from here: one
   * that cannot be opened lets go of what it started following and closes the session, since
   * nobody else holds either.
   */
  static async open(session: Session): Promise<AgentChat> {
    const { channel } = await session.chat();
    const chat = new AgentChat(session, channel as Channel);
    try {
      await chat.follow();
    } catch (cause) {
      await chat.close().catch(() => undefined);
      throw cause;
    }
    return chat;
  }

  /** What the agent is doing while its reply has nothing to show, or undefined. */
  activity(): Activity | undefined {
    const state = this.session.state.get();
    if (state.kind === "working") {
      return { kind: "working", text: workingLabel(state.skills) };
    }
    if (state.kind === "responding") {
      return this.replyHasBegun() ? undefined : { kind: "thinking" };
    }
    // The session can pass through idle between the question and the first of its reply, which
    // would hide the activity for a moment.
    return this.isAwaitingReply() ? { kind: "thinking" } : undefined;
  }

  /** Whether a reply is on its way, from the first thought to the last word. */
  isAnswering(): boolean {
    const state = this.session.state.get().kind;
    if (state === "responding" || state === "working") {
      return true;
    }
    const lastReply = [...this.get().messages].reverse().find(isAgentReply);
    return lastReply?.custom.generating === true;
  }

  /**
   * Asks, with the photos picked for it and, when `thinkHarder`, more time to reason. A question
   * that is what the person did in a reply's UI carries it as `uiAction`, which the router keeps
   * on their message and tells the model after their words.
   */
  async ask(text: string, photos: readonly File[] = [], thinkHarder = false, uiAction?: UIAction): Promise<void> {
    const messages = this.get().messages;
    this.update({ failure: undefined, askedAfter: messages[messages.length - 1]?.id ?? "" });
    try {
      const images = await photoSources(photos);
      await this.session.responses.create(text, {
        ...(images.length > 0 ? { images } : {}),
        ...(thinkHarder ? { reasoningEffort: "high" } : {}),
        ...(uiAction ? { uiAction } : {}),
      });
    } catch (cause) {
      this.update({ failure: messageOf(cause), askedAfter: undefined });
    }
  }

  /** Stops the reply in flight. What was written so far is kept. */
  stop(): void {
    this.update({ askedAfter: undefined });
    try {
      this.session.interrupt();
    } catch {
      // The reply finished first, which is what stopping it was for.
    }
  }

  /** Loads the messages before the oldest one loaded. The paginator's state says when they are in. */
  async loadOlderMessages(): Promise<void> {
    if (!this.channel.messagePaginator.hasMoreTail) return;
    try {
      await this.channel.messagePaginator.toTail();
    } catch (cause) {
      this.update({ failure: messageOf(cause) });
    }
  }

  /** Rates a reply, or takes the rating back with null. */
  rate(reply: string, rating: AIMessageRating | null): void {
    const ratings = { ...this.get().ratings };
    if (rating) {
      ratings[reply] = rating;
    } else {
      delete ratings[reply];
    }
    this.update({ ratings });
  }

  async close(): Promise<void> {
    for (const stop of this.stops.splice(0)) {
      stop();
    }
    await this.session.close();
  }

  private async follow(): Promise<void> {
    // Every update rather than the list kept for us: each live update carries its own window of
    // the reply's thinking, and folding them into one would lose all but the last.
    const updates = this.channel.on("message.updated", (event) => {
      this.reasoning.read(event.message);
      this.update({ reasoningVersion: this.get().reasoningVersion + 1 });
    });
    this.stops.push(() => updates.unsubscribe());
    // The messages are the paginator's: new, updated and deleted ones, and older pages, land there.
    const paginator = this.channel.messagePaginator;
    paginator.pageSize = PAGE_SIZE;
    this.stops.push(
      paginator.state.subscribeWithSelector(
        ({ items, hasMoreTail }) => ({ items, hasMoreTail }),
        ({ items, hasMoreTail }) => this.update({ messages: [...(items ?? [])], hasOlderMessages: hasMoreTail }),
      ),
    );
    // What the agent is doing changes the activity and whether it is answering.
    this.stops.push(this.session.state.subscribe(() => this.update({})));
    // A turn the agent could not answer leaves an empty reply, so say why rather than show nothing.
    let failureAt = this.session.conversation.get().failureAt;
    this.stops.push(
      this.session.conversation.subscribe(() => {
        const failure = this.session.conversation.get();
        if (failure.failureAt <= failureAt) return;
        failureAt = failure.failureAt;
        this.update({ failure: troubleText(failure.failureContext), askedAfter: undefined });
      }),
    );
    // On screen, so it is reloaded after a reconnect: a channel that is only watched is not.
    this.stops.push(this.channel.activate());

    await this.channel.watch({ messages: { limit: PAGE_SIZE } });
  }

  /** Whether the reply to the latest question shows anything yet: a step or some text. */
  private replyHasBegun(): boolean {
    const messages = this.get().messages;
    const last = messages[messages.length - 1];
    return last !== undefined && isAgentReply(last) && !isEmptyReply(last);
  }

  /** Whether the question asked here is in the channel and its reply shows nothing yet. */
  private isAwaitingReply(): boolean {
    const { askedAfter, messages } = this.get();
    const last = messages[messages.length - 1];
    if (askedAfter === undefined || !last || last.id === askedAfter) return false;
    return !isAgentReply(last) || (last.custom.generating === true && !this.replyHasBegun());
  }
}

/** A reply the agent wrote, which Stream's AI components render. */
export function isAgentReply(message: LocalMessage): boolean {
  return message.custom.ai_generated === true;
}

/** A reply with nothing to show yet: no text, no steps and no UI. */
function isEmptyReply(message: LocalMessage): boolean {
  return (
    !message.text &&
    !message.custom.ai_ui &&
    !(message.attachments ?? []).some((each) => each.type === "ai_reasoning" || each.type === "ai_tool_call")
  );
}

declare module "stream-chat" {
  interface CustomMessageData {
    ai_generated?: boolean;
    generating?: boolean;
    /** The UI the agent wrote for a reply, which shows in place of its text. */
    ai_ui?: { v?: number; format: string; version?: string; status?: string; content: unknown };
    /** What the person did in a reply's UI, on their message that says it. */
    ai_ui_action?: Record<string, unknown>;
  }
}
