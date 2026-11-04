import { LiveReasoning, type Session } from "@stream-io/calypso";
import type { Channel, Event, LocalMessage } from "stream-chat";
import type { AIMessageRating } from "stream-chat-react/ai-components";

import { messageOf } from "./errors";
import { photoSources } from "./photos";
import { Store } from "./store";
import { workingLabel } from "./voicePresence";

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

  /** Opens the session's channel and starts following it. */
  static async open(session: Session): Promise<AgentChat> {
    const { channel } = await session.chat();
    const chat = new AgentChat(session, channel as Channel);
    await chat.follow();
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
    return lastReply?.generating === true;
  }

  /** Asks, with the photos picked for it and, when `thinkHarder`, more time to reason. */
  async ask(text: string, photos: readonly File[] = [], thinkHarder = false): Promise<void> {
    const messages = this.get().messages;
    this.update({ failure: undefined, askedAfter: messages[messages.length - 1]?.id ?? "" });
    try {
      const images = await photoSources(photos);
      await this.session.responses.create(text, {
        ...(images.length > 0 ? { images } : {}),
        ...(thinkHarder ? { reasoningEffort: "high" } : {}),
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

  /** Loads the messages before the oldest one loaded. */
  async loadOlderMessages(): Promise<void> {
    const oldest = this.get().messages[0]?.id;
    if (!oldest) return;
    try {
      const page = await this.channel.query({ messages: { limit: PAGE_SIZE, id_lt: oldest } });
      this.update({ messages: [...this.channel.state.messages], hasOlderMessages: page.messages.length >= PAGE_SIZE });
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
    // Every event rather than the list kept for us: each live update carries its own window of
    // the reply's thinking, and folding them into one would lose all but the last.
    const subscription = this.channel.on((event: Event) => {
      if (event.type === "message.updated" && event.message) {
        this.reasoning.read(event.message as { id: string; reasoning?: unknown });
        this.update({ reasoningVersion: this.get().reasoningVersion + 1 });
      }
      if (event.type === "message.new" || event.type === "message.updated" || event.type === "message.deleted") {
        this.update({ messages: [...this.channel.state.messages] });
      }
    });
    this.stops.push(() => subscription.unsubscribe());
    // What the agent is doing changes the activity and whether it is answering.
    this.stops.push(this.session.state.subscribe(() => this.update({})));

    const watched = await this.channel.watch({ messages: { limit: PAGE_SIZE } });
    this.update({
      messages: [...this.channel.state.messages],
      hasOlderMessages: (watched.messages?.length ?? 0) >= PAGE_SIZE,
    });
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
    return !isAgentReply(last) || (last.generating === true && !this.replyHasBegun());
  }
}

/** A reply the agent wrote, which Stream's AI components render. */
export function isAgentReply(message: LocalMessage): boolean {
  return message.ai_generated === true;
}

/** A reply with nothing to show yet: no text and no steps. */
function isEmptyReply(message: LocalMessage): boolean {
  return !message.text && !(message.attachments ?? []).some((each) => each.type === "ai_reasoning" || each.type === "ai_tool_call");
}

declare module "stream-chat" {
  interface CustomMessageData {
    ai_generated?: boolean;
    generating?: boolean;
  }
}
