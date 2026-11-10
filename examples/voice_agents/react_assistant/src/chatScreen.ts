import type { Session, UIAction } from "@stream-io/stream-agents";
import type { VoiceSession } from "@stream-io/stream-agents/video";
import type { LocalMessage } from "stream-chat";
import type { AIPendingMessage } from "stream-chat-react/ai-components";

import { isAgentReply, type AgentChat } from "./agentChat";
import { messageOf } from "./errors";
import { Store } from "./store";

/** What a screen does with the router, handed in so the screen itself holds no client. */
export interface ChatScreenDeps {
  /** Carries on the conversation `sessionId` names, or starts one for undefined, with its chat open. */
  open(sessionId: string | undefined): Promise<AgentChat>;
  /**
   * A conversation's latest messages, read from its channel without its session. The channel,
   * as `type:id`, is passed when the list already said it, and looked up otherwise.
   */
  history(sessionId: string, conversationId: string | undefined): Promise<LocalMessage[]>;
  /** Deletes a conversation's session. */
  remove(sessionId: string): Promise<void>;
  /** The call of a session whose voice is on. */
  voiceOf(session: Session): VoiceSession;
}

interface Asked {
  readonly message: AIPendingMessage;
  /** How many of the person's messages the conversation held when it was asked: the next one is it. */
  readonly after: number;
  /** The photos' object URLs, let go of once Stream's copy has taken over. */
  readonly urls: readonly string[];
}

export interface ChatScreenSnapshot {
  /** The conversation's chat, once its session is open. */
  readonly chat: AgentChat | undefined;
  /** What the person just asked, shown at once until Stream's copy of their message arrives. */
  readonly pending: AIPendingMessage | undefined;
  /** Why opening the conversation failed, or undefined. */
  readonly failure: string | undefined;
  /**
   * A conversation's messages from the list, read from Stream while its session comes back on the
   * router, which takes seconds longer. Shown until the chat is open, which then has them too.
   */
  readonly history: readonly LocalMessage[] | undefined;
  /**
   * Whether a conversation from the list has nothing to show yet: neither its messages nor its
   * chat. The screen says so meanwhile rather than look like a new conversation.
   */
  readonly isLoading: boolean;
  /** Whether voice mode is up, over the chat. */
  readonly isVoiceShown: boolean;
  /** The call in voice mode, once it has started. */
  readonly voice: VoiceSession | undefined;
  /** Why voice mode could not start, or why the camera could not, or undefined. */
  readonly voiceFailure: string | undefined;
  /** When voice mode opened, in milliseconds since the epoch: what the agent failed at before is the chat's. */
  readonly voiceOpenedAt: number;
}

/**
 * One conversation on screen, and what happens to it as the person writes, asks and talks.
 *
 * A new conversation has no session until its first message, so opening the page or picking New
 * chat leaves nothing in the list. Opening a session takes seconds, though, so a new
 * conversation's is opened as its first message is written rather than once it is sent, and
 * deleted again if the person leaves without sending it. A conversation from the list is followed
 * again by its id, and one that ended carries on when it is next asked something.
 */
export class ChatScreen extends Store<ChatScreenSnapshot> {
  private opening: Promise<AgentChat | undefined> | undefined;
  private asked: Asked | undefined;
  /** Whether the list has been told about this new conversation. */
  private isAnnounced = false;
  /** A new conversation's session, opened while the person wrote and not used yet. */
  private unused: string | undefined;
  private left = false;
  private stopFollowing: (() => void) | undefined;
  private stopVoice: (() => void) | undefined;

  /**
   * @param sessionId The conversation to carry on, or undefined for a new one.
   * @param onStarted Says which session a new conversation became, once it is first used.
   * @param conversationId The carried-on conversation's channel, as `type:id`, when the list
   * already said it, which saves looking it up before reading its messages.
   */
  constructor(
    private readonly sessionId: string | undefined,
    private readonly deps: ChatScreenDeps,
    private readonly onStarted: (sessionId: string) => void,
    conversationId?: string,
  ) {
    super({
      chat: undefined,
      pending: undefined,
      failure: undefined,
      history: undefined,
      isLoading: sessionId !== undefined,
      isVoiceShown: false,
      voice: undefined,
      voiceFailure: undefined,
      voiceOpenedAt: 0,
    });
    if (sessionId) {
      void this.open();
      void this.preview(sessionId, conversationId);
    }
  }

  /** Follows what the person is writing, to open a new conversation's session meanwhile. */
  draftChanged(text: string): void {
    if (this.sessionId || this.get().chat || this.opening || !text) return;
    void this.open().then((chat) => {
      if (chat && !this.isAnnounced) this.unused = chat.session.id;
    });
  }

  /**
   * Whether a question is on its way or being answered. The router answers one at a time, so
   * nothing more is sent until it is done or stopped: a new conversation's session can still be
   * opening, with nothing for the session to say about it yet.
   */
  isBusy(): boolean {
    return this.asked !== undefined || (this.get().chat?.isAnswering() ?? false);
  }

  /**
   * Asks, with the photos the composer holds, which show on the question at once. It sends nothing
   * while the screen is busy, and says whether it sent.
   */
  async send(text: string, photos: readonly File[] = [], thinkHarder = false, uiAction?: UIAction): Promise<boolean> {
    if (this.isBusy()) return false;
    const urls = photos.filter((photo) => photo.type.startsWith("image/")).map((photo) => URL.createObjectURL(photo));
    const asking: Asked = {
      message: {
        id: crypto.randomUUID(),
        text,
        attachments: urls.map((url) => ({ kind: "image" as const, url })),
      },
      after: this.userMessages(),
      urls,
    };
    this.setAsked(asking);
    const chat = await this.open();
    if (!chat) {
      this.setAsked(undefined);
      return true;
    }
    this.markUsed(chat);
    await chat.ask(text, photos, thinkHarder, uiAction);
    if (chat.get().failure && this.asked === asking) {
      this.setAsked(undefined);
    }
    return true;
  }

  /**
   * Switches to voice mode on this conversation, starting a session for a new one.
   *
   * Voice is started on the chat's own session, so the agent joins its call with everything said
   * so far, and what is said on the call is written into the same channel. The session's one
   * socket stays open the whole time and runs this page's tools, on the call too. The microphone
   * is asked for as the page joins. The agent leaving ends voice mode.
   */
  async talk(): Promise<void> {
    const chat = await this.open();
    if (!chat || this.get().isVoiceShown) return;
    this.markUsed(chat);
    this.update({ isVoiceShown: true, voiceFailure: undefined, voiceOpenedAt: Date.now() });
    try {
      await chat.session.voice.start();
      const voice = this.deps.voiceOf(chat.session);
      this.update({ voice });
      const ended = chat.session.state.subscribe(() => {
        if (chat.session.state.get().kind === "ended") {
          ended();
          this.hangUp();
        }
      });
      this.stopVoice = ended;
      await voice.join();
    } catch (cause) {
      this.update({ voiceFailure: isDenied(cause) ? "Talking to Juno needs the microphone." : messageOf(cause) });
    }
  }

  /**
   * Leaves the call and takes the agent off it, which carries the conversation on in writing,
   * and goes back to the chat. Leaving a call the chat's session is on keeps the session.
   */
  hangUp(): void {
    const { voice, chat, isVoiceShown } = this.get();
    this.stopVoice?.();
    this.stopVoice = undefined;
    this.update({ isVoiceShown: false, voice: undefined, voiceFailure: undefined });
    if (!isVoiceShown) return;
    void (async () => {
      await voice?.leave().catch(() => undefined);
      // The agent had left already, if this fails, which is what stopping voice was for.
      await chat?.session.voice.stop().catch(() => undefined);
    })();
  }

  async setMuted(muted: boolean): Promise<void> {
    await this.get().voice?.setMuted(muted).catch(() => undefined);
  }

  /** Shows Juno the camera or stops, which asks for the camera the first time. */
  async setCameraEnabled(enabled: boolean): Promise<void> {
    const voice = this.get().voice;
    if (!voice) return;
    this.update({ voiceFailure: undefined });
    try {
      await voice.setCameraEnabled(enabled);
    } catch (cause) {
      this.update({ voiceFailure: isDenied(cause) ? "Showing Juno the camera needs the camera." : messageOf(cause) });
    }
  }

  /**
   * Closes the conversation on screen, and deletes a new one's session if it was never used. Run
   * when another conversation is picked or the page goes.
   */
  leave(): void {
    if (this.left) return;
    this.hangUp();
    this.left = true;
    this.stopFollowing?.();
    const closing = this.get().chat;
    const abandoned = this.unused;
    void (async () => {
      await closing?.close().catch(() => undefined);
      if (abandoned) {
        // Left in the list, untitled, if this fails, which the person can delete there.
        await this.deps.remove(abandoned).catch(() => undefined);
      }
    })();
  }

  private userMessages(): number {
    const { chat, history } = this.get();
    return (chat?.get().messages ?? history ?? []).filter((each) => !isAgentReply(each)).length;
  }

  private setAsked(asked: Asked | undefined): void {
    for (const url of this.asked && this.asked !== asked ? this.asked.urls : []) {
      // Kept a moment longer, for the row Stream's copy takes over from to finish drawing.
      setTimeout(() => URL.revokeObjectURL(url), 5_000);
    }
    this.asked = asked;
    this.update({ pending: asked?.message });
  }

  /** Tells the list about a new conversation once it is used, written or talked to. */
  private markUsed(chat: AgentChat): void {
    this.unused = undefined;
    if (this.sessionId || this.isAnnounced) return;
    this.isAnnounced = true;
    this.onStarted(chat.session.id);
  }

  /** The conversation's chat, starting a session for a new one. */
  private open(): Promise<AgentChat | undefined> {
    const open = this.get().chat;
    if (open) return Promise.resolve(open);
    this.opening ??= this.start().finally(() => {
      this.opening = undefined;
    });
    return this.opening;
  }

  /**
   * Reads a carried-on conversation's messages from Stream, which takes a fraction of the time its
   * session does to come back. Nothing is lost if it fails: the chat reads them again once open.
   */
  private async preview(sessionId: string, conversationId: string | undefined): Promise<void> {
    const history = await this.deps.history(sessionId, conversationId).catch(() => undefined);
    // The session opened first, or the person moved on: there is nothing to show this for.
    if (!history || this.left || this.get().chat) return;
    this.update({ history, isLoading: false });
  }

  /** Carries on the conversation's session, or starts one for a new conversation, and opens its channel. */
  private async start(): Promise<AgentChat | undefined> {
    this.update({ failure: undefined });
    try {
      const chat = await this.deps.open(this.sessionId);
      if (this.left) {
        // The person moved on while it opened.
        await chat.close();
        if (!this.sessionId) await this.deps.remove(chat.session.id).catch(() => undefined);
        return undefined;
      }
      // Stream's copy of the question on its way is the person's first message after the ones
      // there were when they asked: once it is in, the question on its way goes.
      this.stopFollowing = chat.subscribe(() => {
        if (this.asked && this.userMessages() > this.asked.after) {
          this.setAsked(undefined);
        }
      });
      this.update({ chat, history: undefined, isLoading: false });
      return chat;
    } catch (cause) {
      this.update({ failure: messageOf(cause), isLoading: false });
      return undefined;
    }
  }
}

/**
 * Whether the browser refused a device the person did not allow: the browser's own error, or
 * Stream Video's, which says so in words.
 */
function isDenied(cause: unknown): boolean {
  if (cause instanceof DOMException && (cause.name === "NotAllowedError" || cause.name === "SecurityError")) {
    return true;
  }
  return cause instanceof Error && /permission|not allowed|denied/i.test(cause.message);
}
