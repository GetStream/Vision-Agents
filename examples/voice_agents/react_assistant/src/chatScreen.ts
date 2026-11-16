import type { Session } from "@stream-io/stream-agents";
import { AgentChat, agentChatTransport } from "@stream-io/stream-agents/chat";
import type { VoiceSession } from "@stream-io/stream-agents/video";
import type { LocalMessage } from "stream-chat";
import {
  AIConversationController,
  fromStreamMessage,
  type AIToolApprover,
} from "stream-chat-react/ai-components";

import { messageOf } from "./errors";
import { photoSources } from "./photos";
import { Store } from "./store";
import { workingLabel } from "./voicePresence";

/** How many messages are read at a time. */
export const PAGE_SIZE = 30;

/** The composer option that asks with a higher reasoning effort. */
export const THINK_HARDER_ID = "think_harder";

/** What a screen does with the router, handed in so the screen itself holds no client. */
export interface ChatScreenDeps {
  /** Carries on the conversation `sessionId` names, or starts one for undefined. */
  open(sessionId: string | undefined): Promise<Session>;
  /**
   * A listed conversation's channel, read without its session so its messages show while the
   * session comes back. The channel, as `type:id`, is passed when the list already said it.
   */
  channel(sessionId: string, conversationId: string | undefined): Promise<unknown>;
  /** Answers the tools that ask first, and signs in a connector's call, for the chat's session. */
  approver(chat: AgentChat<LocalMessage>): AIToolApprover;
  /** The call of a session whose voice is on. */
  voiceOf(session: Session): VoiceSession;
}

export interface ChatScreenSnapshot {
  /** Why opening the conversation failed, or undefined. */
  readonly failure: string | undefined;
  /** Whether voice mode is up, over the chat. */
  readonly isVoiceShown: boolean;
  /** The call in voice mode, once it has started. */
  readonly voice: VoiceSession | undefined;
  /** Why voice mode could not start, or why the camera could not, or undefined. */
  readonly voiceFailure: string | undefined;
  /** When voice mode opened, in milliseconds since the epoch: what the agent failed at before is the chat's. */
  readonly voiceOpenedAt: number;
  /** The conversation's session, once it is open. */
  readonly session: Session | undefined;
}

/**
 * One conversation on screen: its `AgentChat` (the SDK's, which reads the router's channel and
 * session) and the `AIConversationController` the AI components show, with the chat as its
 * transport. The controller keeps the question on its way, whether the agent answers, errors and
 * retrying; this keeps what is the app's own: voice mode, and telling the list about a new
 * conversation once it is used.
 *
 * A new conversation has no session until it is asked something or talked to, and the chat opens
 * it while the first question is written, since that takes seconds. One that was only looked at is
 * deleted when the screen is left.
 */
export class ChatScreen extends Store<ChatScreenSnapshot> {
  readonly chat: AgentChat<LocalMessage>;
  readonly conversation: AIConversationController;
  private readonly stops: (() => void)[] = [];
  private stopVoice: (() => void) | undefined;
  private left = false;

  /**
   * @param sessionId The conversation to carry on, or undefined for a new one.
   * @param onStarted Says which session a new conversation became, once it is first used.
   * @param conversationId The carried-on conversation's channel, as `type:id`, when the list
   * already said it, which saves looking it up before reading its messages.
   */
  constructor(
    private readonly sessionId: string | undefined,
    private readonly deps: ChatScreenDeps,
    onStarted: (sessionId: string) => void,
    conversationId?: string,
  ) {
    super({
      failure: undefined,
      isVoiceShown: false,
      voice: undefined,
      voiceFailure: undefined,
      voiceOpenedAt: 0,
      session: undefined,
    });
    this.chat = new AgentChat<LocalMessage>({ sessionId, open: deps.open, pageSize: PAGE_SIZE });
    this.conversation = new AIConversationController({
      transport: agentChatTransport(this.chat, {
        adapt: (message, reasoningText) => fromStreamMessage(message, { reasoningText }),
        images: (attachments) => photoSources(attachments as File[]),
        reasoningEffort: (option) => (option?.id === THINK_HARDER_ID ? "high" : undefined),
        workingText: workingLabel,
      }),
      approver: deps.approver(this.chat),
      // The router keeps the conversation in Stream Chat and can't rewind it.
      actions: { edit: false, regenerate: false },
    });
    let announced = sessionId !== undefined;
    this.stops.push(
      this.chat.state.subscribe(({ session, usedSessionId }) => {
        if (session !== this.get().session) this.update({ session });
        // A new conversation joins the list once it is asked something or talked to.
        if (!announced && usedSessionId) {
          announced = true;
          onStarted(usedSessionId);
        }
      }),
    );
    if (sessionId) {
      void this.preview(sessionId, conversationId);
      this.chat.prepare().catch((cause: unknown) => this.update({ failure: messageOf(cause) }));
    }
  }

  /** Follows what the person is writing, to open a new conversation's session meanwhile. */
  draftChanged(text: string): void {
    if (this.sessionId || !text) return;
    this.chat.prepare().catch((cause: unknown) => this.update({ failure: messageOf(cause) }));
  }

  /**
   * Switches to voice mode on this conversation, starting a session for a new one.
   *
   * Voice is started on the chat's own session, so the agent joins its call with everything said
   * so far, and what is said on the call is written into the same channel. The session's one
   * socket stays open the whole time and runs this page's tools, on the call too. The agent
   * leaving ends voice mode.
   */
  async talk(): Promise<void> {
    if (this.get().isVoiceShown) return;
    this.update({ isVoiceShown: true, voiceFailure: undefined, voiceOpenedAt: Date.now() });
    try {
      // Talked to is used, as asked is: the conversation joins the list and is kept.
      const session = await this.chat.use();
      await session.voice.start();
      const voice = this.deps.voiceOf(session);
      this.update({ voice });
      const ended = session.state.subscribe(() => {
        if (session.state.get().kind === "ended") {
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
    const { voice, session, isVoiceShown } = this.get();
    this.stopVoice?.();
    this.stopVoice = undefined;
    this.update({ isVoiceShown: false, voice: undefined, voiceFailure: undefined });
    if (!isVoiceShown) return;
    void (async () => {
      await voice?.leave().catch(() => undefined);
      // The agent had left already, if this fails, which is what stopping voice was for.
      await session?.voice.stop().catch(() => undefined);
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
   * Closes the conversation on screen. The chat deletes a new one's session if it was never used.
   * Run when another conversation is picked or the page goes.
   */
  leave(): void {
    if (this.left) return;
    this.hangUp();
    this.left = true;
    for (const stop of this.stops.splice(0)) stop();
    this.conversation.dispose();
    void this.chat.close();
  }

  /**
   * Reads a carried-on conversation's messages from Stream, which takes a fraction of the time its
   * session does to come back. Nothing is lost if it fails: the chat reads them once open.
   */
  private async preview(sessionId: string, conversationId: string | undefined): Promise<void> {
    const channel = await this.deps.channel(sessionId, conversationId).catch(() => undefined);
    if (!channel || this.left) return;
    await this.chat.follow(channel).catch(() => undefined);
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
