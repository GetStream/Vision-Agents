import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { StateStore } from "stream-chat";
import {
  AIClientIdentity,
  AIConversation,
  AIMessageComposer,
  AISuggestions,
  AITypingIndicator,
  createPopupAuthorizer,
  fromStreamMessage,
  useAIMessageComposerContext,
  useAttachments,
  type AIChatOption,
  type AIMessageComposerStore,
  type AIToolApprover,
} from "stream-chat-react/ai-components";

import type { AgentChat, AgentChatSnapshot } from "./agentChat";
import { Assistant } from "./assistant";
import type { ChatScreen } from "./chatScreen";
import { Store, useStore } from "./store";

const STARTERS = [
  "What's on my calendar today?",
  "Remind me to call mum at 6pm",
  "Plan a weekend in Lisbon",
  "What should I cook tonight?",
];

const THINK_HARDER: AIChatOption = {
  id: "think_harder",
  title: "Think harder",
  description: "More time to reason before answering",
  shortTitle: "Think",
};

/** Followed before a conversation has a chat, so the hooks below always have a store. */
const NO_CHAT = new Store<AgentChatSnapshot>({
  messages: [],
  reasoningVersion: 0,
  hasOlderMessages: false,
  ratings: {},
  failure: undefined,
  askedAfter: undefined,
});

/**
 * One conversation: the messages, the composer floating over the bottom of them, and the way into
 * voice mode.
 *
 * The person's messages are in bubbles, the agent's as their steps and then their answer. What the
 * person asks shows at once, and Stream's copy takes its place. Each question goes to the top as
 * it is asked, and its reply writes itself out below it without moving what is on screen. While
 * the agent thinks, the list shows its thinking, which the reply's first step takes over.
 */
export function ChatView({ screen }: { screen: ChatScreen }) {
  const { chat, pending, failure: openFailure } = useStore(screen);
  const state = useStore<AgentChatSnapshot>(chat ?? NO_CHAT);
  // What the agent is doing comes from the session, which the chat's store follows.
  const activity = chat?.activity();
  // A question on its way counts too: a new conversation's session can still be opening.
  const answering = screen.isBusy();
  const approver = useApprover(chat);
  const [barHeight, setBarHeight] = useState(0);
  const bar = useRef<HTMLDivElement>(null);
  const [thinkHarder, setThinkHarder] = useState(false);
  const composer = useRef<StateStore<AIMessageComposerStore> | null>(null);

  useEffect(() => {
    const element = bar.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setBarHeight(element.offsetHeight));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const messages = useMemo(
    () =>
      state.messages.map((message) =>
        fromStreamMessage(message, { reasoningText: (id, step) => chat?.reasoning.get(id, step) }),
      ),
    // The live thinking is read through `chat.reasoning`; its version says when it grew.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [state.messages, state.reasoningVersion, chat],
  );

  const isWelcoming = state.messages.length === 0 && !pending;
  const failure = openFailure ?? state.failure;

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // The draft waits, as the send button does, until the reply is done or stopped.
    if (screen.isBusy()) return;
    const draft = composer.current?.getLatestValue();
    const text = draft?.text.trim() ?? "";
    const photos = draft?.attachments.map((each) => each.file) ?? [];
    if (!text && photos.length === 0) return;
    void screen.send(text, photos, thinkHarder);
    event.currentTarget.reset();
  };

  return (
    <div className="chat">
      <AIConversation
        approver={approver}
        bottomInset={barHeight}
        emptyContent={<Welcome />}
        footer={activity?.kind === "working" ? <AITypingIndicator text={activity.text} /> : undefined}
        isThinking={activity?.kind === "thinking"}
        loadOlderMessages={state.hasOlderMessages && chat ? () => chat.loadOlderMessages() : undefined}
        messages={messages}
        pendingMessage={pending ?? null}
        replyActions={(message) =>
          chat ? { rating: state.ratings[message.id] ?? null, onRate: (rating) => chat.rate(message.id, rating) } : undefined
        }
      />
      {/* The conversation fades away under the composer, rather than stopping at a hairline. */}
      <div className="composer-bar" ref={bar}>
        {isWelcoming && <AISuggestions onSelect={(text) => void screen.send(text)} suggestions={STARTERS} />}
        {failure && <p className="failure-pill">{failure}</p>}
        <AIMessageComposer
          onChange={(event) => {
            const input = event.target as unknown as HTMLInputElement;
            if (input.name === "message") screen.draftChanged(input.value);
          }}
          onSubmit={submit}
        >
          <ComposerStore into={composer} />
          <PickedPhotos />
          <div className="composer-row">
            <AIMessageComposer.FileInput accept="image/*" multiple name="attachments" />
            <AIMessageComposer.ChatOptions
              icon={<OptionsIcon />}
              onSelectedChange={(option) => setThinkHarder(option?.id === THINK_HARDER.id)}
              options={[THINK_HARDER]}
              selectedId={thinkHarder ? THINK_HARDER.id : null}
            />
            <AIMessageComposer.TextInput name="message" placeholder="Ask Juno" />
            {answering ? (
              <button aria-label="Stop" className="round-button" onClick={() => chat?.stop()} type="button">
                ■
              </button>
            ) : (
              <AIMessageComposer.SubmitButton />
            )}
            <button aria-label="Talk to Juno" className="round-button talk" onClick={() => void screen.talk()} type="button">
              <WaveformIcon />
            </button>
          </div>
        </AIMessageComposer>
      </div>
    </div>
  );
}

/** The photos picked for the next question, each with a way to take it back. */
function PickedPhotos() {
  const { attachments, removeAttachment } = useAttachments();
  if (attachments.length === 0) return null;
  return (
    <AIMessageComposer.AttachmentPreview>
      {attachments.map((each) => (
        <AIMessageComposer.AttachmentPreview.Item
          file={each.file}
          key={each.id}
          onDelete={() => removeAttachment(each.id)}
        />
      ))}
    </AIMessageComposer.AttachmentPreview>
  );
}

/** Hands the composer's draft out, so sending reads its text and photos. */
function ComposerStore({ into }: { into: { current: StateStore<AIMessageComposerStore> | null } }) {
  into.current = useAIMessageComposerContext();
  return null;
}

/**
 * Answers the questions of the tools that ask first, as the person signed in here, and signs them
 * in when a connector's call waits for their account.
 *
 * The router addresses a question to the person whose message the call answers, and Stream's
 * components show it only to them. The answer goes back over the session's socket, which is also
 * what runs the tool once it is allowed. Connect opens the sign-in page in a popup, which the
 * router sends back to this page's origin once the account is connected, and the call carries on
 * at the router. Not now ends the call there.
 */
function useApprover(chat: AgentChat | undefined): AIToolApprover | undefined {
  return useMemo(() => {
    if (!chat) return undefined;
    return {
      userId: Assistant.user.id,
      clientId: AIClientIdentity.installId(),
      authorizer: createPopupAuthorizer({
        returnUrl: Assistant.returnUrl,
        decline: async (call) => chat.session.declineLogin(call.id),
      }),
      decide: async (call, allowed) => {
        // A reminder alerts with a notification, which a browser lets a page ask for only in a
        // click: this is that click.
        if (allowed && call.name === "add_reminder" && "Notification" in window && Notification.permission === "default") {
          void Notification.requestPermission();
        }
        chat.session.decide(call.id, { allowed });
      },
    };
  }, [chat]);
}

/** What a new conversation shows before anything is said. */
function Welcome() {
  const hour = new Date().getHours();
  const part = hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening";
  return (
    <div className="welcome">
      <div className="welcome-icon" aria-hidden>
        ✦
      </div>
      <h2>
        Good {part}, {Assistant.user.name}
      </h2>
      <p>Ask me anything, show me a photo, or tap the waveform to talk.</p>
    </div>
  );
}

/** The way into the composer's options, such as Think harder. */
function OptionsIcon() {
  return (
    <svg aria-hidden height="20" viewBox="0 0 24 24" width="20">
      <path
        d="M4 7h10M18 7h2M4 17h4M12 17h8M16 5v4M10 15v4"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
      />
    </svg>
  );
}

export function WaveformIcon() {
  return (
    <svg aria-hidden height="18" viewBox="0 0 24 24" width="18">
      <path
        d="M4 10v4M8 6v12M12 3v18M16 7v10M20 10v4"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="2"
      />
    </svg>
  );
}
