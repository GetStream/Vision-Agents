import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { LocalMessage } from "stream-chat";
import type { AgentChat, AgentChatState } from "@stream-io/stream-agents/chat";
import {
  AIClientIdentity,
  AIConversation,
  AIMessageComposer,
  AISuggestions,
  AITypingIndicator,
  createPopupAuthorizer,
  isAIConversationAnswering,
  useAIConversationState,
  useAttachments,
  type AIChatOption,
  type AIConversationControllerState,
  type AIToolApprover,
  AIUIHost,
  aiUIActionJSON,
} from "stream-chat-react/ai-components";
import { createA2UIRenderer } from "stream-chat-react/a2ui";
import { createOpenUIRenderer } from "stream-chat-react/openui";

import { Assistant } from "./assistant";
import { THINK_HARDER_ID, type ChatScreen } from "./chatScreen";
import { messageOf } from "./errors";
import { useStore } from "./store";
import { tripComponents } from "./tripComponents";

const STARTERS = [
  "What's on my calendar today?",
  "Remind me to call mum at 6pm",
  "Plan a weekend in Lisbon",
  "What should I cook tonight?",
];

const THINK_HARDER: AIChatOption = {
  id: THINK_HARDER_ID,
  title: "Think harder",
  description: "More time to reason before answering",
  shortTitle: "Think",
};

const conversationSelector = ({ activity, error, messages, pendingMessage, status }: AIConversationControllerState) => ({
  activity,
  error,
  isEmpty: messages.length === 0 && !pendingMessage,
  status,
});

/** What one part of the chat's state is now, kept current. `select` returns a value that stays the same while unchanged. */
function useChatState<S>(chat: AgentChat<LocalMessage>, select: (state: AgentChatState<LocalMessage>) => S): S {
  const subscribe = useCallback((changed: () => void) => chat.state.subscribe(() => changed()), [chat]);
  return useSyncExternalStore(subscribe, () => select(chat.state.getLatestValue()));
}

/**
 * One conversation: the messages, the composer floating over the bottom of them, and the way into
 * voice mode.
 *
 * Everything about the conversation is the controller's (`screen.conversation`): the question on
 * its way, whether the agent thinks or writes, an error with a way to try again, the reply
 * actions and the Send button turning into Stop. The SDK's `AgentChat` is its transport.
 */
export function ChatView({ screen }: { screen: ChatScreen }) {
  const { conversation, chat } = screen;
  const { failure: openFailure } = useStore(screen);
  const { activity, error, isEmpty, status } = useAIConversationState(conversation, conversationSelector);
  const isLoaded = useChatState(chat, (state) => state.isLoaded);
  const readFailure = useChatState(chat, (state) => state.failure);
  const [barHeight, setBarHeight] = useState(0);
  const bar = useRef<HTMLDivElement>(null);
  const { ui, unsent } = useGeneratedUI(screen, status);

  useEffect(() => {
    const element = bar.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setBarHeight(element.offsetHeight));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const isWelcoming = isEmpty && isLoaded;
  const failure = openFailure ?? (readFailure ? messageOf(readFailure) : undefined) ?? unsent;
  // Skills running, said in words; thinking is shown by the conversation itself.
  const working = status !== "ready" && activity?.state === "AI_STATE_EXTERNAL_SOURCES" ? activity.text : undefined;

  return (
    <div className="chat">
      <AIConversation
        bottomInset={barHeight}
        conversation={conversation}
        emptyContent={isLoaded ? <Welcome /> : <LoadingConversation />}
        footer={
          error ? (
            <p className="failure-pill">
              {error.name === "AIAgentError" && !("agentMessage" in error && error.agentMessage) ? "Juno couldn't answer." : messageOf(error)}{" "}
              <button className="link-button" onClick={() => conversation.retry()} type="button">
                Try again
              </button>
            </p>
          ) : working ? (
            <AITypingIndicator text={working} />
          ) : undefined
        }
        ui={ui}
      />
      {/* The conversation fades away under the composer, rather than stopping at a hairline. */}
      <div className="composer-bar" ref={bar}>
        {isWelcoming && <AISuggestions onSelect={(text) => conversation.send({ text })} suggestions={STARTERS} />}
        {failure && <p className="failure-pill">{failure}</p>}
        <AIMessageComposer
          conversation={conversation}
          onChange={(event) => {
            const input = event.target as unknown as HTMLInputElement;
            if (input.name === "message") screen.draftChanged(input.value);
          }}
        >
          <PickedPhotos />
          <div className="composer-row">
            <AIMessageComposer.FileInput accept="image/*" multiple name="attachments" />
            <AIMessageComposer.ChatOptions icon={<OptionsIcon />} options={[THINK_HARDER]} />
            <AIMessageComposer.TextInput name="message" placeholder="Ask Juno" />
            {/* Turns into the Stop button while Juno answers. */}
            <AIMessageComposer.SubmitButton />
            <button aria-label="Talk to Juno" className="round-button talk" onClick={() => void screen.talk()} type="button">
              <WaveformIcon />
            </button>
          </div>
        </AIMessageComposer>
      </div>
    </div>
  );
}

/**
 * The UI the agent writes for a reply, in place of its text: A2UI or OpenUI, with the app's own
 * trip components beside each format's own. What the person does in it is their next question,
 * in their words, carrying the action, so it shows at once like anything they ask. One the screen
 * can't send yet, while the agent still answers, fails, and says why where a failed question
 * does, rather than doing nothing.
 */
function useGeneratedUI(screen: ChatScreen, status: string): { ui: AIUIHost; unsent: string | undefined } {
  const [unsent, setUnsent] = useState<string>();
  const ui = useMemo(
    () =>
      new AIUIHost({
        components: tripComponents,
        renderers: [createA2UIRenderer(), createOpenUIRenderer()],
        onError: (error) => setUnsent(messageOf(error)),
        send: async (action) => {
          setUnsent(undefined);
          const { conversation } = screen;
          if (isAIConversationAnswering(conversation.state.getLatestValue().status)) {
            throw new Error(`"${action.label}" wasn't sent: Juno is still answering. Try again once it's done.`);
          }
          // What the person did travels as `custom.uiAction`, which the SDK's transport hands to the router.
          conversation.send({ text: action.label, custom: { uiAction: aiUIActionJSON(action) } });
        },
      }),
    [screen],
  );
  // It no longer applies once the agent has answered.
  const answering = isAIConversationAnswering(status as never);
  useEffect(() => {
    if (!answering) setUnsent(undefined);
  }, [answering]);
  return { ui, unsent };
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
export function approverFor(chat: AgentChat<LocalMessage>): AIToolApprover {
  return {
    userId: Assistant.user.id,
    clientId: AIClientIdentity.installId(),
    authorizer: createPopupAuthorizer({
      returnUrl: Assistant.returnUrl,
      decline: async (call) => (await chat.session()).declineLogin(call.id),
    }),
    decide: async (call, allowed) => {
      // A reminder alerts with a notification, which a browser lets a page ask for only in a
      // click: this is that click.
      if (allowed && call.name === "add_reminder" && "Notification" in window && Notification.permission === "default") {
        void Notification.requestPermission();
      }
      (await chat.session()).decide(call.id, { allowed });
    },
  };
}

/** What a new conversation shows before anything is said. */
/** Where a conversation's messages will be, while it opens: the shape of a question and its answer. */
function LoadingConversation() {
  return (
    <div aria-label="Loading the conversation" className="loading-conversation" role="status">
      <div className="skeleton skeleton-question" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-line short" />
      <div className="skeleton skeleton-question narrow" />
      <div className="skeleton skeleton-line" />
      <div className="skeleton skeleton-line short" />
    </div>
  );
}

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
