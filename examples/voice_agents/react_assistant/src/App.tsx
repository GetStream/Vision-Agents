import { useEffect, useMemo, useState } from "react";
import { Chat } from "stream-chat-react";

import type { Tools } from "@stream-io/calypso";
import { VoiceSession } from "@stream-io/calypso/video";

import { AgentChat } from "./agentChat";
import { agents, Assistant, chatClient, sessions, signIn } from "./assistant";
import { ChatScreen, type ChatScreenDeps } from "./chatScreen";
import { ChatView } from "./ChatView";
import { ConversationList } from "./ConversationList";
import { Conversations, displayTitle } from "./conversations";
import { messageOf } from "./errors";
import { deviceTools } from "./deviceTools";
import { scheduleReminders } from "./reminders";
import { useStore } from "./store";
import { VoiceMode } from "./VoiceMode";

/**
 * The conversation in front, and the ones there were in a drawer that slides over it.
 *
 * There is no agent picker. Reading the configs is server-side only, so a page cannot ask which
 * agents exist; it is told which one it talks to, in `Assistant.agentName`.
 */
export function App() {
  const [ready, setReady] = useState<"signing-in" | "ready" | string>("signing-in");

  useEffect(() => {
    if (!Assistant.streamApiKey) return;
    scheduleReminders();
    signIn().then(
      () => setReady("ready"),
      (cause: unknown) => setReady(messageOf(cause)),
    );
  }, []);

  if (!Assistant.streamApiKey) {
    return (
      <Notice title="No Stream key yet">
        Put the STREAM_API_KEY the router runs with in <code>.env.local</code>, as <code>VITE_STREAM_API_KEY</code>.
      </Notice>
    );
  }
  if (ready === "signing-in") {
    return <Notice title="Connecting">Signing in as {Assistant.user.name}…</Notice>;
  }
  if (ready !== "ready") {
    return <Notice title="Juno could not sign in">{ready}</Notice>;
  }
  return (
    <Chat client={chatClient}>
      <Assistant_ />
    </Chat>
  );
}

function Assistant_() {
  const conversations = useMemo(() => new Conversations(), []);
  const deps = useMemo(() => screenDeps(deviceTools()), []);
  const [isMenuOpen, setMenuOpen] = useState(false);
  /** The conversation open, or undefined for a new one. */
  const [selected, setSelected] = useState<string | undefined>(undefined);
  // Which conversation the screen was opened for. It changes only when the person picks one, so
  // a new conversation getting its session keeps the screen it was started on.
  const [screen, setScreen] = useState(() => new ChatScreen(undefined, deps, setSelected));
  const { items } = useStore(conversations);
  const { chat, isVoiceShown } = useStore(screen);

  useEffect(() => {
    const leave = () => screen.leave();
    window.addEventListener("pagehide", leave);
    return () => window.removeEventListener("pagehide", leave);
  }, [screen]);

  useEffect(() => {
    // The router titles a conversation as it goes, so the list is read again each time it is shown.
    if (isMenuOpen) void conversations.reload();
  }, [conversations, isMenuOpen]);

  const show = (sessionId: string | undefined) => {
    screen.leave();
    setSelected(sessionId);
    setScreen(new ChatScreen(sessionId, deps, setSelected));
    setMenuOpen(false);
  };

  const title = items.find((each) => each.id === selected);

  return (
    <div className="app str-chat">
      <header className="top-bar">
        <button aria-label="Conversations" className="icon-button" onClick={() => setMenuOpen(!isMenuOpen)} type="button">
          ☰
        </button>
        <h1>{title ? displayTitle(title) : "Juno"}</h1>
      </header>
      {isMenuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
      <aside className={`drawer${isMenuOpen ? " open" : ""}`} inert={!isMenuOpen}>
        <ConversationList
          conversations={conversations}
          onDelete={(session) => {
            void conversations.delete(session.id);
            if (session.id === selected) show(undefined);
          }}
          onNew={() => show(undefined)}
          onOpen={(session) => show(session.id)}
          selected={selected}
        />
      </aside>
      <main className="content">
        <ChatView key={String(screenKey(screen))} screen={screen} />
      </main>
      {isVoiceShown && chat && <VoiceMode chat={chat} screen={screen} />}
    </div>
  );
}

/**
 * What every screen does with the router: carry a conversation on, or start one on the chat
 * model with sign-ins coming back here, with the page's tools; delete one; and join its call.
 */
function screenDeps(tools: Tools): ChatScreenDeps {
  return {
    open: async (sessionId) => {
      const session = sessionId
        ? await sessions().resume(sessionId, { tools })
        : await agents.agent(Assistant.agentName).sessions.create({
            llm: Assistant.chatModel,
            returnUrl: Assistant.returnUrl,
            tools,
          });
      return AgentChat.open(session);
    },
    remove: (sessionId) => sessions().delete(sessionId),
    voiceOf: (session) => VoiceSession.of(session),
  };
}

const keys = new WeakMap<ChatScreen, number>();
let nextKey = 0;

/** A key per screen, so React draws a picked conversation from scratch. */
function screenKey(screen: ChatScreen): number {
  let key = keys.get(screen);
  if (key === undefined) {
    key = nextKey++;
    keys.set(screen, key);
  }
  return key;
}

function Notice({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="notice">
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}
