import { useEffect, useMemo, useState } from "react";
import { Chat } from "stream-chat-react";

import type { Tools } from "@stream-io/calypso";
import { VoiceSession } from "@stream-io/calypso/video";

import { AgentChat } from "./agentChat";
import { agents, Assistant, chatClient, sessions, signIn } from "./assistant";
import { ChatScreen, type ChatScreenDeps } from "./chatScreen";
import { ChatView } from "./ChatView";
import { ConversationList, SidebarRail } from "./ConversationList";
import { Conversations, displayTitle } from "./conversations";
import { messageOf } from "./errors";
import { deviceTools } from "./deviceTools";
import { MenuIcon, NewChatIcon } from "./icons";
import { DARK, useMediaQuery, useStoredFlag, WIDE } from "./layout";
import { scheduleReminders } from "./reminders";
import { useStore } from "./store";
import { VoiceMode } from "./VoiceMode";

/**
 * The conversation in front, and the ones there were in a sidebar beside it, or, on a narrow
 * window, in a drawer that slides over it.
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
  // A wide window keeps the conversations beside the chat, folded to a rail when the person
  // chooses; a narrow one slides them over it from the menu button, as the phones do.
  const isWide = useMediaQuery(WIDE);
  const [isCollapsed, setCollapsed] = useStoredFlag("juno.sidebar-collapsed", false);
  const [isMenuOpen, setMenuOpen] = useState(false);
  const isListShown = isWide ? !isCollapsed : isMenuOpen;
  // Stream's components are light unless told otherwise, so they follow the system with the page.
  const isDark = useMediaQuery(DARK);
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
    // The router titles a conversation as it goes, so the list is read again each time it is
    // shown, and when a new conversation gets its session.
    if (isListShown) void conversations.reload();
  }, [conversations, isListShown, selected]);

  useEffect(() => {
    // And once each reply is written, which is when a new conversation has its title.
    if (!chat || !isListShown) return;
    let wasAnswering = chat.isAnswering();
    return chat.subscribe(() => {
      const isAnswering = chat.isAnswering();
      if (wasAnswering && !isAnswering) void conversations.reload();
      wasAnswering = isAnswering;
    });
  }, [chat, conversations, isListShown]);

  useEffect(() => {
    if (!isMenuOpen) return;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [isMenuOpen]);

  const show = (sessionId: string | undefined) => {
    screen.leave();
    setSelected(sessionId);
    setScreen(new ChatScreen(sessionId, deps, setSelected));
    setMenuOpen(false);
  };

  const title = items.find((each) => each.id === selected);
  const list = (
    <ConversationList
      conversations={conversations}
      onClose={() => (isWide ? setCollapsed(true) : setMenuOpen(false))}
      onDelete={(session) => {
        void conversations.delete(session.id);
        if (session.id === selected) show(undefined);
      }}
      onNew={() => show(undefined)}
      onOpen={(session) => show(session.id)}
      selected={selected}
    />
  );

  return (
    <div className={`app str-chat str-chat__theme-${isDark ? "dark" : "light"} ${isWide ? "wide" : "narrow"}`}>
      {isWide ? (
        isCollapsed ? (
          <SidebarRail onExpand={() => setCollapsed(false)} onNew={() => show(undefined)} />
        ) : (
          <aside className="sidebar">{list}</aside>
        )
      ) : (
        <>
          {isMenuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} />}
          <aside className={`drawer${isMenuOpen ? " open" : ""}`} inert={!isMenuOpen}>
            {list}
          </aside>
        </>
      )}
      <div className="main">
        <header className="top-bar">
          {!isWide && (
            <button aria-label="Open sidebar" className="icon-button" onClick={() => setMenuOpen(true)} type="button">
              <MenuIcon />
            </button>
          )}
          <h1>{title ? displayTitle(title) : "Juno"}</h1>
          {!isWide && (
            <button aria-label="New chat" className="icon-button" onClick={() => show(undefined)} type="button">
              <NewChatIcon />
            </button>
          )}
        </header>
        <main className="content">
          <ChatView key={String(screenKey(screen))} screen={screen} />
          {/* Over the conversation in this pane on a wide window, over everything on a narrow one.
              Picking another conversation in the sidebar hangs up, as leaving this one does. */}
          {isVoiceShown && chat && <VoiceMode chat={chat} screen={screen} />}
        </main>
      </div>
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
