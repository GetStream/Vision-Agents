import { useEffect, useMemo, useRef, useState } from "react";
import { formatMessage } from "stream-chat";
import { Chat } from "stream-chat-react";

import { Session, type Tools } from "@stream-io/calypso";
import { VoiceSession } from "@stream-io/calypso/video";

import { AgentChat, PAGE_SIZE } from "./agentChat";
import { agents, Assistant, chatClient, sessions, signIn } from "./assistant";
import { ChatScreen, type ChatScreenDeps } from "./chatScreen";
import { ChatView } from "./ChatView";
import { ConversationList, SidebarRail } from "./ConversationList";
import { Conversations, displayTitle, type SessionRow } from "./conversations";
import { messageOf } from "./errors";
import { ReplyFormatMenu } from "./ReplyFormatMenu";
import { deviceTools } from "./deviceTools";
import { MenuIcon, NewChatIcon } from "./icons";
import { useMediaQuery, useStoredFlag, useTheme, WIDE } from "./layout";
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
  // Stream's components are light unless told otherwise, so they are told the page's appearance.
  const [theme, setTheme] = useTheme();
  /** The conversation open, or undefined for a new one. A link to one opens it. */
  const [selected, setSelected] = useState<string | undefined>(conversationInUrl);
  // A new conversation goes in the URL once it has a session, in place of the empty one it was
  // started on, so reloading or bookmarking it comes back to it.
  const started = (sessionId: string) => {
    setSelected(sessionId);
    putConversationInUrl(sessionId, "replace");
  };
  // Which conversation the screen was opened for. It changes only when the person picks one, so
  // a new conversation getting its session keeps the screen it was started on.
  const [screen, setScreen] = useState(() => new ChatScreen(conversationInUrl(), deps, started));
  const { items } = useStore(conversations);
  const { chat, isVoiceShown } = useStore(screen);

  useEffect(() => {
    const leave = () => screen.leave();
    window.addEventListener("pagehide", leave);
    return () => window.removeEventListener("pagehide", leave);
  }, [screen]);

  useEffect(() => {
    // A page put away whole by the browser and brought back by Back left its conversation as it
    // went, so it opens it again rather than showing one whose session is closed.
    const reopen = (event: PageTransitionEvent) => {
      if (event.persisted) latest.current.open(latest.current.selected);
    };
    window.addEventListener("pageshow", reopen);
    return () => window.removeEventListener("pageshow", reopen);
  }, []);

  useEffect(() => {
    // The router titles a conversation as it goes, so the list is read again each time it is shown.
    if (isListShown) void conversations.reload();
  }, [conversations, isListShown]);

  // And when the conversation open is not in it: a new one that just got its session, or one
  // opened from a link.
  const isListed = selected === undefined || items.some((each) => each.id === selected);
  useEffect(() => {
    if (isListShown && !isListed) void conversations.reload();
  }, [conversations, isListShown, isListed]);

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

  const open = (sessionId: string | undefined) => {
    screen.leave();
    setSelected(sessionId);
    const row = items.find((each) => each.id === sessionId);
    setScreen(new ChatScreen(sessionId, deps, started, row?.conversation_id ?? undefined));
    setMenuOpen(false);
  };

  /**
   * Opens a conversation, or a new one, as a page of its own that Back returns from, or in place
   * of the current page when that one is gone.
   */
  const show = (session: SessionRow | undefined, how: "push" | "replace" = "push") => {
    open(session?.id);
    putConversationInUrl(session?.id, how);
  };

  // What the window's listeners act on, kept current without subscribing them again each render.
  const latest = useRef({ open, selected });
  latest.current = { open, selected };

  useEffect(() => {
    // Back and Forward go between the conversations opened, as between pages.
    const followUrl = () => {
      const sessionId = conversationInUrl();
      if (sessionId !== latest.current.selected) latest.current.open(sessionId);
    };
    window.addEventListener("popstate", followUrl);
    return () => window.removeEventListener("popstate", followUrl);
  }, []);

  const title = items.find((each) => each.id === selected);

  useEffect(() => {
    // What a bookmark or a tab is called.
    document.title = title ? `${displayTitle(title)} · Juno` : "Juno";
  }, [title]);
  const list = (
    <ConversationList
      conversations={conversations}
      onClose={() => (isWide ? setCollapsed(true) : setMenuOpen(false))}
      onDelete={(session) => {
        void conversations.delete(session.id);
        // Back must not lead to a conversation that is gone.
        if (session.id === selected) show(undefined, "replace");
      }}
      onNew={() => show(undefined)}
      onOpen={(session) => show(session)}
      onTheme={setTheme}
      selected={selected}
      theme={theme}
    />
  );

  return (
    <div className={`app str-chat str-chat__theme-${theme} ${isWide ? "wide" : "narrow"}`}>
      {isWide ? (
        // One sidebar whose width moves between the list and the rail, so folding it is seen.
        <aside className={`sidebar${isCollapsed ? " collapsed" : ""}`}>
          {isCollapsed ? (
            <SidebarRail onExpand={() => setCollapsed(false)} onNew={() => show(undefined)} onTheme={setTheme} theme={theme} />
          ) : (
            list
          )}
        </aside>
      ) : (
        <>
          {/* Always there, so it can fade out as well as in. */}
          <div aria-hidden className={`scrim${isMenuOpen ? " open" : ""}`} onClick={() => setMenuOpen(false)} />
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
          <ReplyFormatMenu />
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
  // A conversation being opened is read from the router once, by whichever of carrying it on and
  // reading its messages asks first, and forgotten once it is open.
  const reading = new Map<string, Promise<SessionRow>>();
  const read = (sessionId: string) => {
    let row = reading.get(sessionId);
    if (!row) {
      row = sessions().get(sessionId);
      reading.set(sessionId, row);
    }
    return row;
  };
  return {
    open: async (sessionId) => {
      if (!sessionId) {
        const session = await agents.agent(Assistant.agentName).sessions.create({
          // An overwrite rather than `llm`: the router keeps it on the session, so a conversation
          // carried on after it ended still runs on this model rather than the agent's own.
          modelOverwrites: { llm: Assistant.chatModel },
          returnUrl: Assistant.returnUrl,
          tools,
        });
        return AgentChat.open(session);
      }
      try {
        // What `sessions.resume` does, with the row the messages may already have read.
        return await AgentChat.open(await Session.watching(agents, await read(sessionId), { tools }));
      } finally {
        reading.delete(sessionId);
      }
    },
    history: async (sessionId, conversationId) => {
      // The channel the session writes to, read as the page's user, who is a member of it. A
      // conversation opened from a link has only its id, so its session says which channel.
      const channelId = conversationId ?? (await read(sessionId)).conversation_id;
      if (!channelId) return [];
      const [type = "agent", ...rest] = channelId.split(":");
      const channel = chatClient.channel(type, rest.join(":"));
      // A query's messages are its answer only: stream-chat 10 keeps them out of the channel's list.
      const { messages } = await channel.query({ messages: { limit: PAGE_SIZE }, watch: false });
      return messages.map(formatMessage);
    },
    remove: (sessionId) => sessions().delete(sessionId),
    voiceOf: (session) => VoiceSession.of(session),
  };
}

/** The query parameter a conversation's link names it by. */
const CONVERSATION_PARAM = "conversation";

/** The conversation the page's URL names, or undefined for a new one. */
function conversationInUrl(): string | undefined {
  return new URLSearchParams(window.location.search).get(CONVERSATION_PARAM) || undefined;
}

/**
 * Names a conversation, or none for a new one, in the page's URL: as a new history entry when it
 * was picked, or in place of the current one when a new conversation got its session.
 */
function putConversationInUrl(sessionId: string | undefined, how: "push" | "replace"): void {
  const url = new URL(window.location.href);
  if ((url.searchParams.get(CONVERSATION_PARAM) || undefined) === sessionId) return;
  if (sessionId) {
    url.searchParams.set(CONVERSATION_PARAM, sessionId);
  } else {
    url.searchParams.delete(CONVERSATION_PARAM);
  }
  if (how === "push") {
    window.history.pushState(null, "", url);
  } else {
    window.history.replaceState(null, "", url);
  }
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
