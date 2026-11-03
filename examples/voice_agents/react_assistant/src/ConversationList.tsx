import { useEffect, useState } from "react";
import { Avatar } from "stream-chat-react";

import { Assistant } from "./assistant";
import { displayTitle, type Conversations, type SessionRow } from "./conversations";
import { MoonIcon, NewChatIcon, SidebarIcon, SunIcon, TrashIcon } from "./icons";
import type { Theme } from "./layout";
import { daySections } from "./sections";
import { useStore } from "./store";

/**
 * The sidebar: a new conversation, a search, the conversations there were, by day, and who is
 * signed in.
 */
export function ConversationList({
  conversations,
  selected,
  onClose,
  onNew,
  onOpen,
  onDelete,
  theme,
  onTheme,
}: {
  conversations: Conversations;
  /** The conversation open beside this, or undefined for a new one. */
  selected: string | undefined;
  /** Folds the sidebar away: to a rail on a wide window, off the side on a narrow one. */
  onClose: () => void;
  onNew: () => void;
  onOpen: (session: SessionRow) => void;
  onDelete: (session: SessionRow) => void;
  theme: Theme;
  onTheme: (theme: Theme) => void;
}) {
  const { items, failure } = useStore(conversations);
  const [search, setSearch] = useState("");

  useEffect(() => {
    // Waits for typing to pause before searching.
    conversations.setSearch(search);
    const timer = setTimeout(() => void conversations.reload(), search ? 300 : 0);
    return () => clearTimeout(timer);
  }, [conversations, search]);

  return (
    <nav className="conversation-list" aria-label="Conversations">
      <div className="sidebar-head">
        <button aria-label="Close sidebar" className="icon-button" onClick={onClose} title="Close sidebar" type="button">
          <SidebarIcon />
        </button>
        <span className="brand">Juno</span>
      </div>
      <button className="new-chat" onClick={onNew} type="button">
        <NewChatIcon /> New chat
      </button>
      <input
        aria-label="Search conversations"
        className="search"
        onChange={(event) => setSearch(event.target.value)}
        placeholder="Search conversations"
        type="search"
        value={search}
      />
      <div className="sections">
        {daySections(items, (each) => new Date(each.created_at)).map((day) => (
          // A day can appear twice, so its first conversation names it.
          <section key={day.items[0]?.id ?? day.title}>
            <h3>{day.title}</h3>
            <ul>
              {day.items.map((session) => (
                <li className={session.id === selected ? "selected" : undefined} key={session.id}>
                  <button
                    aria-current={session.id === selected ? "page" : undefined}
                    className="open"
                    onClick={() => onOpen(session)}
                    type="button"
                  >
                    {displayTitle(session)}
                  </button>
                  <button
                    aria-label={`Delete ${displayTitle(session)}`}
                    className="delete"
                    onClick={() => onDelete(session)}
                    title="Delete"
                    type="button"
                  >
                    <TrashIcon />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
        {search && items.length === 0 && !failure && <p className="list-empty">No conversations match “{search}”.</p>}
        {failure && <p className="list-failure">{failure}</p>}
      </div>
      <SignedIn>
        <ThemeSwitch onTheme={onTheme} theme={theme} />
      </SignedIn>
    </nav>
  );
}

/** The sidebar folded on a wide window: open it again, start a conversation, and who is signed in. */
export function SidebarRail({
  onExpand,
  onNew,
  theme,
  onTheme,
}: {
  onExpand: () => void;
  onNew: () => void;
  theme: Theme;
  onTheme: (theme: Theme) => void;
}) {
  return (
    <nav className="sidebar-rail" aria-label="Conversations">
      <button aria-label="Open sidebar" className="icon-button" onClick={onExpand} title="Open sidebar" type="button">
        <SidebarIcon />
      </button>
      <button aria-label="New chat" className="icon-button" onClick={onNew} title="New chat" type="button">
        <NewChatIcon />
      </button>
      <div className="rail-bottom">
        <ThemeSwitch onTheme={onTheme} theme={theme} />
      </div>
      <div title={Assistant.user.name}>
        <Avatar size="md" userName={Assistant.user.name} />
      </div>
    </nav>
  );
}

/** Who the page talks to the router as. There is no sign-in, so there is nothing to sign out of. */
function SignedIn({ children }: { children: React.ReactNode }) {
  return (
    <div className="signed-in">
      <Avatar size="md" userName={Assistant.user.name} />
      <div className="signed-in-text">
        <span className="signed-in-name">{Assistant.user.name}</span>
        <span className="signed-in-id">{Assistant.user.id}</span>
      </div>
      {children}
    </div>
  );
}

/** Light or dark, whichever the page is not in. */
function ThemeSwitch({ theme, onTheme }: { theme: Theme; onTheme: (theme: Theme) => void }) {
  const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
  return (
    <button
      aria-label={label}
      className="icon-button"
      onClick={() => onTheme(theme === "dark" ? "light" : "dark")}
      title={label}
      type="button"
    >
      {/* Keyed, so the new icon turns in as the page changes. */}
      <span className="theme-icon" key={theme}>
        {theme === "dark" ? <SunIcon /> : <MoonIcon />}
      </span>
    </button>
  );
}
