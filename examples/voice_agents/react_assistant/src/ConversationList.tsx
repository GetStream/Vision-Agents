import { useEffect, useState } from "react";

import { displayTitle, type Conversations, type SessionRow } from "./conversations";
import { daySections } from "./sections";
import { useStore } from "./store";

/** The sidebar: a new conversation, a search, and the conversations there were, by day. */
export function ConversationList({
  conversations,
  selected,
  onNew,
  onOpen,
  onDelete,
}: {
  conversations: Conversations;
  /** The conversation open beside this, or undefined for a new one. */
  selected: string | undefined;
  onNew: () => void;
  onOpen: (session: SessionRow) => void;
  onDelete: (session: SessionRow) => void;
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
      <button className="new-chat" onClick={onNew} type="button">
        ✎ New chat
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
                  <button className="open" onClick={() => onOpen(session)} type="button">
                    {displayTitle(session)}
                  </button>
                  <button
                    aria-label={`Delete ${displayTitle(session)}`}
                    className="delete"
                    onClick={() => onDelete(session)}
                    type="button"
                  >
                    🗑
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
        {failure && <p className="list-failure">{failure}</p>}
      </div>
    </nav>
  );
}
