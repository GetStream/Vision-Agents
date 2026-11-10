import type { Schemas } from "@stream-io/stream-agents";

import { sessions } from "./assistant";
import { messageOf } from "./errors";
import { Store } from "./store";

export type SessionRow = Schemas["Session"];

/**
 * This person's conversations with the assistant, most recently active first.
 *
 * The router keeps them: a session belongs to whoever opened it, so this lists this page's
 * user's and nobody else's. It titles each one itself once there is something to go on, and a
 * conversation that ended carries on when it is next asked something, so there is nothing to
 * keep here but the list.
 */
export class Conversations extends Store<{
  items: readonly SessionRow[];
  /** The search `items` were read for, or empty for all of them. */
  searched: string;
  failure: string | undefined;
}> {
  /** What the list is narrowed to, or empty for all of them. */
  private search = "";
  /** Counts the reads, so only the latest one's answer is shown. */
  private reads = 0;

  constructor() {
    super({ items: [], searched: "", failure: undefined });
  }

  /** Narrows the list to conversations matching `text` from the next read on. */
  setSearch(text: string): void {
    this.search = text;
  }

  /** Reads the list again, narrowed to the search when there is one. */
  async reload(): Promise<void> {
    const read = ++this.reads;
    const search = this.search;
    try {
      const query = { limit: 50 };
      const page = search ? await sessions().search(search, query) : await sessions().query(query);
      // A newer read was asked for meanwhile, such as another search: its answer is the one shown.
      if (read !== this.reads) return;
      // Only the ones kept in writing can be opened again as a chat: a session that was only
      // ever a call, or incognito, has no channel to read back.
      this.update({ items: page.items.filter((each) => Boolean(each.conversation_id)), searched: search, failure: undefined });
    } catch (cause) {
      if (read === this.reads) this.update({ failure: messageOf(cause) });
    }
  }

  /** Deletes a conversation: its turns, and what it remembered, go with it. */
  async delete(id: string): Promise<void> {
    this.update({ items: this.get().items.filter((each) => each.id !== id) });
    try {
      await sessions().delete(id);
    } catch (cause) {
      this.update({ failure: messageOf(cause) });
      await this.reload();
    }
  }
}

/** What a conversation is called in the list: the router's title once it has given it one. */
export function displayTitle(session: SessionRow | undefined): string {
  return session?.title || "New conversation";
}
