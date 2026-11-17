import type { Client, Schemas } from "./client.js";

/** Which of the user's tasks to list. A filter left out is not applied. */
export interface ScheduledTaskQuery {
  /** Only the ones in this state. */
  state?: Schemas["ScheduledTask"]["state"];
  /** Only the ones on this agent config. */
  configId?: string;
  /** Only the ones following up in this conversation. */
  sessionId?: string;
  /** Up to 200. Omitted is 25. */
  limit?: number;
  /** The `next_cursor` of the page before, asked with the same filters. Omitted is the first page. */
  cursor?: string;
}

/**
 * The reminders and later work agents hold for the user `setUser` named.
 *
 * The agent makes a task in a conversation, when the person asks for one, and the app's backend
 * can make one in their name. A device cannot: what the agent is told when a task comes due is
 * the backend's to write. What is here is what a person does with a reminder they already have:
 * find it, move it, pause it and cancel it.
 *
 * ```ts
 * const page = await api.scheduledTasks.list();
 * await api.scheduledTasks.update(page.items[0].id, { next: { in: "PT15M" } });
 * ```
 */
export class ScheduledTasks {
  private readonly client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /**
   * One page of the user's tasks, newest first, the ones their backend made included. Pass the
   * page's `next_cursor` as `cursor` for the next one.
   */
  list(query: ScheduledTaskQuery = {}): Promise<Schemas["ScheduledTaskPage"]> {
    return this.client.get("/v1/agents/scheduled-tasks", {
      query: {
        ...(query.state ? { state: query.state } : {}),
        ...(query.configId ? { config_id: query.configId } : {}),
        ...(query.sessionId ? { session_id: query.sessionId } : {}),
        ...(query.limit ? { limit: query.limit } : {}),
        ...(query.cursor ? { cursor: query.cursor } : {}),
      },
    });
  }

  /** One task. Somebody else's is reported as not found. */
  get(id: string): Promise<Schemas["ScheduledTask"]> {
    return this.client.get("/v1/agents/scheduled-tasks/{id}", { path: { id } });
  }

  /**
   * Changes a task and answers it as it now is. A field left out is left as it is.
   *
   * `schedule` replaces the whole schedule, but for a timezone it leaves out. `next` moves only
   * the next occurrence: a snooze, or a skip when it lands past the one after. Either lands 15
   * minutes from now at the soonest, unless it is no earlier than the current next occurrence.
   * An empty `deliver_to` is the conversation alone, and `quiet_hours` with both ends empty
   * clears the hold.
   */
  update(id: string, changes: Schemas["UpdateScheduledTaskRequest"]): Promise<Schemas["ScheduledTask"]> {
    return this.client.patch("/v1/agents/scheduled-tasks/{id}", { path: { id }, body: changes });
  }

  /**
   * Holds a task: nothing comes due while it is paused, and what it slept through is never
   * replayed.
   */
  pause(id: string): Promise<Schemas["ScheduledTask"]> {
    return this.client.post("/v1/agents/scheduled-tasks/{id}/pause", { path: { id } });
  }

  /**
   * Starts a paused or failed task again, counting its next occurrence from now. A one-shot
   * whose time came while it was paused is completed instead, and a task with no occurrence
   * left needs a new schedule first.
   */
  resume(id: string): Promise<Schemas["ScheduledTask"]> {
    return this.client.post("/v1/agents/scheduled-tasks/{id}/resume", { path: { id } });
  }

  /** Cancels a task: it and its runs are gone. */
  delete(id: string): Promise<void> {
    return this.client.delete("/v1/agents/scheduled-tasks/{id}", { path: { id } });
  }

  /**
   * One page of the times a task came due, newest first. Snoozes and pauses write none. Pass
   * the page's `next_cursor` as `cursor` for the next one.
   */
  runs(id: string, options: { limit?: number; cursor?: string } = {}): Promise<Schemas["ScheduledTaskRunPage"]> {
    return this.client.get("/v1/agents/scheduled-tasks/{id}/runs", {
      path: { id },
      query: {
        ...(options.limit ? { limit: options.limit } : {}),
        ...(options.cursor ? { cursor: options.cursor } : {}),
      },
    });
  }
}
