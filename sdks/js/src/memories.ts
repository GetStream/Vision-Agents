import type { Client } from "./client.js";
import { ConfigurationError } from "./errors.js";

/**
 * What the app's agents remember about its users between conversations.
 *
 * Server side only: deleting a user's memories is the app's decision, and a page asking gets
 * a 403.
 */
export class Memories {
  private readonly client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /**
   * Deletes everything remembered about one user: every session's and every agent's,
   * whatever memory filter it was written under.
   *
   * `userId` is the `user_id` of the memory filter the sessions were opened with. A user
   * nothing is known about is not an error.
   */
  async truncate(userId: string): Promise<void> {
    if (!userId) {
      throw new ConfigurationError("truncating memories needs a user id");
    }
    await this.client.delete("/v1/agents/users/{user_id}/memories", {
      path: { user_id: userId },
    });
  }
}
