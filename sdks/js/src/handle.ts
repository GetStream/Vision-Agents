import type { Client, Schemas } from "./client.js";
import { Sessions } from "./sessions.js";

/**
 * An agent, addressed by the name it is configured under.
 *
 * A handle rather than a description: nothing here says what the agent is, only which one
 * it is. What it is was decided once, in the backend — its instructions, its skills, its
 * models — which is the point of naming it rather than spelling it out per conversation.
 *
 * ```ts
 * const agent = api.agent("docs");
 * const session = await agent.sessions.create({ title: "Is Stream better than Sendbird" });
 * ```
 *
 * A name that matches nothing configured is refused when a session is opened rather than
 * here, because this costs no request: `api.agent("docs")` is a string in a wrapper, and
 * making it validate would turn every reference into a round trip.
 */
export class AgentHandle {
  /** What the agent is called, which is what a caller knows it as. */
  readonly name: string;
  /** This agent's conversations: opening one, and reading the old ones back. */
  readonly sessions: Sessions;

  private readonly client: Client;

  constructor(client: Client, name: string) {
    this.client = client;
    this.name = name;
    this.sessions = new Sessions(client, name);
  }

  /**
   * How the agent is configured, as the backend has it.
   *
   * Server side only, which is why it is a method and not a property: how an agent is
   * configured is not a browser's to read, and a page asking gets a 403.
   */
  async config(): Promise<Schemas["AgentConfig"] | undefined> {
    const stored = await this.client.get("/v1/agents/configs", {
      query: { name: this.name },
    });
    return stored[0];
  }
}
