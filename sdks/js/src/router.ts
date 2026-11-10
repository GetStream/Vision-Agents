import type { Client, Schemas } from "./client.js";

/** Where a router's calls are made from: a stored router config, and cost labels. */
export interface RouterOptions {
  /**
   * A stored router config, by the name it was stored under or by its id. Without one, every
   * call says what it wants for itself.
   */
  config?: string;
  /** Cost labels carried onto everything routed here, on top of the config's own. */
  tags?: Record<string, string>;
}

/**
 * Looking something up, configured once.
 *
 * Search is the one routed modality a device may reach: a question and its answer are one round
 * trip, and the answer is for whoever asked rather than for the app. Everything the named config
 * holds is a default that a call's own options override.
 *
 * ```ts
 * const router = api.router({ config: "healthcare" });
 * const answer = await router.search("what changed in the pricing page");
 * ```
 */
export class Router {
  private readonly client: Client;
  private readonly options: RouterOptions;

  constructor(client: Client, options: RouterOptions = {}) {
    this.client = client;
    this.options = options;
  }

  /** Answers one question out of what is true now. */
  search(query: string, options?: Schemas["SearchOptions"]): Promise<Schemas["SearchAnswer"]> {
    const { config, tags } = this.options;
    return this.client.post("/v1/search", {
      body: {
        query,
        ...(config ? { config_id: config } : {}),
        ...(options ? { options } : {}),
        ...(tags && Object.keys(tags).length > 0 ? { tags } : {}),
      },
    });
  }
}
