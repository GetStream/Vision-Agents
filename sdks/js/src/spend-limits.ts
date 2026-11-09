import { checkCategory, checkTeam } from "./backend.js";
import type { Client, Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";
import { amountOf, type Amount } from "./usd.js";

/** What a limit, a grant, a reset or a usage read is about. */
export type LimitSubject = Schemas["LimitSubject"];
/** The window a limit counts in: `daily`, `weekly`, `biweekly` or `monthly`. */
export type LimitInterval = Schemas["LimitInterval"];
/** Whose ids a limit names, which the router works out from the ids when none is given. */
export type LimitLevel = Schemas["LimitLevel"];

/** The ids that name one subject. Each is left out where the subject has none. */
export interface SubjectIds {
  readonly appId?: string;
  readonly apiKey?: string;
  readonly userId?: string;
  /** A team, for subject `team`, or `app_team` with an `appId`. */
  readonly team?: string;
  /** A usage category, for subject `category`, or `app_category` with an `appId`. Never a team. */
  readonly category?: string;
}

/**
 * A limit to create: what one subject may spend in each window of `interval`.
 *
 * The level is the most specific one the ids name, which the router works out: `user` with a
 * `userId` is that user's own limit, `user` with only an `appId` the default for every user
 * of the app, and `user` alone the default for every user of the organization. Keys go the
 * same way. `level` overrides it. A `team` or `category` with an `appId` is the one within
 * that app.
 */
export interface LimitSpec extends SubjectIds {
  readonly subject: LimitSubject;
  readonly interval: LimitInterval;
  readonly amount: Amount;
  readonly level?: LimitLevel;
  readonly note?: string;
}

/** Which limits `list` reads. Empty is every limit. */
export interface LimitFilter extends SubjectIds {
  readonly subject?: LimitSubject;
  readonly level?: LimitLevel;
  readonly interval?: LimitInterval;
}

/** One subject, for what it spent. */
export interface LimitTarget extends SubjectIds {
  readonly subject: LimitSubject;
}

/**
 * Counts a subject's limits from now until each window ends: every interval's, or only those
 * in `intervals`. Nothing is refunded: what it spent stays spent.
 */
export interface LimitReset extends LimitTarget {
  readonly reason: string;
  readonly intervals?: readonly LimitInterval[];
}

/**
 * A one-time amount one subject may spend past its own spent limits, until it is used up,
 * `expiresAt` passes or it is revoked. It never lifts a parent's limit or Stream's.
 */
export interface GrantSpec extends LimitTarget {
  readonly amount: Amount;
  readonly reason: string;
  /** Left out, the router's default. */
  readonly expiresAt?: Date;
}

/** Which grants `grants` reads. Empty is every grant. */
export interface GrantFilter extends SubjectIds {
  readonly subject?: LimitSubject;
  readonly state?: Schemas["LimitGrantState"];
}

/** The page size lists are read in, the most the router answers. */
const PAGE = 200;

/**
 * The customer's spend limits: on the organization, its apps, keys, end users, teams and
 * usage categories. Server side only: the router refuses a user's token with a 403
 * `server_side_only`.
 *
 * Every limit here is the customer's own. Stream's spend limit for the organization is never
 * among them, and nothing here lifts it.
 *
 * Work a limit refuses fails with a `RouterError` of status 402 and code
 * `spend_limit_reached`, or 429 `user_limit_reached` for one end user's limit. Stream's own
 * limit refuses with 402 `system_limit_reached`.
 *
 * ```ts
 * await api.spendLimits.create({ subject: "organization", interval: "monthly", amount: USD.dollars(20_000) });
 * ```
 */
export class SpendLimits {
  private readonly client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /** Every limit the filter matches, read page by page. */
  async list(filter: LimitFilter = {}): Promise<readonly Schemas["Limit"][]> {
    const subject = filtering(filter);
    const limits: Schemas["Limit"][] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await this.client.get("/v1/spend/limits", {
        query: {
          subject,
          level: filter.level,
          interval: filter.interval,
          ...queryIds(filter),
          limit: PAGE,
          cursor,
        },
      });
      limits.push(...(page.items ?? []));
      if (!page.has_more || !page.next_cursor) {
        return limits;
      }
      cursor = page.next_cursor;
    }
  }

  /** One limit, by its id. */
  get(limitId: number): Promise<Schemas["Limit"]> {
    return this.client.get("/v1/spend/limits/{limit_id}", { path: { limit_id: limitId } });
  }

  /**
   * Creates a limit.
   *
   * The router refuses with code `limit_above_parent` (the limit is above one of a subject
   * that contains it), `limit_exists` (the subject has one at that level and interval: change
   * it with `setAmount`) or `use_team`.
   */
  async create(limit: LimitSpec): Promise<Schemas["Limit"]> {
    const subject = target(limit);
    if (!limit.interval) {
      throw new ConfigurationError("a spend limit needs an interval");
    }
    return this.client.post("/v1/spend/limits", {
      body: {
        subject,
        interval: limit.interval,
        limit_micros: limitMicros(limit.amount),
        ...bodyIds(limit),
        ...(limit.level ? { level: limit.level } : {}),
        ...(limit.note ? { note: limit.note } : {}),
      },
    });
  }

  /** Changes what a limit allows. What it is on never changes. */
  async setAmount(limitId: number, amount: Amount): Promise<Schemas["Limit"]> {
    return this.update(limitId, { limit_micros: limitMicros(amount) });
  }

  /** Changes a limit's note. An empty one clears it. */
  setNote(limitId: number, note: string): Promise<Schemas["Limit"]> {
    return this.update(limitId, { note });
  }

  /** @internal Changes the amount and the note in one request. */
  update(limitId: number, change: Schemas["LimitChange"]): Promise<Schemas["Limit"]> {
    return this.client.patch("/v1/spend/limits/{limit_id}", {
      path: { limit_id: limitId },
      body: change,
    });
  }

  /** Deletes a limit. Its subject's spend from then on is held only to its parents'. */
  delete(limitId: number): Promise<void> {
    return this.client.delete("/v1/spend/limits/{limit_id}", { path: { limit_id: limitId } });
  }

  /**
   * Counts a subject's limits from now. The router refuses with code `no_limit` a subject
   * with no limit of the customer's.
   */
  async reset(reset: LimitReset): Promise<Schemas["SubjectUsage"]> {
    if (!reset.reason) {
      throw new ConfigurationError("a reset needs a reason");
    }
    const subject = target(reset);
    return this.client.post("/v1/spend/limits/usage/reset", {
      body: {
        subject,
        reason: reset.reason,
        ...bodyIds(reset),
        ...(reset.intervals && reset.intervals.length > 0 ? { intervals: reset.intervals } : {}),
      },
    });
  }

  /** What one subject spent in its current windows, against its limits and grants. */
  async usage(of: LimitTarget): Promise<Schemas["SubjectUsage"]> {
    const subject = target(of);
    const listed = (value: string | undefined) => (value ? [value] : undefined);
    const answered = await this.client.get("/v1/spend/limits/usage", {
      query: {
        subject,
        app_id: listed(of.appId),
        api_key_id: listed(of.apiKey),
        user_id: listed(of.userId),
        team: listed(of.team),
        category: listed(of.category),
      },
    });
    const [one] = answered;
    if (answered.length !== 1 || !one) {
      throw new ConfigurationError(
        `reading the spend of one ${subject}: the router answered ${answered.length}`,
      );
    }
    return one;
  }

  /**
   * Gives one subject a one-time amount past its spent limits. "$10 more today" is a grant
   * that expires at the next midnight UTC.
   */
  async grant(grant: GrantSpec): Promise<Schemas["LimitGrant"]> {
    if (!grant.reason) {
      throw new ConfigurationError("a grant needs a reason");
    }
    const subject = target(grant);
    return this.client.post("/v1/spend/limits/grants", {
      body: {
        subject,
        amount_micros: limitMicros(grant.amount),
        reason: grant.reason,
        ...bodyIds(grant),
        ...(grant.expiresAt ? { expires_at: grant.expiresAt.toISOString() } : {}),
      },
    });
  }

  /** Every grant the filter matches, newest first, read page by page. */
  async grants(filter: GrantFilter = {}): Promise<readonly Schemas["LimitGrant"][]> {
    const subject = filtering(filter);
    const grants: Schemas["LimitGrant"][] = [];
    let cursor: string | undefined;
    for (;;) {
      const page = await this.client.get("/v1/spend/limits/grants", {
        query: { subject, state: filter.state, ...queryIds(filter), limit: PAGE, cursor },
      });
      grants.push(...(page.items ?? []));
      if (!page.has_more || !page.next_cursor) {
        return grants;
      }
      cursor = page.next_cursor;
    }
  }

  /** Ends a grant before it is used up or expires. */
  async revokeGrant(grantId: number, reason: string): Promise<Schemas["LimitGrant"]> {
    if (!reason) {
      throw new ConfigurationError("revoking a grant needs a reason");
    }
    return this.client.post("/v1/spend/limits/grants/{grant_id}/revoke", {
      path: { grant_id: grantId },
      body: { reason },
    });
  }

  /** Gives back a revoked grant what it had left. */
  async unrevokeGrant(grantId: number, reason: string): Promise<Schemas["LimitGrant"]> {
    if (!reason) {
      throw new ConfigurationError("unrevoking a grant needs a reason");
    }
    return this.client.post("/v1/spend/limits/grants/{grant_id}/unrevoke", {
      path: { grant_id: grantId },
      body: { reason },
    });
  }
}

function limitMicros(amount: Amount): number {
  const usd = amountOf(amount);
  if (usd.micros < 0n) {
    throw new ConfigurationError(`a spend limit of ${usd.toString()} is below zero`);
  }
  return usd.toWire();
}

/** The ids as a body names them, each only when set. */
function bodyIds(ids: SubjectIds): {
  app_id?: string;
  api_key?: string;
  user_id?: string;
  team?: string;
  category?: string;
} {
  return {
    ...(ids.appId ? { app_id: ids.appId } : {}),
    ...(ids.apiKey ? { api_key: ids.apiKey } : {}),
    ...(ids.userId ? { user_id: ids.userId } : {}),
    ...(ids.team ? { team: ids.team } : {}),
    ...(ids.category ? { category: ids.category } : {}),
  };
}

/** The ids as a list's query names them; `api_key` there is the caller's own credential. */
function queryIds(ids: SubjectIds) {
  return {
    app_id: ids.appId || undefined,
    api_key_id: ids.apiKey || undefined,
    user_id: ids.userId || undefined,
    team: ids.team || undefined,
    category: ids.category || undefined,
  };
}

/** The subject a write is about, as the router names it. */
function target(of: LimitTarget): LimitSubject {
  if (!of.subject) {
    throw new ConfigurationError(
      "name the subject: organization, app, api_key, user, team or category",
    );
  }
  return kindOf(of);
}

/** The subject a list filters on, as the router names it, with its names checked. */
function filtering(filter: SubjectIds & { readonly subject?: LimitSubject }): LimitSubject | undefined {
  if (filter.subject) {
    return kindOf({ ...filter, subject: filter.subject });
  }
  if (filter.team) {
    checkTeam(filter.team);
  }
  if (filter.category) {
    checkCategory(filter.category);
  }
  return undefined;
}

/**
 * The subject as the router names it: a team or category within an app is `app_team` or
 * `app_category`. It refuses what the router would, before anything is sent.
 */
function kindOf(of: LimitTarget): LimitSubject {
  let subject = of.subject;
  if (of.category) {
    checkCategory(of.category);
  }
  if (of.team) {
    checkTeam(of.team);
  }
  if (subject === "team" || subject === "app_team") {
    if (!of.team) {
      throw new ConfigurationError(`a ${subject} subject names its team in team`);
    }
    if (subject === "team" && of.appId) {
      subject = "app_team";
    }
  }
  if (subject === "category" || subject === "app_category") {
    if (!of.category) {
      throw new ConfigurationError(`a ${subject} subject names its category in category`);
    }
    if (subject === "category" && of.appId) {
      subject = "app_category";
    }
  }
  if (of.team && subject !== "team" && subject !== "app_team") {
    throw new ConfigurationError(`team is for the team subject, not ${subject}`);
  }
  if (of.category && subject !== "category" && subject !== "app_category") {
    throw new ConfigurationError(`category is for the category subject, not ${subject}`);
  }
  if ((subject === "app_team" || subject === "app_category") && !of.appId) {
    throw new ConfigurationError(`an ${subject} subject names its app in appId`);
  }
  return subject;
}
