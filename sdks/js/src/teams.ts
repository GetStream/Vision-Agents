import { checkTeam } from "./backend.js";
import type { Client, Schemas } from "./client.js";
import { ConfigurationError, RouterError } from "./errors.js";
import type { LimitInterval, SpendLimits } from "./spend-limits.js";
import { amountOf, type Amount } from "./usd.js";

/** What one team may spend in each window of `interval`: across the organization, or within `appId`. */
export interface TeamLimit {
  readonly interval: LimitInterval;
  readonly amount: Amount;
  readonly appId?: string;
  readonly note?: string;
}

/** Which of a team's limits: across the organization, or within `appId`. */
export interface TeamLimitTarget {
  readonly interval: LimitInterval;
  readonly appId?: string;
}

/**
 * The organization's teams. Server side only.
 *
 * A team has no members list: a user is in the team their token names, from
 * `createUserToken(userId, { team })` or a client built with `team`. A team's spend is the
 * spend of the work those tokens ran, across the organization or within one app.
 *
 * ```ts
 * await api.teams.setLimit("finance", { interval: "monthly", amount: USD.dollars(8_000) });
 * ```
 */
export class Teams {
  private readonly client: Client;
  private readonly limits: SpendLimits;

  constructor(client: Client, limits: SpendLimits) {
    this.client = client;
    this.limits = limits;
  }

  /**
   * Every team with a limit or with spend this month, each with its apps, its spend in the
   * current windows and its limits. With an `appId`, only within that app.
   */
  list(options: { readonly appId?: string } = {}): Promise<readonly Schemas["Team"][]> {
    return this.client.get("/v1/teams", { query: { app_id: options.appId || undefined } });
  }

  /**
   * Makes the team's limit for the interval the amount given: created when it has none,
   * changed when it has one.
   */
  async setLimit(team: string, limit: TeamLimit): Promise<Schemas["Limit"]> {
    const amount = amountOf(limit.amount);
    if (amount.micros < 0n) {
      throw new ConfigurationError(`a spend limit of ${amount.toString()} is below zero`);
    }
    let existing = await this.limit(team, limit);
    if (!existing) {
      try {
        return await this.limits.create({
          subject: "team",
          team,
          interval: limit.interval,
          amount,
          ...(limit.appId ? { appId: limit.appId } : {}),
          ...(limit.note ? { note: limit.note } : {}),
        });
      } catch (error) {
        // Somebody else created it since it was looked up.
        if (!(error instanceof RouterError) || error.code !== "limit_exists") {
          throw error;
        }
      }
      existing = await this.limit(team, limit);
      if (!existing) {
        throw new ConfigurationError(
          `the ${limit.interval} limit on team ${team} exists and could not be found`,
        );
      }
    }
    return this.limits.update(existing.id, {
      limit_micros: amount.toWire(),
      ...(limit.note ? { note: limit.note } : {}),
    });
  }

  /**
   * Deletes the team's limit for the interval, across the organization or within `appId`.
   * Nothing happens when it has none.
   */
  async removeLimit(team: string, target: TeamLimitTarget): Promise<void> {
    const existing = await this.limit(team, target);
    if (!existing) {
      return;
    }
    try {
      await this.limits.delete(existing.id);
    } catch (error) {
      // Somebody else deleted it since it was looked up.
      if (!(error instanceof RouterError) || error.status !== 404) {
        throw error;
      }
    }
  }

  /**
   * What the team spent in its current windows, against its limits and grants, across the
   * organization or within `appId`.
   */
  usage(team: string, options: { readonly appId?: string } = {}): Promise<Schemas["SubjectUsage"]> {
    return this.limits.usage({ subject: "team", team, ...(options.appId ? { appId: options.appId } : {}) });
  }

  /** The team's own limit in the interval, across the organization or within the app. */
  private async limit(team: string, target: TeamLimitTarget): Promise<Schemas["Limit"] | undefined> {
    checkTeam(team);
    if (!target.interval) {
      throw new ConfigurationError("a team's limit needs an interval");
    }
    const inApp = target.appId ?? "";
    const limits = await this.limits.list({
      subject: "team",
      team,
      interval: target.interval,
      ...(inApp ? { appId: inApp } : {}),
    });
    return limits.find(
      (limit) => limit.team === team && limit.interval === target.interval && (limit.app_id ?? "") === inApp,
    );
  }
}
