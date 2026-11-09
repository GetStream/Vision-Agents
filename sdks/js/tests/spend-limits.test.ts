import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, ConfigurationError, RouterError, USD, type Schemas } from "../src/index.js";
import { TestRouter, claimsOf } from "./router.js";

/** A limit as the router describes one, with only the fields a test cares about set. */
function limit(over: Partial<Schemas["Limit"]> = {}): Schemas["Limit"] {
  return {
    id: 7,
    subject: "organization",
    level: "organization",
    interval: "monthly",
    limit_micros: 20_000_000_000,
    effective_limit_micros: 20_000_000_000,
    created_at: "2026-11-09T00:00:00Z",
    updated_at: "2026-11-09T00:00:00Z",
    updated_by: "server",
    ...over,
  };
}

function usage(over: Partial<Schemas["SubjectUsage"]> = {}): Schemas["SubjectUsage"] {
  return { subject: "user", grants: [], windows: [], ...over };
}

describe("USD", () => {
  it("holds dollars, cents and micros exactly", () => {
    assert.equal(USD.dollars(20_000).micros, 20_000_000_000n);
    assert.equal(USD.cents(1_250).micros, 12_500_000n);
    assert.equal(USD.fromMicros(300).micros, 300n);
    assert.equal(USD.dollars(10n).micros, 10_000_000n);
  });

  it("reads an amount written out", () => {
    for (const [written, micros] of [
      ["12", 12_000_000n],
      ["12.5", 12_500_000n],
      ["$12.50", 12_500_000n],
      ["-3.25", -3_250_000n],
      ["0.000300", 300n],
      [".5", 500_000n],
      ["0.1000000", 100_000n],
    ] as const) {
      assert.equal(USD.parse(written).micros, micros, written);
    }
  });

  it("refuses what is not plain dollars, or is finer than a micro", () => {
    for (const written of ["", "$", ".", "12.", "1,000", "1e3", "12.0000001", "abc", "--1"]) {
      assert.throws(() => USD.parse(written), ConfigurationError, written);
    }
  });

  it("refuses a fractional number, which is where 0.1 + 0.2 comes from", () => {
    assert.throws(() => USD.dollars(0.1 + 0.2), /USD\.parse/);
    assert.throws(() => USD.cents(12.5), ConfigurationError);
    assert.throws(() => USD.fromMicros(Number.NaN), ConfigurationError);
  });

  it("prints with cents always and finer digits only when it has them", () => {
    assert.equal(USD.dollars(20_000).toString(), "$20000.00");
    assert.equal(USD.parse("12.5").toString(), "$12.50");
    assert.equal(USD.fromMicros(300).toString(), "$0.0003");
    assert.equal(USD.parse("-3.25").toString(), "-$3.25");
    assert.ok(USD.parse(USD.parse("7.000001").toString()).equals(USD.parse("7.000001")));
  });

  it("refuses an amount a JSON number cannot carry exactly", () => {
    assert.throws(() => USD.fromMicros(2n ** 60n).toWire(), ConfigurationError);
    assert.throws(() => USD.fromMicros(2n ** 63n), ConfigurationError);
  });
});

describe("spend limits", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("creates a limit with the amount in micros and only the ids it names", async () => {
    router.serve("POST", "/v1/spend/limits", { status: 201, body: limit() });

    await api.spendLimits.create({ subject: "organization", interval: "monthly", amount: USD.dollars(20_000) });
    await api.spendLimits.create({
      subject: "api_key",
      appId: "1234",
      apiKey: "mz7kq2xa9b",
      interval: "daily",
      amount: "10",
      note: "CI key",
    });

    const [org, key] = router.requestsTo("POST", "/v1/spend/limits");
    assert.deepEqual(org?.body, { subject: "organization", interval: "monthly", limit_micros: 20_000_000_000 });
    assert.deepEqual(key?.body, {
      subject: "api_key",
      interval: "daily",
      limit_micros: 10_000_000,
      app_id: "1234",
      api_key: "mz7kq2xa9b",
      note: "CI key",
    });
  });

  it("names a team or category within an app as the app's", async () => {
    router.serve("POST", "/v1/spend/limits", { status: 201, body: limit() });

    await api.spendLimits.create({ subject: "team", team: "finance", appId: "1234", interval: "weekly", amount: "1" });
    await api.spendLimits.create({ subject: "category", category: "project:x", appId: "1234", interval: "weekly", amount: "1" });

    const [team, category] = router.requestsTo("POST", "/v1/spend/limits");
    assert.equal((team?.body as Record<string, unknown>)["subject"], "app_team");
    assert.equal((category?.body as Record<string, unknown>)["subject"], "app_category");
  });

  it("refuses before sending what the router would refuse", async () => {
    const refused: Parameters<Client["spendLimits"]["create"]>[0][] = [
      { subject: "organization", interval: "daily", amount: "-1" },
      { subject: "team", interval: "daily", amount: "1" },
      { subject: "team", team: "has space", interval: "daily", amount: "1" },
      { subject: "team", team: "x".repeat(60), interval: "daily", amount: "1" },
      { subject: "category", category: "Team:finance", interval: "daily", amount: "1" },
      { subject: "user", team: "finance", interval: "daily", amount: "1" },
      { subject: "app_team", team: "finance", interval: "daily", amount: "1" },
      { subject: "organization", interval: "daily", amount: "1.0000001" },
    ];
    for (const spec of refused) {
      await assert.rejects(() => api.spendLimits.create(spec), ConfigurationError, JSON.stringify(spec));
    }
    await assert.rejects(
      () => api.spendLimits.create({ subject: "organization", interval: "daily", amount: 12.5 as unknown as string }),
      /a number/,
    );
    assert.equal(router.received.length, 0);
  });

  it("lists every page, with the key filter named api_key_id", async () => {
    router.serve("GET", "/v1/spend/limits", (_, calls) => ({
      body:
        calls === 0
          ? { items: [limit({ id: 1 })], has_more: true, next_cursor: "c1" }
          : { items: [limit({ id: 2 })], has_more: false },
    }));

    const limits = await api.spendLimits.list({ subject: "api_key", appId: "1234", apiKey: "mz7kq2xa9b" });

    assert.deepEqual(limits.map((one) => one.id), [1, 2]);
    const [first, second] = router.requestsTo("GET", "/v1/spend/limits");
    assert.equal(first?.query.get("subject"), "api_key");
    assert.equal(first?.query.get("app_id"), "1234");
    assert.equal(first?.query.get("api_key_id"), "mz7kq2xa9b");
    assert.equal(first?.query.get("limit"), "200");
    assert.equal(first?.query.has("cursor"), false);
    assert.equal(second?.query.get("cursor"), "c1");
  });

  it("gets, changes and deletes one limit by id", async () => {
    router.serve("GET", "/v1/spend/limits/7", { body: limit() });
    router.serve("PATCH", "/v1/spend/limits/7", { body: limit({ limit_micros: 25_000_000 }) });
    router.serve("DELETE", "/v1/spend/limits/7", { status: 204 });

    assert.equal((await api.spendLimits.get(7)).id, 7);
    const raised = await api.spendLimits.setAmount(7, USD.dollars(25));
    await api.spendLimits.setNote(7, "");
    await api.spendLimits.delete(7);

    assert.ok(USD.fromMicros(raised.limit_micros).equals(USD.dollars(25)));
    const [amount, note] = router.requestsTo("PATCH", "/v1/spend/limits/7");
    assert.deepEqual(amount?.body, { limit_micros: 25_000_000 });
    assert.deepEqual(note?.body, { note: "" }, "an empty note clears it");
    assert.equal(router.requestsTo("DELETE", "/v1/spend/limits/7").length, 1);
  });

  it("resets a subject's windows with a reason", async () => {
    router.serve("POST", "/v1/spend/limits/usage/reset", { body: usage() });

    await api.spendLimits.reset({
      subject: "user",
      appId: "1234",
      userId: "user-42",
      reason: "refunded a duplicate run",
      intervals: ["weekly"],
    });

    assert.deepEqual(router.last.body, {
      subject: "user",
      reason: "refunded a duplicate run",
      app_id: "1234",
      user_id: "user-42",
      intervals: ["weekly"],
    });
    await assert.rejects(() => api.spendLimits.reset({ subject: "user", userId: "user-42", reason: "" }), /reason/);
  });

  it("reads one subject's usage, and grants past its limits", async () => {
    router.serve("GET", "/v1/spend/limits/usage", { body: [usage({ subject: "team", team: "finance" })] });
    router.serve("POST", "/v1/spend/limits/grants", { status: 201, body: { id: 3 } });

    const read = await api.spendLimits.usage({ subject: "team", team: "finance" });
    await api.spendLimits.grant({
      subject: "user",
      userId: "user-42",
      appId: "1234",
      amount: USD.dollars(10),
      reason: "a demo",
      expiresAt: new Date("2026-11-10T00:00:00Z"),
    });

    assert.equal(read.team, "finance");
    const asked = router.requestsTo("GET", "/v1/spend/limits/usage")[0];
    assert.equal(asked?.query.get("subject"), "team");
    assert.deepEqual(asked?.query.getAll("team"), ["finance"]);
    assert.deepEqual(router.last.body, {
      subject: "user",
      amount_micros: 10_000_000,
      reason: "a demo",
      app_id: "1234",
      user_id: "user-42",
      expires_at: "2026-11-10T00:00:00.000Z",
    });
  });
});

describe("teams", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("lists the teams, within one app when asked", async () => {
    router.serve("GET", "/v1/teams", { body: [{ name: "finance", apps: [], limits: [], spent: [] }] });

    const teams = await api.teams.list({ appId: "1234" });

    assert.equal(teams[0]?.name, "finance");
    assert.equal(router.last.query.get("app_id"), "1234");
  });

  it("creates the team's limit when it has none", async () => {
    router.serve("GET", "/v1/spend/limits", { body: { items: [], has_more: false } });
    router.serve("POST", "/v1/spend/limits", { status: 201, body: limit({ subject: "team", team: "finance" }) });

    await api.teams.setLimit("finance", { interval: "monthly", amount: USD.dollars(8_000) });

    const looked = router.requestsTo("GET", "/v1/spend/limits")[0];
    assert.equal(looked?.query.get("subject"), "team");
    assert.equal(looked?.query.get("team"), "finance");
    assert.equal(looked?.query.get("interval"), "monthly");
    assert.deepEqual(router.last.body, {
      subject: "team",
      interval: "monthly",
      limit_micros: 8_000_000_000,
      team: "finance",
    });
  });

  it("changes the one it has, and not another app's", async () => {
    router.serve("GET", "/v1/spend/limits", {
      body: {
        items: [
          limit({ id: 1, subject: "app_team", team: "finance", app_id: "other", interval: "weekly" }),
          limit({ id: 2, subject: "app_team", team: "finance", app_id: "1234", interval: "weekly" }),
        ],
        has_more: false,
      },
    });
    router.serve("PATCH", "/v1/spend/limits/2", { body: limit({ id: 2 }) });

    await api.teams.setLimit("finance", { interval: "weekly", amount: "2500", appId: "1234", note: "Q4" });

    assert.equal(router.requestsTo("GET", "/v1/spend/limits")[0]?.query.get("subject"), "app_team");
    assert.deepEqual(router.last.body, { limit_micros: 2_500_000_000, note: "Q4" });
  });

  it("changes the one somebody else created since it looked", async () => {
    router.serve("GET", "/v1/spend/limits", (_, calls) => ({
      body: {
        items: calls === 0 ? [] : [limit({ id: 9, subject: "team", team: "finance" })],
        has_more: false,
      },
    }));
    router.serve("POST", "/v1/spend/limits", {
      status: 409,
      body: { error: { message: "exists", type: "conflict", code: "limit_exists" } },
    });
    router.serve("PATCH", "/v1/spend/limits/9", { body: limit({ id: 9 }) });

    const set = await api.teams.setLimit("finance", { interval: "monthly", amount: "10" });

    assert.equal(set.id, 9);
  });

  it("removes the team's limit, and does nothing when it has none", async () => {
    router.serve("GET", "/v1/spend/limits", (_, calls) => ({
      body: {
        items: calls === 0 ? [limit({ id: 4, subject: "team", team: "finance", interval: "weekly" })] : [],
        has_more: false,
      },
    }));
    router.serve("DELETE", "/v1/spend/limits/4", { status: 204 });

    await api.teams.removeLimit("finance", { interval: "weekly" });
    await api.teams.removeLimit("finance", { interval: "weekly" });

    assert.equal(router.requestsTo("DELETE", "/v1/spend/limits/4").length, 1);
  });

  it("refuses a team name the router would", async () => {
    await assert.rejects(() => api.teams.setLimit("fin:ance", { interval: "daily", amount: "1" }), ConfigurationError);
    await assert.rejects(() => api.teams.removeLimit("", { interval: "daily" }), ConfigurationError);
    assert.equal(router.received.length, 0);
  });

  it("passes on a refusal that is not a race", async () => {
    router.serve("GET", "/v1/spend/limits", { body: { items: [], has_more: false } });
    router.serve("POST", "/v1/spend/limits", {
      status: 400,
      body: { error: { message: "above", type: "invalid_request", code: "limit_above_parent" } },
    });

    await assert.rejects(
      () => api.teams.setLimit("finance", { interval: "daily", amount: "1" }),
      (error: unknown) => error instanceof RouterError && error.code === "limit_above_parent",
    );
  });
});

describe("a team on the client", () => {
  let router: TestRouter;

  beforeEach(async () => {
    router = await TestRouter.start();
  });

  afterEach(async () => {
    await router.stop();
  });

  it("is signed into the tokens it mints and sent as the header a router without the proxy reads", async () => {
    const api = new Client({
      url: router.url,
      apiKey: "vak_live_x",
      apiSecret: "shh",
      team: "finance",
      usageCategories: ["project:b", "project:a", "project:b"],
    });

    const headers = await api.backend.headers();

    assert.equal(headers["X-Stream-Usage-Categories"], "project:a,project:b,team:finance");
    assert.deepEqual(api.backend.usageCategories, ["project:a", "project:b"]);
    const claims = claimsOf(headers["Authorization"]?.replace(/^Bearer /, "") ?? "");
    assert.equal(claims["server"], true);
    assert.equal(claims["team"], "finance");
    assert.deepEqual(claims["usage_categories"], ["project:a", "project:b"]);
  });

  it("is on the token alone behind the proxy, which deletes the header", async () => {
    const api = new Client({
      url: router.url,
      authenticate: true,
      apiKey: "vak_live_x",
      apiSecret: "shh",
      userId: "user-42",
      team: "finance",
    });

    const headers = await api.backend.headers();

    assert.equal(headers["X-Stream-Usage-Categories"], undefined);
    const claims = claimsOf(headers["Authorization"]?.replace(/^Bearer /, "") ?? "");
    assert.equal(claims["user_id"], "user-42");
    assert.equal(claims["team"], "finance");
    assert.equal(claims["usage_categories"], undefined);
  });

  it("is sent as the header to a router reached by customer id", async () => {
    router.serve("GET", "/v1/teams", { body: [] });
    const api = new Client({ url: router.url, customerId: "local", team: "finance" });

    await api.teams.list();

    assert.equal(router.last.headers["x-stream-usage-categories"], "team:finance");
    assert.equal(router.last.headers["x-customer-id"], "local");
  });

  it("sends no header when there is no team", async () => {
    const api = new Client({ url: router.url, customerId: "local" });

    assert.deepEqual(await api.backend.headers(), { "X-Customer-Id": "local" });
  });

  it("refuses a team or categories beside a token, which carries its own", async () => {
    assert.throws(
      () => new Client({ url: router.url, apiKey: "vak_live_x", token: "t", team: "finance" }),
      /createUserToken/,
    );
    assert.throws(
      () => new Client({ url: router.url, apiKey: "vak_live_x", token: "t", usageCategories: ["a"] }),
      ConfigurationError,
    );
    const page = new Client({ url: router.url, apiKey: "vak_live_x", team: "finance" });
    await assert.rejects(() => page.setUser("user-42", "t"), /createUserToken/);
  });

  it("refuses a team or category the router would", () => {
    for (const options of [
      { team: "fin:ance" },
      { team: "x".repeat(60) },
      { usageCategories: ["team:finance"] },
      { usageCategories: ["TEAM:finance"] },
      { usageCategories: ["has space"] },
      { usageCategories: ["x".repeat(65)] },
      { usageCategories: "project:a" as unknown as string[] },
      { team: "finance", usageCategories: Array.from({ length: 10 }, (_, i) => `c${i}`) },
    ]) {
      assert.throws(() => new Client({ url: router.url, customerId: "local", ...options }), ConfigurationError);
    }
    assert.doesNotThrow(
      () => new Client({ url: router.url, customerId: "local", usageCategories: Array.from({ length: 10 }, (_, i) => `c${i}`) }),
    );
  });

  it("mints a user token naming the team and categories", async () => {
    const api = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh" });

    const token = await api.createUserToken("user-42", {
      team: "finance",
      usageCategories: ["project:b", "project:a"],
      validitySeconds: 60,
    });

    const claims = claimsOf(token);
    assert.equal(claims["user_id"], "user-42");
    assert.equal(claims["team"], "finance");
    assert.deepEqual(claims["usage_categories"], ["project:a", "project:b"]);
    assert.equal((claims["exp"] as number) - (claims["iat"] as number), 60);
    assert.deepEqual(Object.keys(claimsOf(await api.createUserToken("user-42"))).sort(), ["exp", "iat", "user_id"]);
  });

  it("mints no user token without the secret, nor for a team the router would refuse", async () => {
    const page = new Client({ url: router.url, apiKey: "vak_live_x", token: "t" });
    const server = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh" });

    await assert.rejects(() => page.createUserToken("user-42"), /secret/);
    await assert.rejects(() => server.createUserToken(""), ConfigurationError);
    await assert.rejects(() => server.createUserToken("user-42", { team: "a b" }), ConfigurationError);
    await assert.rejects(() => server.createUserToken("user-42", { usageCategories: ["team:x"] }), ConfigurationError);
  });
});

describe("a team on the session", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "session-1",
        call_id: "",
        call_type: "",
        user_id: "jlahey",
        agent_id: "docs",
        state: "live",
        modality: "text",
        created_at: new Date().toISOString(),
      },
    });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("is sent in the request when set, and only then", async () => {
    for (const options of [{ team: "finance" }, {}, { team: "" }]) {
      const opening = api.agent("docs").sessions.create(options);
      await router.socket();
      await (await opening).close();
    }

    const bodies = router.requestsTo("POST", "/v1/agents/sessions").map((one) => one.body as Record<string, unknown>);
    assert.equal(bodies[0]?.["team"], "finance");
    assert.ok(bodies.slice(1).every((body) => !("team" in body)));
    assert.equal(bodies.length, 3);
  });

  it("refuses a team the router would", async () => {
    await assert.rejects(() => api.agent("docs").sessions.create({ team: "fin ance" }), ConfigurationError);
  });
});
