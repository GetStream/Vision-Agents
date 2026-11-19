import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  Client,
  ConfigurationError,
  RouterError,
  type BackendOptions,
  type Connection,
  type Schemas,
} from "../src/index.js";
import { TestRouter, claimsOf, type Received, type Reply } from "./router.js";

/** The stand-in's page size, below what `list` asks for, so a longer list shows the cursor is followed. */
const STAND_IN_PAGE = 2;

/**
 * Answers the connection routes with the router's owner rule (acceleration/internal/api/connections.go,
 * `ownerOf` and `mayReach`): a user's connection is created and reached only by a backend naming that user in
 * `X-Stream-User-Id`, the app's by any backend whatever it names.
 */
class StandIn {
  /** The stored connections, oldest first. */
  readonly connections: Connection[] = [];

  constructor(private readonly router: TestRouter) {
    router.serve("POST", "/v1/agents/connections", (request) => this.create(request));
    router.serve("GET", "/v1/agents/connections", (request) => this.list(request));
  }

  private create(request: Received): Reply {
    const sent = request.body as Schemas["ConnectionRequest"];
    const acting = request.headers["x-stream-user-id"] ?? "";
    const named = sent.owner.user_id;
    if (sent.owner.type === "app" ? named !== undefined : !named || named !== acting) {
      return refused(400, "invalid_request", "owner.user_id must be the user this backend acts for");
    }
    const id = `connection-${this.connections.length + 1}`;
    const created = connection({
      id,
      connector_id: sent.connector_id,
      owner: sent.owner,
      ...(sent.label ? { label: sent.label } : {}),
    });
    this.connections.push(created);
    const path = `/v1/agents/connections/${id}`;
    this.router.serve("GET", path, (one) => this.reached(one, created) ?? { body: created });
    this.router.serve("DELETE", path, (one) => {
      const refusal = this.reached(one, created);
      if (refusal) {
        return refusal;
      }
      this.connections.splice(this.connections.indexOf(created), 1);
      return { status: 204 };
    });
    this.router.serve("POST", `${path}/authorizations`, (one) => this.reached(one, created) ?? {
      status: 201,
      body: {
        id: "attempt-1",
        kind: "consent",
        launch_url: `https://router.example.com${path}/launch`,
        handoff_token: "handoff-1",
        expires_at: "2026-11-19T12:10:00Z",
      } satisfies Schemas["Authorization"],
    });
    return { status: 201, body: created };
  }

  private list(request: Received): Reply {
    const ownerType = request.query.get("owner_type") ?? "app";
    const acting = request.headers["x-stream-user-id"] ?? "";
    const connector = request.query.get("connector_id");
    const matched = [...this.connections].reverse().filter(
      (one) =>
        one.owner.type === ownerType &&
        (ownerType === "app" || !acting || one.owner.user_id === acting) &&
        (!connector || one.connector_id === connector),
    );
    const from = Number((request.query.get("cursor") ?? "after-0").replace("after-", ""));
    const to = Math.min(from + STAND_IN_PAGE, matched.length);
    const page: Schemas["ConnectionPage"] = {
      items: matched.slice(from, to),
      has_more: to < matched.length,
      ...(to < matched.length ? { next_cursor: `after-${to}` } : {}),
    };
    return { body: page };
  }

  /** The refusal of a request for a connection it may not reach, or undefined when it may. */
  private reached(request: Received, connection: Connection): Reply | undefined {
    if (!this.connections.includes(connection)) {
      return refused(404, "not_found", "no such connection");
    }
    const acting = request.headers["x-stream-user-id"] ?? "";
    if (connection.owner.type === "app" || (acting && connection.owner.user_id === acting)) {
      return undefined;
    }
    return refused(404, "not_found", "no such connection");
  }
}

function refused(status: number, type: string, message: string): Reply {
  return { status, body: { error: { message, type, code: type, doc_url: "" } } };
}

/** A connection as the router describes one, with only the fields a test cares about set. */
function connection(over: Partial<Connection> & Pick<Connection, "id" | "connector_id" | "owner">): Connection {
  return {
    auth_scheme: "oauth2_code",
    created_at: "2026-11-19T00:00:00Z",
    updated_at: "2026-11-19T00:00:00Z",
    definition_revision: 1,
    definition_status: "current",
    granted_scopes: [],
    inputs: {},
    metadata: {},
    revision: 1,
    status: "pending",
    used_by: [],
    ...over,
  };
}

/** The headers fetch itself sends, which say nothing about who is calling. */
const TRANSPORT = new Set([
  "host",
  "connection",
  "accept",
  "accept-language",
  "accept-encoding",
  "sec-fetch-mode",
  "user-agent",
]);

/** The `X-Stream-User-Id` of every request the router got, in order; the empty string where none was sent. */
function usersSent(router: TestRouter): string[] {
  return router.received.map((one) => one.headers["x-stream-user-id"] ?? "");
}

describe("Connections", () => {
  let router: TestRouter;
  let api: Client;
  let standIn: StandIn;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh" });
    standIn = new StandIn(router);
  });

  afterEach(async () => {
    await router.stop();
  });

  it("names a user's connection as its owner and as the user acted for, so the router takes it", async () => {
    const created = await api.connections.create("linear", { label: "work", userId: "anya" });

    assert.deepEqual(router.last.body, {
      connector_id: "linear",
      label: "work",
      owner: { type: "user", user_id: "anya" },
    });
    assert.deepEqual(usersSent(router), ["anya"]);
    assert.equal(created.owner.user_id, "anya");
    assert.equal(created.status, "pending");
  });

  it("still speaks as the app's backend for a user, since a user's own token is refused here", async () => {
    for (const authenticate of [false, true]) {
      const backend = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh", authenticate });
      await backend.connections.get((await backend.connections.create("linear", { userId: "anya" })).id, {
        userId: "anya",
      });

      for (const request of router.received.splice(0)) {
        const claims = claimsOf((request.headers["authorization"] ?? "").replace(/^Bearer /, ""));
        assert.equal(claims["server"], true, `${request.method} ${request.path} authenticate=${authenticate}`);
        assert.equal(claims["user_id"], undefined, `${request.method} ${request.path} authenticate=${authenticate}`);
        assert.equal(request.headers["x-stream-user-id"], "anya");
      }
    }
  });

  it("sends the user on every call about their connection, reads included", async () => {
    const created = await api.connections.create("linear", { userId: "anya" });
    const consent = await api.connections.authorize(created.id, {
      returnUrl: "https://app.example.com/settings",
      userId: "anya",
    });
    const read = await api.connections.get(created.id, { userId: "anya" });
    const listed = await api.connections.list({ userId: "anya" });
    await api.connections.delete(created.id, { userId: "anya" });

    assert.deepEqual(usersSent(router), ["anya", "anya", "anya", "anya", "anya"]);
    assert.equal(consent.handoff_token, "handoff-1");
    assert.equal(read.id, created.id);
    assert.deepEqual(listed.map((one) => one.id), [created.id]);
    assert.equal(router.requestsTo("GET", "/v1/agents/connections")[0]?.query.get("owner_type"), "user");
    assert.deepEqual(router.requestsTo("POST", `/v1/agents/connections/${created.id}/authorizations`)[0]?.body, {
      return_url: "https://app.example.com/settings",
    });
    assert.deepEqual(standIn.connections, []);
  });

  it("names no user for the app's own connection, on any call", async () => {
    const created = await api.connections.create("github");
    await api.connections.authorize(created.id);
    await api.connections.get(created.id);
    const listed = await api.connections.list();
    await api.connections.delete(created.id);

    assert.deepEqual(router.requestsTo("POST", "/v1/agents/connections")[0]?.body, {
      connector_id: "github",
      owner: { type: "app" },
    });
    assert.deepEqual(usersSent(router), ["", "", "", "", ""]);
    assert.deepEqual(listed.map((one) => one.id), [created.id]);
    assert.equal(router.requestsTo("GET", "/v1/agents/connections")[0]?.query.get("owner_type"), "app");
    assert.equal(router.requestsTo("POST", `/v1/agents/connections/${created.id}/authorizations`)[0]?.body, undefined);
  });

  it("reads an empty user id as the app, rather than as a user nobody is", async () => {
    await api.connections.create("github", { userId: "" });
    await api.connections.list({ userId: "" });

    assert.deepEqual(router.requestsTo("POST", "/v1/agents/connections")[0]?.body, {
      connector_id: "github",
      owner: { type: "app" },
    });
    assert.equal(router.requestsTo("GET", "/v1/agents/connections")[0]?.query.get("owner_type"), "app");
    assert.deepEqual(usersSent(router), ["", ""]);
  });

  it("reaches the app's connection whoever it names, and only a user's own", async () => {
    const apps = await api.connections.create("github");
    const bobs = await api.connections.create("linear", { userId: "bob" });

    assert.equal((await api.connections.get(apps.id, { userId: "anya" })).id, apps.id);
    await assert.rejects(
      api.connections.get(bobs.id, { userId: "anya" }),
      (error: unknown) => error instanceof RouterError && error.status === 404,
    );
    await assert.rejects(
      api.connections.delete(bobs.id),
      (error: unknown) => error instanceof RouterError && error.status === 404,
    );
    assert.deepEqual(standIn.connections.map((one) => one.id), [apps.id, bobs.id]);
  });

  it("lists every page of one user's connections, newest first, and not the app's or another user's", async () => {
    const mine: string[] = [];
    for (let at = 0; at < 5; at += 1) {
      mine.push((await api.connections.create("linear", { label: `n${at}`, userId: "anya" })).id);
      await api.connections.create("linear", { userId: "bob" });
      await api.connections.create("linear");
    }

    const listed = await api.connections.list({ userId: "anya" });

    assert.deepEqual(listed.map((one) => one.id), mine.reverse());
    const pages = router.requestsTo("GET", "/v1/agents/connections");
    assert.deepEqual(pages.map((one) => one.query.get("cursor")), [null, "after-2", "after-4"]);
    assert.deepEqual(pages.map((one) => one.query.get("limit")), ["200", "200", "200"]);
  });

  it("stops at a page that says it is the last, or that gives no cursor to go on from", async () => {
    for (const last of [
      { items: null, has_more: false, next_cursor: "after-1" },
      { items: [], has_more: true },
    ] satisfies Schemas["ConnectionPage"][]) {
      router.serve("GET", "/v1/agents/connections", (_, calls) =>
        calls === 0
          ? {
              body: {
                items: [connection({ id: "c1", connector_id: "linear", owner: { type: "app" } })],
                has_more: true,
                next_cursor: "after-1",
              },
            }
          : { body: last },
      );

      assert.deepEqual((await api.connections.list()).map((one) => one.id), ["c1"], JSON.stringify(last));
      assert.equal(router.requestsTo("GET", "/v1/agents/connections").length, 2, JSON.stringify(last));
      router.received.splice(0);
    }
  });

  it("leaves a client built for a user sending that user when a call names none, as its other calls do", async () => {
    const jims = new Client({ url: router.url, apiKey: "vak_live_x", apiSecret: "shh", userId: "jim" });

    await jims.connections.list();
    await jims.connections.list({ userId: "anya" });

    assert.deepEqual(usersSent(router), ["jim", "anya"]);
  });

  it("keeps one connector's when asked, and asks for every connector's otherwise", async () => {
    await api.connections.create("linear", { userId: "anya" });
    const github = await api.connections.create("github", { userId: "anya" });

    assert.deepEqual((await api.connections.list({ connectorId: "github", userId: "anya" })).map((one) => one.id), [
      github.id,
    ]);
    assert.equal((await api.connections.list({ userId: "anya" })).length, 2);
    const pages = router.requestsTo("GET", "/v1/agents/connections");
    assert.deepEqual(pages.map((one) => one.query.get("connector_id")), ["github", null]);
  });

  it("leaves a label out of the request when none is given", async () => {
    await api.connections.create("linear", { label: "", userId: "anya" });

    assert.deepEqual(router.last.body, { connector_id: "linear", owner: { type: "user", user_id: "anya" } });
  });

  it("refuses a call naming no connection before asking the router", async () => {
    for (const call of [
      () => api.connections.get(""),
      () => api.connections.authorize("", { userId: "anya" }),
      () => api.connections.delete(""),
    ]) {
      await assert.rejects(call(), ConfigurationError);
    }
    assert.equal(router.received.length, 0);
  });

  it("is still reached through the deprecated actingFor, sending what it sent before userId", async () => {
    router.serve("GET", "/v1/agents/configs", { body: { items: [], has_more: false } });
    // Each expected request is what the base commit (fa2a6a66d) sent for the same shape, read off its test router.
    const shapes: [BackendOptions, Record<string, string>][] = [
      [{ apiKey: "k", apiSecret: "s" }, { "x-api-key": "k", "x-stream-user-id": "anya", "stream-auth-type": "server" }],
      [
        { apiKey: "k", apiSecret: "s", authenticate: true },
        { api_key: "k", "stream-auth-type": "jwt", "x-stream-user-id": "anya" },
      ],
      [{ customerId: "local" }, { "x-customer-id": "local", "x-stream-user-id": "anya" }],
      [
        { apiKey: "k", apiSecret: "s", userId: "jim" },
        { "x-api-key": "k", "x-stream-user-id": "anya", "stream-auth-type": "server" },
      ],
    ];
    for (const [shape, expected] of shapes) {
      await new Client(new Client({ url: router.url, ...shape }).backend.actingFor("anya")).get("/v1/agents/configs");

      const { authorization, ...sent } = router.last.headers;
      const kept = Object.fromEntries(Object.entries(sent).filter(([name]) => !TRANSPORT.has(name)));
      assert.deepEqual(kept, expected, JSON.stringify(shape));
      if (authorization) {
        const { iat: _iat, exp: _exp, ...claims } = claimsOf(authorization.replace(/^Bearer /, ""));
        assert.deepEqual(claims, { server: true }, JSON.stringify(shape));
      } else {
        assert.equal(shape.customerId, "local", "only a router reached by customer id is shown no token");
      }
      assert.equal(router.last.query.toString(), "");
    }
  });
});
