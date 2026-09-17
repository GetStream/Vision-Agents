import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, RouterError } from "../src/index.js";
import { TestRouter } from "./router.js";

describe("Client", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("returns what the router answered with", async () => {
    router.serve("GET", "/v1/agents/configs", {
      body: [{ id: "cfg_1", name: "john", created_at: "2026-01-01T00:00:00Z" }],
    });

    const configs = await api.get("/v1/agents/configs");

    assert.equal(configs.length, 1);
    assert.equal(configs[0]?.name, "john");
  });

  it("puts path parameters into the path rather than sending the template", async () => {
    router.serve("GET", "/v1/agents/sessions/sess_1", { body: { id: "sess_1" } });

    await api.get("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });

    assert.equal(router.last.path, "/v1/agents/sessions/sess_1");
  });

  it("escapes a path parameter, so one cannot reach a path of its own", async () => {
    router.serve("GET", "/v1/agents/sessions/..%2Fconfigs", { body: { id: "x" } });

    await api.get("/v1/agents/sessions/{id}", { path: { id: "../configs" } });

    assert.equal(router.last.path, "/v1/agents/sessions/..%2Fconfigs");
  });

  it("refuses to send a template with a parameter missing", async () => {
    await assert.rejects(
      () => api.get("/v1/agents/sessions/{id}", { path: {} as { id: string } }),
      (raised: RouterError) => {
        assert.match(raised.message, /needs a id/);
        return true;
      },
    );
    assert.equal(router.received.length, 0, "nothing was sent");
  });

  it("carries the credentials and the body on a write", async () => {
    router.serve("POST", "/v1/agents/sessions", { status: 201, body: { id: "sess_1" } });

    await api.post("/v1/agents/sessions", { body: { call_id: "demo", text: false } });

    assert.equal(router.last.headers["x-customer-id"], "local");
    assert.equal(router.last.headers["content-type"], "application/json");
    assert.deepEqual(router.last.body, { call_id: "demo", text: false });
  });

  it("writes a query parameter once per value, so a list arrives as a list", async () => {
    router.serve("GET", "/v1/agents/logs", { body: { logs: [], next: "" } });

    await api.get("/v1/agents/logs", { query: { severity: "error", limit: 10 } });

    assert.equal(router.last.query.get("severity"), "error");
    assert.equal(router.last.query.get("limit"), "10");
  });

  it("leaves out a query parameter that was not set", async () => {
    router.serve("GET", "/v1/agents/calls", { body: [] });

    await api.get("/v1/agents/calls", { query: { limit: undefined } });

    assert.equal(router.last.query.has("limit"), false);
  });

  it("answers a 204 with nothing rather than failing to parse it", async () => {
    router.serve("DELETE", "/v1/agents/sessions/sess_1", { status: 204 });

    const answer = await api.delete("/v1/agents/sessions/{id}", { path: { id: "sess_1" } });

    assert.equal(answer, undefined);
  });

  it("raises what the router said went wrong, with the status and the operation", async () => {
    router.serve("POST", "/v1/agents/sessions", {
      status: 400,
      body: { error: "a call id is required unless the session is text" },
    });

    await assert.rejects(
      () => api.post("/v1/agents/sessions", { body: {} }),
      (raised: RouterError) => {
        assert.equal(raised.status, 400);
        assert.equal(raised.operation, "POST /v1/agents/sessions");
        assert.equal(raised.message, "a call id is required unless the session is text");
        return true;
      },
    );
  });

  it("falls back to the status when the failure is not the spec's shape", async () => {
    router.serve("GET", "/v1/agents/calls", { status: 502, body: "<html>bad gateway</html>" });

    await assert.rejects(
      () => api.get("/v1/agents/calls"),
      (raised: RouterError) => {
        assert.equal(raised.status, 502);
        assert.match(raised.message, /bad gateway/);
        return true;
      },
    );
  });

  it("reports a router it could not reach as no status at all", async () => {
    const unreachable = new Client({ url: "http://127.0.0.1:1", customerId: "local" });

    await assert.rejects(
      () => unreachable.get("/health"),
      (raised: RouterError) => {
        assert.equal(raised.status, 0);
        assert.match(raised.message, /could not reach the router/);
        return true;
      },
    );
  });

  it("stops a request when the caller aborts it", async () => {
    const stop = new AbortController();
    stop.abort();

    await assert.rejects(() => api.get("/health", { signal: stop.signal }));
  });
});
