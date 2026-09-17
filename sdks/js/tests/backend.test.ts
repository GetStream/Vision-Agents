import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";

import { Backend, ConfigurationError, signToken } from "../src/index.js";
import { TestRouter, claimsOf } from "./router.js";

describe("Backend", () => {
  let router: TestRouter;

  before(async () => {
    router = await TestRouter.start();
  });

  after(async () => {
    await router.stop();
  });

  it("refuses a backend nobody is billed for", () => {
    assert.throws(() => new Backend({ url: "http://localhost:8080" }), ConfigurationError);
  });

  it("refuses a key with neither the secret it belongs to nor a token", () => {
    assert.throws(() => new Backend({ apiKey: "vak_live_x" }), ConfigurationError);
  });

  it("names the customer when that is all there is to go on", async () => {
    const backend = new Backend({ url: router.url, customerId: "local" });

    assert.deepEqual(await backend.headers(), { "X-Customer-Id": "local" });
    assert.equal(backend.serverSide, true);
  });

  it("presents a server token for a process the customer runs", async () => {
    const backend = new Backend({
      url: router.url,
      apiKey: "vak_live_x",
      apiSecret: "shh",
    });

    const headers = await backend.headers();
    assert.equal(headers["X-Api-Key"], "vak_live_x");
    assert.equal(headers["Stream-Auth-Type"], "server");
    assert.equal(backend.serverSide, true);

    const claims = claimsOf((headers["Authorization"] ?? "").replace("Bearer ", ""));
    assert.equal(claims["server"], true);
    assert.equal(claims["user_id"], undefined, "a server token names no user");
  });

  it("says which of its users a backend is acting for in a header, not the token", async () => {
    const backend = new Backend({
      url: router.url,
      apiKey: "vak_live_x",
      apiSecret: "shh",
      userId: "ana",
    });

    const headers = await backend.headers();
    assert.equal(headers["X-Stream-User-Id"], "ana");
    assert.equal(claimsOf((headers["Authorization"] ?? "").replace("Bearer ", ""))["server"], true);
  });

  it("presents a browser's token as a jwt rather than as a backend", async () => {
    const backend = new Backend({
      url: router.url,
      apiKey: "vak_live_x",
      token: await signToken({ user_id: "ana" }, "shh"),
    });

    const headers = await backend.headers();
    assert.equal(headers["Stream-Auth-Type"], "jwt");
    assert.equal(backend.serverSide, false, "a device may not reach a server-side operation");
    assert.equal(claimsOf((headers["Authorization"] ?? "").replace("Bearer ", ""))["user_id"], "ana");
  });

  it("spells the credential Stream's way for a router behind the proxy", async () => {
    const backend = new Backend({
      url: router.url,
      apiKey: "vak_live_x",
      apiSecret: "shh",
      authenticate: true,
    });

    const headers = await backend.headers();
    assert.equal(headers["api_key"], "vak_live_x", "the proxy reads no X-Api-Key");
    // Never `server`: the proxy classifies the caller itself and refuses being told.
    assert.equal(headers["stream-auth-type"], "jwt");
    assert.equal(headers["X-Api-Key"], undefined);

    const claims = claimsOf((headers["Authorization"] ?? "").replace("Bearer ", ""));
    assert.equal(claims["server"], true, "a backend still presents itself as one");
  });

  it("presents a proxied backend's user in the token, the only place the proxy looks", async () => {
    const backend = new Backend({
      url: router.url,
      apiKey: "vak_live_x",
      apiSecret: "shh",
      authenticate: true,
      userId: "u_42",
    });

    const headers = await backend.headers();
    const claims = claimsOf((headers["Authorization"] ?? "").replace("Bearer ", ""));
    assert.equal(claims["user_id"], "u_42");
    assert.equal(claims["server"], undefined, "a token for a user is not a server's");
  });

  it("refuses to authenticate with nothing to authenticate as", () => {
    assert.throws(
      () => new Backend({ url: router.url, customerId: "local", authenticate: true }),
      ConfigurationError,
    );
  });

  it("does not reach a local router with a credential because a key is in the environment", async () => {
    process.env["STREAM_API_KEY"] = "vak_live_ambient";
    try {
      const backend = new Backend({ url: router.url, customerId: "local" });

      assert.deepEqual(await backend.headers(), { "X-Customer-Id": "local" });
    } finally {
      delete process.env["STREAM_API_KEY"];
    }
  });

  it("does not turn a browser's client into a backend because a secret is in the environment", async () => {
    // A token handed in is the caller's answer to who they are. Without this, a client
    // built the way a page builds one becomes server-side in any process holding the
    // secret, which is every test and most servers.
    process.env["STREAM_API_SECRET"] = "shh";
    try {
      const backend = new Backend({
        url: router.url,
        apiKey: "vak_live_x",
        token: await signToken({ user_id: "u_1" }, "shh"),
      });

      assert.equal(backend.serverSide, false);
      assert.equal((await backend.headers())["Stream-Auth-Type"], "jwt");
    } finally {
      delete process.env["STREAM_API_SECRET"];
    }
  });

  it("asks a token callback again on every request, so an idle client never holds a stale one", async () => {
    let minted = 0;
    const backend = new Backend({
      url: router.url,
      apiKey: "vak_live_x",
      token: () => {
        minted += 1;
        return `token-${minted}`;
      },
    });

    assert.equal((await backend.headers())["Authorization"], "Bearer token-1");
    assert.equal((await backend.headers())["Authorization"], "Bearer token-2");
  });

  it("carries the credentials on a socket in the query, where a browser can put them", async () => {
    const backend = new Backend({ url: router.url, customerId: "local" });

    const url = new URL(await backend.socketURL("/v1/dispatch", { capacity: "2" }));
    assert.equal(url.protocol, "ws:");
    assert.equal(url.pathname, "/v1/dispatch");
    assert.equal(url.searchParams.get("customer_id"), "local");
    assert.equal(url.searchParams.get("capacity"), "2");
    assert.equal(
      url.searchParams.get("stream-auth-type"),
      null,
      "there is no query counterpart, so a browser socket cannot claim to be a backend",
    );
  });

  it("upgrades an https router to wss", async () => {
    const backend = new Backend({ url: "https://router.example.com", customerId: "local" });
    const url = await backend.socketURL("/v1/agents/sessions/x/events");

    assert.ok(url.startsWith("wss://router.example.com/v1/agents/sessions/x/events"));
  });

  it("signs a token the router can verify, with a life of its own", async () => {
    const token = await signToken({ user_id: "ana" }, "shh", 60);
    const claims = claimsOf(token);

    assert.equal(token.split(".").length, 3);
    assert.equal(claims["user_id"], "ana");
    assert.equal(Number(claims["exp"]) - Number(claims["iat"]), 60);
  });
});
