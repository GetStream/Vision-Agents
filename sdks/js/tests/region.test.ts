import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, ConfigurationError, DEFAULT_URL, type BackendOptions } from "../src/index.js";
import { TestRouter, type Received } from "./router.js";

const REGIONAL_HOST = "accelerate-us-east4.gcp.stream-io-api.com";
const VARIABLES = [
  "STREAM_ACCELERATION_URL",
  "STREAM_ACCELERATION_REGION",
  "STREAM_ACCELERATION_CUSTOMER_ID",
  "STREAM_ACCELERATION_AUTHENTICATE",
  "STREAM_API_KEY",
  "STREAM_API_SECRET",
];

describe("Regional clients", () => {
  let router: TestRouter;
  let saved: Record<string, string | undefined>;
  /** The URLs the SDK asked for, before they were sent to the test router instead. */
  let requested: URL[];

  beforeEach(async () => {
    saved = Object.fromEntries(VARIABLES.map((name) => [name, process.env[name]]));
    for (const name of VARIABLES) {
      delete process.env[name];
    }
    process.env["STREAM_API_KEY"] = "test-key";
    process.env["STREAM_API_SECRET"] = "test-secret";

    router = await TestRouter.start();
    router.serve("GET", "/v1/agents/configs", (request) =>
      authenticated(request)
        ? { body: [{ id: "cfg_1", name: "john", created_at: "2026-01-01T00:00:00Z" }] }
        : { status: 401, body: { error: "expected an authenticated regional request" } },
    );
    requested = [];
  });

  afterEach(async () => {
    await router.stop();
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  });

  /** Keeps the regional URL the SDK built, but connects only to the test router. */
  function client(options: BackendOptions = {}): Client {
    const redirect: typeof fetch = (input, init) => {
      const original = new URL(input instanceof Request ? input.url : input.toString());
      requested.push(original);
      const local = new URL(original.pathname + original.search, router.url);
      return fetch(local, init);
    };
    return new Client({ ...options, fetch: redirect });
  }

  async function listConfigs(api: Client): Promise<void> {
    const configs = await api.get("/v1/agents/configs");

    assert.equal(configs[0]?.name, "john");
    assert.equal(requested.at(-1)?.host, REGIONAL_HOST);
    assert.equal(requested.at(-1)?.protocol, "https:");
  }

  it("lists configs from the regional host with the default client", async () => {
    await listConfigs(client());
  });

  it("lists configs from the regional host with an explicit region", async () => {
    await listConfigs(client({ region: "us-east4" }));
  });

  it("lists configs from the regional host with a region from the environment", async () => {
    process.env["STREAM_ACCELERATION_REGION"] = "us-east4";

    await listConfigs(client());
  });

  it("refuses a conflicting url and region", () => {
    assert.throws(() => new Client({ region: "europe-west4", url: DEFAULT_URL }), ConfigurationError);
  });

  it("refuses an invalid region with a url", () => {
    assert.throws(
      () => new Client({ region: "us-east4.evil.com/x", url: DEFAULT_URL }),
      ConfigurationError,
    );
  });

  it("lets a regional client act for a user", async () => {
    const api = client({ region: "us-east4" });

    await api.setUser("jean", "user-token");

    assert.equal(api.backend.url, DEFAULT_URL);
    assert.equal(api.backend.userId, "jean");
    await listConfigs(api);
    assert.equal(router.last.headers["authorization"], "Bearer user-token");
  });
});

function authenticated(request: Received): boolean {
  return (
    request.headers["api_key"] === "test-key" &&
    request.headers["stream-auth-type"] === "jwt" &&
    Boolean(request.headers["authorization"])
  );
}
