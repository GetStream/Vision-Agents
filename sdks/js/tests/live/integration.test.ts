import assert from "node:assert/strict";
import { subscribe } from "node:diagnostics_channel";
import type { ClientRequest } from "node:http";
import { after, before, describe, it } from "node:test";

import { StreamChat } from "stream-chat";

import "../../src/chat.js";
import "../../src/video.js";
import { Client, Dispatch, Session, signToken } from "../../src/index.js";
import { close, conversationModel, eventually, uniqueId, unreachable } from "./target.js";

/**
 * The SDK as a device holds it, against a real router and a real Stream app.
 *
 * Two promises a client SDK makes that no unit test can keep for it. A device holds a token
 * naming one person, so nothing it can send mints a token or reaches an operation that is
 * the app's backend's to make. And an app already running Stream Chat and Video beside this
 * SDK still has one connection to each as its user, however many conversations are open:
 * a second one is a second online presence, and a second device registration is a second
 * copy of every push notification.
 *
 * The first needs a router that verifies credentials, which the local one in proxy mode
 * does not, so it runs against a router in `api_key` mode and a key minted for it with
 * `router keys create`. The second needs Stream itself.
 */
const keyedURL = process.env["INTEGRATION_ROUTER_URL"] ?? "http://localhost:8099";
const routerKey = process.env["INTEGRATION_ROUTER_KEY"] ?? "";
const routerSecret = process.env["INTEGRATION_ROUTER_SECRET"] ?? "";

const localURL = process.env["STREAM_ACCELERATION_URL"] ?? "http://localhost:8080";
const customerId = process.env["STREAM_ACCELERATION_CUSTOMER_ID"] ?? "";
const streamKey = process.env["STREAM_API_KEY"] ?? "";
const streamSecret = process.env["STREAM_API_SECRET"] ?? "";
const pushProvider = process.env["INTEGRATION_PUSH_PROVIDER"] ?? "firebase";

const CHAT_SOCKET = "chat.stream-io-api.com/connect";
const VIDEO_SOCKET = "video.stream-io-api.com/video/connect";

/**
 * Every WebSocket this process opened, as host and path.
 *
 * Read off Node's own diagnostics channels rather than off the clients, so what is counted
 * is the connections that actually left the process: `stream-chat` opens its socket with
 * `ws` over `http`, and the video client with the global WebSocket.
 */
const sockets: string[] = [];
subscribe("http.client.request.start", (message) => {
  const { request } = message as { request: ClientRequest };
  if (request.getHeader("upgrade") === "websocket") {
    sockets.push(`${request.host}${request.path.split("?")[0]}`);
  }
});
subscribe("undici:websocket:open", (message) => {
  const { websocket } = message as { websocket: { url: string } };
  const url = new URL(websocket.url);
  sockets.push(`${url.host}${url.pathname}`);
});

/** How many sockets to `target` were opened after `since` were. */
function openedSince(since: number, target: string): number {
  return sockets.slice(since).filter((socket) => socket === target).length;
}

describe(
  "a device against a router that verifies credentials",
  {
    skip:
      routerKey && routerSecret
        ? await unreachable(keyedURL)
        : "INTEGRATION_ROUTER_KEY and INTEGRATION_ROUTER_SECRET are not set",
  },
  () => {
    const backend = new Client({ url: keyedURL, apiKey: routerKey, apiSecret: routerSecret });
    let device: Client;

    before(async () => {
      device = new Client({ url: keyedURL, apiKey: routerKey });
      const id = uniqueId("device");
      await device.setUser({ id }, await signToken({ user_id: id }, routerSecret));
    });

    it("is not the backend, whatever secret is lying about in the environment", () => {
      assert.ok(process.env["STREAM_API_SECRET"], "the suite reads .env, which holds one");
      assert.equal(device.backend.serverSide, false);
      assert.equal(backend.backend.serverSide, true);
    });

    const serverOnly = [
      ["POST", "/v1/agents/calls/default:anything/token", "mint a call token"],
      ["POST", "/v1/agents/chat-token", "mint a chat token"],
      ["POST", "/v1/agents/guests/claim", "claim a guest's conversations"],
      ["GET", "/v1/agents/configs", "read the agent configs"],
      ["POST", "/v1/agents/configs", "write an agent config"],
      ["POST", "/v1/agents/sync", "sync agents from a folder"],
      ["POST", "/v1/agents/simulations", "start a simulation"],
      ["DELETE", "/v1/agents/users/anyone/memories", "forget everything about a user"],
    ] as const;

    for (const [method, path, what] of serverOnly) {
      it(`may not ${what}`, async () => {
        const response = await fetch(`${keyedURL}${path}`, {
          method,
          headers: { ...(await device.backend.headers()), "Content-Type": "application/json" },
          ...(method === "POST" ? { body: "{}" } : {}),
        });
        const body = (await response.json()) as { error?: { code?: string } };

        assert.equal(response.status, 403, `${method} ${path} answered ${response.status}`);
        assert.equal(body.error?.code, "server_side_only");
      });
    }

    it("may not promote itself by saying it is a server", async () => {
      const response = await fetch(`${keyedURL}/v1/agents/chat-token`, {
        method: "POST",
        headers: {
          ...(await device.backend.headers()),
          "Content-Type": "application/json",
          "Stream-Auth-Type": "server",
        },
        body: "{}",
      });

      assert.equal(response.status, 403);
    });

    it("is refused what its backend is allowed, so the refusal is about who is asking", async () => {
      const configs = await backend.get("/v1/agents/configs");

      assert.ok(Array.isArray(configs));
      await assert.rejects(() => device.get("/v1/agents/configs"), { status: 403 });
    });

    it("refuses to claim a guest before asking the router", () => {
      assert.throws(() => device.claimGuestUser("guest-whoever", "someone"), /server side only/);
    });

    it("may not open the dispatch socket that is offered other people's callers", async () => {
      const dispatch = new Dispatch({ client: device }).onCall(() => {});

      await assert.rejects(() => dispatch.run(AbortSignal.timeout(10_000)));
    });

    it("may still mint a guest, so a visitor can ask before signing in", async () => {
      const guest = await device.guestUser({ name: "Integration Guest" });

      assert.ok(guest.id);
      assert.ok(guest.token);
    });
  },
);

describe(
  "the SDK beside the app's own Stream clients",
  {
    skip:
      streamKey && streamSecret && customerId
        ? await unreachable(localURL)
        : "STREAM_API_KEY, STREAM_API_SECRET and STREAM_ACCELERATION_CUSTOMER_ID are not set",
  },
  () => {
    const stream = new StreamChat(streamKey, streamSecret);
    const users: string[] = [];
    const opened: { api: Client; id: string }[] = [];
    let model = "";

    /** A client as a page builds it: the router by customer id, Stream as the user. */
    async function asUser(): Promise<{ api: Client; id: string }> {
      const id = uniqueId("person");
      users.push(id);
      const api = new Client({ url: localURL, customerId, apiKey: streamKey });
      await api.setUser({ id }, await signToken({ user_id: id }, streamSecret));
      return { api, id };
    }

    async function open(api: Client): Promise<Session> {
      const session = await Session.open(api, { agent_id: uniqueId("conv"), llm: model });
      opened.push({ api, id: session.id });
      return session;
    }

    async function online(id: string): Promise<boolean> {
      const { users: found } = await stream.queryUsers({ id }, {}, {});
      return found[0]?.online === true;
    }

    before(async () => {
      model = await conversationModel(new Client({ url: localURL, customerId }));
    });

    after(async () => {
      for (const { api, id } of opened) {
        await close(api, id);
      }
      if (users.length > 0) {
        await stream.deleteUsers(users, { user: "hard" }).catch(() => undefined);
      }
    });

    it("connects to Stream Chat once, however many sessions ask", async () => {
      const { api, id } = await asUser();
      const since = sockets.length;

      const sessions = await Promise.all([open(api), open(api), open(api)]);
      const chats = await Promise.all(sessions.map((session) => session.chat()));

      assert.equal(openedSince(since, CHAT_SOCKET), 1);
      assert.ok(chats.every((each) => each.client === chats[0]?.client));
      assert.equal(await online(id), true);

      await api.disconnect();
      assert.equal(
        await eventually(() => online(id), (is) => !is),
        false,
        "the user is still online after the client disconnected",
      );
    });

    it("connects to Stream Video once, however many times it is asked", async () => {
      const { api } = await asUser();
      const since = sockets.length;

      const clients = await Promise.all([api.peers.video(), api.peers.video(), api.peers.video()]);
      await eventually(
        async () => openedSince(since, VIDEO_SOCKET),
        (count) => count > 0,
      );

      assert.equal(openedSince(since, VIDEO_SOCKET), 1);
      assert.ok(clients.every((each) => each === clients[0]));
      await api.disconnect();
    });

    it("opens no connection of its own when the app hands over its StreamChat", async (t) => {
      const { api, id } = await asUser();
      const app = new StreamChat(streamKey);
      const since = sockets.length;
      await app.connectUser({ id }, await signToken({ user_id: id }, streamSecret));
      api.use({ chat: app });

      const sessions = await Promise.all([open(api), open(api)]);
      const chats = await Promise.all(sessions.map((session) => session.chat()));

      assert.equal(openedSince(since, CHAT_SOCKET), 1, "only the app's own socket");
      assert.ok(chats.every((each) => each.client === app));

      await api.disconnect();
      assert.equal(app.userID, id, "disconnecting the SDK disconnected the app's client");
      assert.equal(await online(id), true);
      await app.disconnectUser();

      t.diagnostic(`checked ${sessions.length} sessions on one socket`);
    });

    it("leaves the app with the one push device it registered", async (t) => {
      const { api, id } = await asUser();
      const app = new StreamChat(streamKey);
      await app.connectUser({ id }, await signToken({ user_id: id }, streamSecret));
      const registered = await app
        .addDevice(uniqueId("device"), "firebase", undefined, pushProvider)
        .then(() => true)
        .catch((raised: Error) => {
          t.skip(`this Stream app cannot register a ${pushProvider} device: ${raised.message}`);
          return false;
        });
      if (!registered) {
        await app.disconnectUser();
        return;
      }
      api.use({ chat: app });

      const sessions = await Promise.all([open(api), open(api)]);
      await Promise.all(sessions.map((session) => session.chat()));
      const { devices } = await stream.getDevices(id);

      assert.equal(devices?.length, 1, "a second device is a second copy of every push");
      await app.disconnectUser();
    });
  },
);
