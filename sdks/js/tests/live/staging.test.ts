import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";

import WebSocketWithHeaders from "ws";

import {
  Client,
  RouterError,
  Session,
  signToken,
  type AgentHandle,
  type WebSocketConstructor,
} from "../../src/index.js";
import { close, conversationModel, exhausted, uniqueId, unreachable } from "./target.js";

/**
 * A WebSocket that carries credentials in headers, which the standard one cannot.
 *
 * Only a server can do this, and through the proxy only a server can open a socket at
 * all. `ws` is a dev dependency here rather than one the package ships, so this is the
 * test's own doing and not something a caller inherits.
 */
function authenticated(headers: Record<string, string>): WebSocketConstructor {
  return class extends WebSocketWithHeaders {
    constructor(url: string) {
      super(url, { headers });
    }
  } as unknown as WebSocketConstructor;
}

/**
 * The SDK against the deployments a change is shipped to.
 *
 * Two of them, because a page needs both: the hosted Accelerate, which holds the
 * conversations, and the hosted documentation corpus, which is a service of its own with
 * its own address and no credential. Separate suites rather than one, so the corpus is
 * still checked on a checkout that has no Stream credentials to reach Accelerate with.
 *
 * Unlike the local suite this one authenticates, which is the part worth covering here:
 * every deployment a browser talks to has a key and a token in front of it, and what a
 * token is allowed to reach is the whole of what keeps one person's conversation from
 * another's.
 */
const acceleration =
  process.env["STAGING_ACCELERATION_URL"] ?? "https://accelerate.gcp.stream-io-api.com";
const docs =
  process.env["STAGING_DOCS_SEARCH_URL"] ?? "https://search-production-98b1.up.railway.app";

const apiKey = process.env["STREAM_API_KEY"] ?? "";
const apiSecret = process.env["STREAM_API_SECRET"] ?? "";

describe(
  "staging acceleration",
  {
    skip:
      apiKey && apiSecret
        ? await unreachable(acceleration)
        : "STREAM_API_KEY and STREAM_API_SECRET are not set",
  },
  () => {
    /** What a process the customer runs holds: the secret, and every operation. */
    const backend = new Client({ url: acceleration, apiKey, apiSecret, authenticate: true });
    const opened: { api: Client; id: string }[] = [];
    let model = "";
    /** An agent this deployment has configured, which is what a name resolves against. */
    let agentName = "";

    /**
     * What a browser holds: the key, and a token naming one person.
     *
     * Built the way the requested shape builds it — a client against a URL and a key, with
     * the identity arriving afterwards — because that is the shape a page can actually
     * write: the token comes from the app's own backend, after the page has loaded.
     */
    async function asUser(userId: string): Promise<Client> {
      const page = new Client({ url: acceleration, apiKey, authenticate: true });
      await page.setUser({ id: userId }, await signToken({ user_id: userId }, apiSecret));
      return page;
    }

    /** The agent handle a client addresses this deployment's agent through. */
    function agentOf(api: Client): AgentHandle {
      return api.agent(agentName);
    }

    before(async () => {
      model = await conversationModel(backend);
      const configs = await backend.get("/v1/agents/configs").catch(() => []);
      agentName = configs.find((config) => config.name)?.name ?? "";
    });

    after(async () => {
      for (const { api, id } of opened) {
        await close(api, id);
      }
    });

    it("is healthy", async () => {
      const health = await backend.get("/health");

      assert.equal(health.status, "ok");
    });

    it("knows a backend from a browser by what it authenticated as", async () => {
      assert.equal(backend.backend.serverSide, true);
      assert.equal((await asUser("qa-reader")).backend.serverSide, false);
    });

    it("lets a backend read the agent configs it holds", async () => {
      const configs = await backend.get("/v1/agents/configs");

      assert.ok(Array.isArray(configs));
    });

    it(
      "refuses a browser the operations that configure an agent",
      {
        todo:
          "the proxy does not tell the router the caller is an end user, so the router " +
          "reads every proxied caller as that app's backend and allows this",
      },
      async () => {
        // The whole reason a page is given a token rather than the secret. A caller
        // holding a user's token may hold a conversation and may not rewrite the agent
        // holding it.
        const page = await asUser(uniqueId("refused"));

        await assert.rejects(
          () => page.get("/v1/agents/configs"),
          (raised: RouterError) => {
            assert.equal(raised.status, 403, `it answered ${raised.status}`);
            return true;
          },
        );
      },
    );

    it("mints a guest, so a visitor can ask before they have an account", async (t) => {
      // No store to remember them in on a server, which is the point: a process handling
      // two visitors would otherwise hand them each other's conversations.
      const guest = await backend.guestUser({ name: "QA Guest" }).catch((raised: RouterError) => {
        if (raised.status === 403) {
          t.skip("this app does not admit guests");
          return undefined;
        }
        throw raised;
      });
      if (!guest) {
        return;
      }

      assert.ok(guest.id, "a guest with no id cannot hold a conversation");
      assert.ok(guest.token, "a guest with no token cannot make a request");
    });

    it("refuses a page the claim that moves a guest's conversations", async () => {
      // The one operation that most has to be server side: only the backend that just
      // authenticated an account knows which guest it was.
      const page = await asUser(uniqueId("claimer"));

      assert.throws(
        () => page.claimGuestUser("guest-whoever", "jlahey"),
        /server side only/,
      );
    });

    it("opens a labelled conversation for an end user and names the channel itself", async (t) => {
      if (!agentName) {
        t.skip("this deployment has no agent configured to address by name");
        return;
      }
      const page = await asUser(uniqueId("opens"));
      const title = uniqueId("titled");

      const session = await agentOf(page).sessions.create({
        title,
        project: "qa",
        persist_conversation: true,
        llm: model,
      });
      opened.push({ api: page, id: session.id });

      assert.equal(session.created.text, true);
      assert.equal(session.created.state, "live");
      assert.equal(session.created.title, title);
      assert.match(session.conversationId, /^agent:support-[0-9a-f-]{36}$/);

      const found = await agentOf(page).sessions.search(title, { limit: 50 });
      assert.ok(
        found.some((each) => each.id === session.id),
        "a conversation cannot be found by the title it was given",
      );
    });

    it("does not show one person's conversation to another", {
      todo:
        "same cause: read as a backend, a proxied caller reaches every session its app " +
        "has, so neither the list nor the id is scoped to whose token was presented",
    }, async (t) => {
      if (!agentName) {
        t.skip("this deployment has no agent configured to address by name");
        return;
      }
      // A token names who is calling, and it is what stops a list being a way to read
      // somebody else's conversation. Nothing but the token differs between these two.
      const mine = await asUser(uniqueId("mine"));
      const theirs = await asUser(uniqueId("theirs"));

      const session = await agentOf(mine).sessions.create({ llm: model });
      opened.push({ api: mine, id: session.id });

      const listed = await theirs.get("/v1/agents/sessions");
      assert.ok(
        !listed.some((each) => each.id === session.id),
        "somebody else's session was listed",
      );

      // Reported as missing rather than refused, so this is not a way to find out whose
      // an id is.
      await assert.rejects(
        () => theirs.get("/v1/agents/sessions/{id}", { path: { id: session.id } }),
        (raised: RouterError) => {
          assert.equal(raised.status, 404, `it answered ${raised.status}`);
          return true;
        },
      );
    });

    it("refuses a socket carrying its credentials the only way a browser can", {
      todo:
        "the proxy requires the stream-auth-type header, which has no query parameter " +
        "counterpart and which a browser WebSocket cannot set, so a page cannot watch a " +
        "session on a hosted deployment at all",
    }, async () => {
      // What Backend.socketURL builds: credentials in the query string, because that is
      // all a browser has.
      const page = await asUser(uniqueId("browser"));
      const session = await Session.open(page, {
        text: true,
        agent_id: uniqueId("conv"),
        llm: model,
      });
      opened.push({ api: page, id: session.id });
      await session.close();
    });

    it("answers over the session socket when the credentials can go in headers", async (t) => {
      // A server can send them, which is what makes this reachable at all today. The SDK
      // takes the WebSocket to use, so supplying one is how a Node caller does it without
      // the package growing a dependency for everybody.
      const page = new Client({
        url: acceleration,
        apiKey,
        apiSecret,
        authenticate: true,
        webSocket: authenticated(await backend.backend.headers()),
      });
      const session = await Session.open(page, {
        text: true,
        persist_conversation: true,
        agent_id: uniqueId("conv"),
        llm: model,
      });
      opened.push({ api: page, id: session.id });

      const answering = await session.responses.create("Reply with the single word: pong.");

      const seen: string[] = [];
      let answered = "";
      for await (const event of session.events()) {
        seen.push(event.kind);
        if (event.kind === "responded") {
          answered = event.text;
          break;
        }
        if (event.kind === "error") {
          if (exhausted(event.error)) {
            t.skip(`nothing left to answer with: ${event.error}`);
            return;
          }
          assert.fail(`the session reported: ${event.error}`);
        }
      }

      assert.ok(answered.length > 0, `nothing was answered; saw ${seen.join(", ")}`);

      if (answering.id) {
        const items = await answering.items.all();
        assert.ok(items.length > 0, "the turn was answered and nothing was written down");
        assert.equal(items[0]?.kind, "said", "every turn opens with what was asked");
      }

      await session.close();
    });

    it("forks a conversation into one of its own", async (t) => {
      if (!agentName) {
        t.skip("this deployment has no agent configured to address by name");
        return;
      }
      const page = new Client({
        url: acceleration,
        apiKey,
        apiSecret,
        authenticate: true,
        webSocket: authenticated(await backend.backend.headers()),
      });

      const parent = await agentOf(page).sessions.create({
        title: "the first ask",
        persist_conversation: true,
        llm: model,
      });
      opened.push({ api: page, id: parent.id });

      const forked = await parent.fork({ title: "asked again" });
      opened.push({ api: page, id: forked.id });

      assert.equal(forked.created.forked_from, parent.id);
      assert.equal(forked.created.title, "asked again");
      assert.notEqual(
        forked.conversationId,
        parent.conversationId,
        "a fork writes its own transcript",
      );

      await forked.close();
      await parent.close();
    });
  },
);

describe("the hosted documentation corpus", { skip: await unreachable(docs) }, () => {
    it("is indexed, and says as of when", async () => {
      const response = await fetch(`${docs}/health`);
      const health = (await response.json()) as {
        ready: boolean;
        catalog?: { revision: string; pages: number };
      };

      assert.equal(response.status, 200);
      assert.equal(health.ready, true);
      assert.ok(health.catalog?.revision, "a corpus that cannot say what it was built from");
      assert.ok((health.catalog?.pages ?? 0) > 0);
    });

    it("publishes the scopes a page may narrow a search to", async () => {
      const catalog = (await fetch(`${docs}/v1/search/catalog`).then((r) => r.json())) as {
        scopes: { product: string; sdk: string; current: boolean }[];
      };

      const current = catalog.scopes.filter((scope) => scope.current);
      assert.ok(current.length > 0, "nothing current to search");
      assert.ok(
        current.some((scope) => scope.product === "chat"),
        "the chat documentation is not in the corpus",
      );
    });

    it("answers a search with citable hits", async () => {
      const url = new URL(`${docs}/v1/search`);
      url.search = new URLSearchParams({
        q: "connect a user",
        product: "chat",
        sdk: "react",
        limit: "5",
      }).toString();

      const found = (await fetch(url).then((r) => r.json())) as {
        hits: { title: string; url: string; snippet: string; revision: string }[];
      };

      assert.ok(found.hits.length > 0, "the corpus held nothing for a plain query");
      for (const hit of found.hits) {
        assert.ok(hit.title, "a hit with no title cannot be shown");
        assert.match(hit.url, /^https:\/\//, "a hit has to be somewhere to send somebody");
        assert.ok(hit.snippet);
        assert.ok(hit.revision, "a hit that cannot be cited as of a known commit");
      }
    });

    it("lets a browser read it directly", async () => {
      // Without this a page can only reach the corpus through a proxy of its own, which
      // is what the website's dev server had to stand up before these headers existed.
      const response = await fetch(`${docs}/v1/search?q=upload`, {
        headers: { Origin: "https://getstream.io" },
      });

      assert.equal(response.headers.get("access-control-allow-origin"), "*");

      const preflight = await fetch(`${docs}/v1/search`, {
        method: "OPTIONS",
        headers: {
          Origin: "https://getstream.io",
          "Access-Control-Request-Method": "GET",
        },
      });

      assert.ok(preflight.status < 400, `a preflight answered ${preflight.status}`);
      assert.match(preflight.headers.get("access-control-allow-methods") ?? "", /GET/);
    });

    it("refuses a query it cannot run rather than answering it empty", async () => {
      const response = await fetch(`${docs}/v1/search?q=`);

      assert.equal(response.status, 400);
      const failure = (await response.json()) as { code: string; message: string };
      assert.ok(failure.message, "a refusal that says nothing");
    });
});
