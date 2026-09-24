import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";

import { Client, RouterError, type AgentHandle } from "../../src/index.js";
import { close, conversationModel, exhausted, uniqueId, unreachable } from "./target.js";

/**
 * The SDK against a router running on this machine.
 *
 * A local router takes a customer id at face value rather than a credential, because
 * nothing outside the machine can reach it, which is what makes this suite runnable
 * without secrets. What it proves is the part unit tests cannot: that the requests this
 * SDK builds are ones a real router accepts, and that what it says about conversations is
 * true of the backend rather than of the test server.
 */
const url = process.env["LOCAL_ACCELERATION_URL"] ?? "http://127.0.0.1:8098";
const customerId = process.env["LOCAL_ACCELERATION_CUSTOMER_ID"] ?? "support-local";

describe("the local router", { skip: await unreachable(url) }, () => {
  const api = new Client({ url, customerId });
  const opened: string[] = [];
  let model = "";
  /** An agent this router actually has configured, which is what a name resolves against. */
  let agent: AgentHandle | undefined;

  before(async () => {
    model = await conversationModel(api);
    const configs = await api.get("/v1/agents/configs").catch(() => []);
    const named = configs.find((config) => config.name);
    agent = named?.name ? api.agent(named.name) : undefined;
  });

  after(async () => {
    for (const id of opened) {
      await close(api, id);
    }
  });

  /** Remembers a session so a failing test does not leave one running. */
  function remember<T extends { id: string }>(session: T): T {
    if (!opened.includes(session.id)) {
      opened.push(session.id);
    }
    return session;
  }

  it("is healthy, and says what it depends on", async () => {
    const health = await api.get("/health");

    assert.equal(health.status, "ok");
    assert.equal(health.dependencies?.["llm"], "ok", "no conversation could be held");
  });

  it("resolves an agent config by the name it is stored under", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const config = await agent.config();

    assert.ok(config, `there is no config called ${agent.name}`);
    assert.equal(config.name, agent.name);
  });

  it("opens a conversation with labels, and reads them back", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const title = uniqueId("titled");
    const session = await agent.sessions.create({
      title,
      description: "opened by the local live suite",
      project: "docs",
      custom: { suite: "local" },
      persist_conversation: true,
      llm: model,
    });
    remember(session);

    assert.equal(session.created.title, title);
    assert.equal(session.created.project, "docs");
    assert.deepEqual(session.created.custom, { suite: "local" });
    assert.match(
      session.conversationId,
      /^agent:support-[0-9a-f-]{36}$/,
      "the backend names the channel, in the only shape it accepts",
    );

    await session.close();
  });

  it("finds a conversation again by what it was called", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const title = `Sendbird ${uniqueId("search")}`;
    const session = remember(
      await agent.sessions.create({ title, persist_conversation: true, llm: model }),
    );
    await session.close();

    const listed = await agent.sessions.query({ limit: 50 });
    assert.ok(
      listed.some((each) => each.id === session.id),
      "a conversation that ended is not being listed",
    );

    const found = await agent.sessions.search("Sendbird", { limit: 50 });
    assert.ok(
      found.some((each) => each.id === session.id),
      `searching for the title found nothing; ${found.length} other results`,
    );
  });

  it("keeps nothing about an incognito conversation", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const title = uniqueId("incognito");
    const session = remember(await agent.sessions.create({ incognito: true, title, llm: model }));
    assert.equal(session.created.incognito, true);
    assert.equal(session.conversationId, "", "an incognito session opens no channel");
    await session.close();

    const found = await agent.sessions.search(title, { limit: 50 });
    assert.equal(found.length, 0, "an incognito conversation was written down after all");
  });

  it("names the turn it answers, and writes down what the turn was made of", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const session = remember(
      await agent.sessions.create({ persist_conversation: true, llm: model }),
    );

    const answering = await session.responses.create("Reply with the single word: pong.");
    assert.ok(answering.id, "a recorded session names its turns");
    assert.equal(answering.status, "running");

    for await (const event of session.events()) {
      if (event.kind === "responded") {
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

    const items = await answering.items.all();
    assert.ok(items.length > 0, "the turn was answered and nothing was written down");
    assert.equal(items[0]?.kind, "said", "every turn opens with what was asked");
    assert.ok(
      items.some((item) => item.kind === "answer"),
      `no answer was recorded; saw ${items.map((item) => item.kind).join(", ")}`,
    );

    await session.close();
  });

  it("forks a conversation into one of its own", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const parent = remember(
      await agent.sessions.create({
        title: "the first ask",
        project: "docs",
        persist_conversation: true,
        llm: model,
      }),
    );

    const forked = remember(await parent.fork({ title: "asked again" }));

    assert.notEqual(forked.id, parent.id);
    assert.equal(forked.created.forked_from, parent.id);
    assert.equal(forked.created.title, "asked again");
    assert.equal(forked.created.project, "docs", "what the fork did not mention it inherits");
    assert.notEqual(
      forked.conversationId,
      parent.conversationId,
      "a fork writes its own transcript",
    );

    await forked.close();
    await parent.close();
  });

  it("rewinds to a turn, taking the later ones out of the conversation", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const session = remember(await agent.sessions.create({ llm: model }));
    for (const question of ["Reply with the single word: one.", "Reply with the single word: two."]) {
      await session.responses.create(question);
      for await (const event of session.events()) {
        if (event.kind === "responded") {
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
    }

    const [kept] = await session.responses.list();
    assert.ok(kept, "the first turn was not written down");
    await session.responses.rewind(kept);

    const left = await session.responses.list();
    assert.deepEqual(left.map((response) => response.id), [kept.id]);
    const forked = remember(await session.fork({ response_id: kept.id }));
    assert.equal(forked.created.forked_from, session.id);

    await forked.close();
    await session.close();
  });

  it("refuses to rewind a conversation kept in Stream Chat, which would bring the turns back", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const session = remember(await agent.sessions.create({ persist_conversation: true, llm: model }));

    await assert.rejects(
      () => session.responses.rewind("anything"),
      (raised: RouterError) => {
        assert.equal(raised.status, 400);
        assert.match(raised.message, /fork it at the response/);
        return true;
      },
    );

    await session.close();
  });

  it("refuses to fork an incognito conversation, since there is nothing to fork from", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const session = remember(await agent.sessions.create({ incognito: true, llm: model }));

    await assert.rejects(
      () => session.fork(),
      (raised: RouterError) => {
        assert.equal(raised.status, 400);
        assert.match(raised.message, /nothing to fork/);
        return true;
      },
    );

    await session.close();
  });

  it("stops listing a conversation as running once it is closed", async (t) => {
    if (!agent) {
      t.skip("this router has no agent configured to address by name");
      return;
    }

    const session = remember(await agent.sessions.create({ llm: model }));

    await session.close();

    const running = await agent.sessions.query({ state: "running", limit: 50 });
    assert.ok(
      !running.some((each) => each.id === session.id),
      "a closed session is still listed as running",
    );
  });
});
