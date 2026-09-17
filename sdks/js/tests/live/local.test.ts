import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";

import { Client, RouterError, Session, conversation } from "../../src/index.js";
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

  before(async () => {
    model = await conversationModel(api);
  });

  after(async () => {
    for (const id of opened) {
      await close(api, id);
    }
  });

  /** Opens a conversation the suite will close, and remembers it in case one fails. */
  async function open(id: string, conversationId?: string) {
    const session = await conversation(api, {
      id,
      llm: model,
      ...(conversationId ? { conversationId } : {}),
    });
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

  it("holds agent configs this SDK can read", async () => {
    const configs = await api.get("/v1/agents/configs");

    for (const config of configs) {
      assert.ok(config.id, "a stored config with no id could not be named in a session");
      assert.ok(config.name);
    }
  });

  it("opens a persistent conversation and names the channel itself", async () => {
    const id = uniqueId("opens");

    const session = await open(id);

    assert.equal(session.agent_id, id);
    assert.equal(session.text, true);
    assert.equal(session.state, "live");
    assert.match(
      session.conversation_id ?? "",
      /^agent:support-[0-9a-f-]{36}$/,
      "the backend names the channel, in the only shape it accepts",
    );
  });

  it("returns the session already holding a conversation rather than opening a second", async () => {
    const id = uniqueId("reuse");

    const first = await open(id);
    const again = await open(id);

    assert.equal(again.id, first.id);
    assert.equal(again.conversation_id, first.conversation_id);
  });

  it("resumes a closed conversation on the same channel", async () => {
    const id = uniqueId("resume");
    const first = await open(id);
    const channel = first.conversation_id ?? "";
    await close(api, first.id);

    const resumed = await open(id, channel);

    assert.notEqual(resumed.id, first.id, "a resume is a new session");
    assert.equal(resumed.conversation_id, channel, "holding the conversation it left");
  });

  it("refuses a channel nothing has been held in, which is why one is never named up front", async () => {
    // The reason `conversation` takes no channel on a first open. A resume reads the
    // channel without creating it, so a name nothing has used is not a conversation to
    // resume — and a caller who invented one would be told so only by the backend.
    await assert.rejects(
      () =>
        api.post("/v1/agents/sessions", {
          body: {
            text: true,
            persist_conversation: true,
            agent_id: uniqueId("invented"),
            conversation_id: `agent:support-${crypto.randomUUID()}`,
            llm: model,
          },
        }),
      (raised: RouterError) => {
        assert.ok(raised.status >= 400, `the backend took it: ${raised.message}`);
        return true;
      },
    );
  });

  it("answers over the session socket, and keeps what was said", async (t) => {
    const id = uniqueId("answers");
    const session = await Session.open(api, {
      text: true,
      persist_conversation: true,
      agent_id: id,
      llm: model,
    });
    opened.push(session.id);

    assert.ok(session.conversationId, "a persistent session carries its channel");

    session.respond("Reply with the single word: pong.");

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
    assert.ok(
      seen.includes("conversation_updated"),
      `what was said was never persisted; saw ${seen.join(", ")}`,
    );

    await session.close();
  });

  it("stops listing a conversation once it is closed", async () => {
    const id = uniqueId("closes");
    const session = await open(id);

    await close(api, session.id);

    const running = await api.get("/v1/agents/sessions");
    assert.ok(
      !running.some((each) => each.id === session.id),
      "a closed session is still listed as running",
    );
  });
});
