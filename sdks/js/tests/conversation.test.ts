import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import { Client, ConfigurationError, conversation, type Schemas } from "../src/index.js";
import { TestRouter } from "./router.js";

/** A session as the router returns one, with only the fields this covers filled in. */
function held(id: string, channel: string): Schemas["Session"] {
  return {
    id: `sess_${id}`,
    agent_id: id,
    conversation_id: `agent:${channel}`,
    call_id: "",
    call_type: "agent",
    created_at: "2026-01-01T00:00:00Z",
    state: "live",
    user_id: "u_1",
  };
}

describe("conversation", () => {
  let router: TestRouter;
  let api: Client;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
  });

  afterEach(async () => {
    await router.stop();
  });

  it("opens a persistent text conversation without naming a channel", async () => {
    router.serve("GET", "/v1/agents/sessions", { body: [] });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: held("support-1", "support-abc"),
    });

    const opened = await conversation(api, { id: "support-1", config_id: "cfg_1" });

    const sent = router.last.body as Schemas["CreateSessionRequest"];
    assert.equal(sent.agent_id, "support-1");
    assert.equal(sent.text, true);
    assert.equal(sent.persist_conversation, true);
    assert.equal(sent.config_id, "cfg_1", "the caller's own fields are carried");
    assert.ok(
      !("conversation_id" in sent),
      "naming a channel nothing has been held in is refused by the backend",
    );
    assert.equal(opened.conversation_id, "agent:support-abc", "the backend named it");
  });

  it("resumes the channel it is given", async () => {
    router.serve("GET", "/v1/agents/sessions", { body: [] });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: held("support-1", "support-abc"),
    });

    await conversation(api, { id: "support-1", conversationId: "agent:support-abc" });

    const sent = router.last.body as Schemas["CreateSessionRequest"];
    assert.equal(sent.conversation_id, "agent:support-abc");
    assert.equal(sent.agent_id, "support-1", "a resume returns as the id it left as");
  });

  it("returns the session already holding a conversation rather than a second one", async () => {
    router.serve("GET", "/v1/agents/sessions", {
      body: [held("support-1", "support-abc")],
    });

    const opened = await conversation(api, { id: "support-1" });

    assert.equal(opened.id, "sess_support-1");
    assert.deepEqual(
      router.received.map((request) => request.method),
      ["GET"],
      "nothing was opened",
    );
  });

  it("opens one when somebody else's conversation is the one running", async () => {
    router.serve("GET", "/v1/agents/sessions", {
      body: [held("support-2", "support-xyz")],
    });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: held("support-1", "support-abc"),
    });

    const opened = await conversation(api, { id: "support-1" });

    assert.equal(opened.agent_id, "support-1");
  });

  it("refuses a channel written as anything but the backend writes it", async () => {
    for (const wrong of ["support-abc", "messaging:support-abc", ""]) {
      await assert.rejects(
        () => conversation(api, { id: "support-1", conversationId: wrong }),
        (raised: ConfigurationError) => {
          assert.match(raised.message, /is not a conversation id/);
          return true;
        },
        `${JSON.stringify(wrong)} was taken`,
      );
    }
    assert.equal(router.received.length, 0, "nothing was sent");
  });

  it("refuses a conversation with nothing to find it by", async () => {
    await assert.rejects(
      () => conversation(api, { id: "" }),
      (raised: ConfigurationError) => {
        assert.match(raised.message, /needs an id/);
        return true;
      },
    );
    assert.equal(router.received.length, 0, "nothing was sent");
  });
});
