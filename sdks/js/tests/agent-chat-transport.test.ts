import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it } from "node:test";

import {
  AgentBranchError,
  AgentChat,
  agentChatTransport,
  type AgentChatMessage,
  type AgentTransportActivity,
  type AgentTransportFeed,
  type AgentTransportRequest,
} from "../src/chat.js";
import { Client, Session, supportBranchOf, type SupportBranch } from "../src/index.js";
import { messageField } from "../src/message-field.js";
import { TestRouter } from "./router.js";
import { QUERY_PATH, arrive, channelPage, chatClient, chatMessage, rewind, until, update, type TestChat } from "./stream-chat.js";

/** What the app makes of a Stream message: what the tests read back of the feed. */
interface Shown {
  readonly id: string;
  readonly text: unknown;
  readonly place: SupportBranch | undefined;
  /** The message a copy sent again was made of, which the controller never takes for a reply. */
  readonly resentFrom: unknown;
}

class Feed implements AgentTransportFeed<Shown> {
  isConnected = true;
  latest: readonly Shown[] = [];
  updateMessages(messages: readonly Shown[]) {
    this.latest = messages;
  }
  updateHasOlderMessages() {}
  updateActivity(_activity: AgentTransportActivity | undefined) {}
  get ids() {
    return this.latest.map((message) => message.id);
  }
  find(id: string): Shown {
    const found = this.latest.find((message) => message.id === id);
    assert.ok(found, `${id} is not shown`);
    return found;
  }
}

/**
 * What the router writes on a message: where it stands among its alternatives, and who wrote it.
 * A message with no alternatives has no `of`, `index` or `count`.
 */
function mark(responseId: string, of: "reply" | "question", index: number, count: number, question = responseId) {
  const place = count > 1 ? { of, index, count } : {};
  return {
    support_branch: { response_id: responseId, question_response_id: question, parent_response_id: "", ...place },
    support_runtime: { role: of === "reply" ? "assistant" : "user" },
  };
}

const at = (second: number) => new Date(Date.UTC(2026, 0, 1, 0, 0, second)).toISOString();

/** A message of a hidden branch, as `responses.branch` reads it back. */
function copied(id: string, second: number, custom: Record<string, unknown>, user = "john") {
  return { id, text: id, attachments: [], custom, created_at: at(second), user_id: user, role: user === "john" ? "user" : "assistant" };
}

// A conversation of two turns. The second question's reply was regenerated twice: r1 and r3
// are hidden, and r2, the second of three, is on the conversation.
const q0 = chatMessage("q0", 1, mark("resp_0", "question", 1, 1));
const r0 = chatMessage("r0", 2, { ...mark("resp_0", "reply", 1, 1), ai_generated: true }, "agent");
const q1 = chatMessage("q1", 3, mark("resp_1", "question", 1, 1));
const r2 = chatMessage("r2", 5, { ...mark("resp_2", "reply", 2, 3, "resp_1"), ai_generated: true }, "agent");
const conversationOf = [q0, r0, q1, r2];

const siblings = {
  body: {
    items: [
      { id: "resp_1", created_at: at(4), active: false, index: 1 },
      { id: "resp_2", created_at: at(5), active: true, index: 2 },
      { id: "resp_3", created_at: at(6), active: false, index: 3 },
    ],
    has_more: false,
  },
};

/** What `branch` reads of a regeneration of the second reply: its question, and its reply. */
function branchOf(responseId: string, index: number, second: number) {
  return {
    body: {
      active: false,
      items: [
        copied("q1", 3, mark("resp_1", "question", 1, 1)),
        copied(`r${index}`, second, { ...mark(responseId, "reply", index, 3, "resp_1"), ai_generated: true }, "agent"),
      ],
      has_more: false,
    },
  };
}

function refusal(status: number, code: string) {
  return { status, body: { error: { message: `refused: ${code}`, type: status === 410 ? "gone" : "conflict", code } } };
}

describe("agentChatTransport, branching", () => {
  let router: TestRouter;
  let api: Client;
  let closing: (() => Promise<void>) | undefined;

  beforeEach(async () => {
    router = await TestRouter.start();
    api = new Client({ url: router.url, customerId: "local" });
    router.serve("POST", "/v1/agents/sessions", {
      status: 201,
      body: {
        id: "sess_1",
        conversation_id: "agent:support-1",
        user_id: "john",
        agent_id: "john",
        state: "live",
        created_at: "2026-01-01T00:00:00Z",
      },
    });
    for (const id of ["resp_1", "resp_2", "resp_3"]) {
      router.serve("GET", `/v1/agents/sessions/sess_1/responses/${id}/siblings`, siblings);
    }
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch", branchOf("resp_1", 1, 4));
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_3/branch", branchOf("resp_3", 3, 6));
  });

  afterEach(async () => {
    await closing?.();
    closing = undefined;
    await router.stop();
  });

  /**
   * An AgentChat over a real stream-chat client (9, or 10 in CI's js-stream-chat-10 job) reading
   * the channel from the test router, and the transport, connected to a feed.
   */
  async function connected(messages: Record<string, unknown>[] = conversationOf) {
    const streamChat = await chatClient(router.url);
    api.use({ chat: streamChat as never });
    router.serve("POST", QUERY_PATH, channelPage(messages));
    const chat = new AgentChat({
      sessionId: "sess_1",
      open: async () => {
        const opening = Session.open(api, {});
        await router.socket();
        return opening;
      },
    });
    await chat.prepare();
    const transport = agentChatTransport(chat, {
      adapt: (message: AgentChatMessage): Shown => ({
        id: message.id,
        text: (message as { text?: unknown }).text,
        place: supportBranchOf(message),
        resentFrom: messageField(message, "resent_from"),
      }),
    });
    const feed = new Feed();
    const disconnect = transport.connect(feed);
    closing = async () => {
      disconnect();
      await chat.close();
    };
    await until(() => feed.ids.length === messages.length, "the channel never loaded");
    return { streamChat, chat, transport, feed };
  }

  const request = (patch: Partial<AgentTransportRequest<Shown>>): AgentTransportRequest<Shown> => ({
    kind: "message",
    messageId: "ctl_1",
    text: "",
    attachments: [],
    ...patch,
  });

  const responsesPath = "/v1/agents/sessions/sess_1/responses";

  function answering(rewoundTo: { message_id: string; inclusive: boolean } | undefined) {
    return {
      status: 202,
      body: {
        id: "resp_9",
        session_id: "sess_1",
        status: "running",
        created_at: at(9),
        ...(rewoundTo ? { rewound_to: rewoundTo } : {}),
      },
    };
  }

  /** The POSTs the transport made other than reading the channel and opening the session. */
  const writes = () =>
    router.received
      .filter((one) => one.method === "POST" && one.path !== QUERY_PATH && one.path !== "/v1/agents/sessions")
      .map((one) => one.path);

  /**
   * What activating the first regeneration of the second reply answers, the copies it sends
   * again arriving `after` milliseconds later, or never.
   */
  function activating(streamChat: TestChat, after: number | undefined) {
    const copies = [
      chatMessage("c_q1", 10, { ...mark("resp_1", "question", 1, 1), resent_from: "q1" }),
      chatMessage("c_r1", 11, { ...mark("resp_1", "reply", 1, 3), ai_generated: true, resent_from: "r1" }, "agent"),
    ];
    router.serve("POST", "/v1/agents/sessions/sess_1/responses/resp_1/activate", () => {
      if (after !== undefined) {
        setTimeout(() => copies.forEach((copy) => arrive(streamChat, copy)), after);
      }
      return { body: { leaf_response_id: "resp_1", already_active: false, resent: 2, rewound_to: { message_id: "r0", inclusive: false } } };
    });
    return copies;
  }

  it("regenerates a reply through its response, dropping what it hid at once", async () => {
    const { transport, feed } = await connected();
    router.serve("POST", responsesPath, answering({ message_id: "q1", inclusive: false }));

    await transport.send(request({ kind: "regenerate", message: feed.find("r2"), messageId: "ctl_regen" }));

    assert.deepEqual(feed.ids, ["q0", "r0", "q1"], "the old answer is gone before the new one arrives");
    const regenerate = router.requestsTo("POST", responsesPath)[0]?.body as Record<string, unknown>;
    assert.equal(regenerate["text"], "");
    assert.equal(regenerate["replaces"], "resp_2");
    assert.equal(regenerate["request_id"], "ctl_regen", "the controller's id, new for each regenerate and kept by its retry");
  });

  it("retries a regenerate whose reply the first attempt already hid, as the request the router took", async () => {
    const { streamChat, transport, feed } = await connected();
    router.serve("POST", responsesPath, (_request, calls) => {
      if (calls === 0) {
        // The router took it and Chat hid the reply, but its answer was lost on the way back.
        rewind(streamChat as TestChat, "q1", false, 9);
        return { status: 500, body: { error: { message: "lost", type: "internal" } } };
      }
      // A request id the router already took is answered with no id.
      return { status: 202, body: { session_id: "sess_1", status: "running", created_at: at(9) } };
    });
    const regenerate = request({ kind: "regenerate", message: feed.find("r2"), messageId: "ctl_regen" });
    await assert.rejects(transport.send(regenerate));
    await until(() => feed.ids.join() === "q0,r0,q1", `shows ${feed.ids.join()}`);

    // The controller's retry: the same request, whose reply is no longer shown.
    await transport.send(regenerate);

    const [first, retry] = router.requestsTo("POST", responsesPath).map((one) => one.body);
    assert.deepEqual(retry, first, "the same request, which the router answers once");
  });

  it("asks again under a fresh request id through the response that took the hidden reply's place", async () => {
    const { transport, feed } = await connected();
    router.serve("POST", responsesPath, answering({ message_id: "q1", inclusive: false }));
    const target = feed.find("r2");
    await transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_regen" }));

    // The agent couldn't answer: the controller's retry is the same target under a new id.
    await transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_again" }));

    const [, again] = router.requestsTo("POST", responsesPath).map((one) => one.body as Record<string, unknown>);
    assert.equal(again?.["replaces"], "resp_9", "the response the first attempt started, on the conversation");
    assert.equal(again?.["request_id"], "ctl_again");
  });

  it("regenerates a version still shown through its own response, not the one the last regenerate started", async () => {
    const { transport, feed } = await connected();
    // Answered without hiding anything, so r2 is still shown under its own mark, as a version the
    // arrows show again keeps its id.
    router.serve("POST", responsesPath, answering(undefined));
    const target = feed.find("r2");
    await transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_1" }));

    await transport.send(request({ kind: "regenerate", message: feed.find("r2"), messageId: "ctl_2" }));

    const replaces = router.requestsTo("POST", responsesPath).map((one) => (one.body as Record<string, unknown>)["replaces"]);
    assert.deepEqual(replaces, ["resp_2", "resp_2"], "the response the shown version carries");
  });

  it("forgets the response remembered for a target once the router says it is no longer there", async () => {
    const { transport, feed } = await connected();
    router.serve("POST", responsesPath, (_request, calls) =>
      calls === 1 ? refusal(409, "conflict") : answering({ message_id: "q1", inclusive: false }),
    );
    const target = feed.find("r2");
    await transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_1" }));
    await assert.rejects(transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_2" })));

    // r2 is hidden and nothing is remembered for it any more.
    await assert.rejects(
      transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_3" })),
      (error: unknown) => error instanceof AgentBranchError && error.code === "not_branchable",
    );

    const replaces = router.requestsTo("POST", responsesPath).map((one) => (one.body as Record<string, unknown>)["replaces"]);
    assert.deepEqual(replaces, ["resp_2", "resp_9"]);
  });

  it("keeps the response remembered for a target while the router finishes an unsettled switch", async () => {
    const { transport, feed } = await connected();
    router.serve("POST", responsesPath, (_request, calls) =>
      calls === 1 ? refusal(409, "switch_unfinished") : answering({ message_id: "q1", inclusive: false }),
    );
    const target = feed.find("r2");
    await transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_1" }));
    await assert.rejects(transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_2" })));

    await transport.send(request({ kind: "regenerate", message: target, messageId: "ctl_3" }));

    const replaces = router.requestsTo("POST", responsesPath).map((one) => (one.body as Record<string, unknown>)["replaces"]);
    assert.deepEqual(replaces, ["resp_2", "resp_9", "resp_9"], "switch_unfinished clears on a retry: the conversation did not change");
  });

  it("regenerates a reply with no alternatives yet, whose mark has no place", async () => {
    const { transport, feed } = await connected();
    router.serve("POST", responsesPath, answering({ message_id: "q0", inclusive: false }));

    await transport.send(request({ kind: "regenerate", message: feed.find("r0") }));

    assert.equal((router.requestsTo("POST", responsesPath)[0]?.body as Record<string, unknown>)["replaces"], "resp_0");
    assert.deepEqual(feed.ids, ["q0"]);
  });

  it("edits a question through the response on the conversation that answers it", async () => {
    const { transport, feed } = await connected();
    router.serve("POST", responsesPath, answering({ message_id: "q1", inclusive: true }));

    await transport.send(request({ kind: "edit", message: feed.find("q1"), messageId: "ctl_edit", text: "Rome instead" }));

    // q1 keeps the mark of resp_1, whose reply was regenerated away: the router edits through resp_2.
    assert.deepEqual(router.requestsTo("POST", responsesPath)[0]?.body, {
      text: "Rome instead",
      replaces: "resp_2",
      request_id: "ctl_edit",
    });
    assert.deepEqual(feed.ids, ["q0", "r0"]);
  });

  it("refuses a message from before branches, before asking the router anything", async () => {
    const unmarked = chatMessage("old", 1, {}, "agent");
    const { transport, feed } = await connected([unmarked]);

    for (const kind of ["regenerate", "edit"]) {
      await assert.rejects(
        transport.send(request({ kind, message: feed.find("old"), text: "again" })),
        (error: unknown) => error instanceof AgentBranchError && error.code === "not_branchable" && /before/.test(error.message),
      );
    }
    await assert.rejects(transport.switchBranch(feed.find("old"), "previous"), AgentBranchError);
    assert.deepEqual(writes(), []);
    assert.equal(router.received.filter((one) => one.method === "GET").length, 0);
  });

  it("pages through a reply's versions without writing to Chat, back to the conversation's own", async () => {
    const { transport, feed } = await connected();

    await transport.switchBranch(feed.find("r2"), "previous");
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r1"], "the shared question once, then the other answer");
    assert.equal(feed.find("r1").place?.index, 1);
    assert.equal(router.requestsTo("GET", "/v1/agents/sessions/sess_1/responses/resp_2/siblings")[0]?.query.get("of"), "reply");

    await transport.switchBranch(feed.find("r1"), "next");
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r2"], "the version on the conversation is the channel's own");
    assert.deepEqual(router.requestsTo("GET", "/v1/agents/sessions/sess_1/responses/resp_2/branch"), []);

    await transport.switchBranch(feed.find("r2"), "next");
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r3"]);

    await transport.switchBranch(feed.find("r3"), "next");
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r3"], "there is no fourth");

    await transport.switchBranch(feed.find("r3"), "previous");
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r2"]);
    assert.deepEqual(writes(), [], "showing a version writes nothing");
  });

  it("pages through a question's edits from where the edit parts, and back", async () => {
    const qe = chatMessage("qe", 7, mark("resp_5", "question", 2, 2));
    const re = chatMessage("re", 8, { ...mark("resp_5", "reply", 1, 1), ai_generated: true }, "agent");
    const { transport, feed } = await connected([q0, r0, qe, re]);
    const edits = {
      body: {
        items: [
          { id: "resp_1", created_at: at(3), active: false, index: 1, said: "q1" },
          { id: "resp_5", created_at: at(7), active: true, index: 2, said: "qe" },
        ],
        has_more: false,
      },
    };
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_5/siblings", edits);
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/siblings", edits);
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch", {
      body: {
        active: false,
        items: [
          copied("q1", 3, mark("resp_1", "question", 1, 2)),
          copied("r1", 4, { ...mark("resp_1", "reply", 1, 1), ai_generated: true }, "agent"),
        ],
        has_more: false,
      },
    });

    await transport.switchBranch(feed.find("qe"), "previous");
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r1"], "the edited question and its answer in place of the edit's");
    assert.equal(router.requestsTo("GET", "/v1/agents/sessions/sess_1/responses/resp_5/siblings")[0]?.query.get("of"), "question");

    await transport.switchBranch(feed.find("q1"), "next");
    assert.deepEqual(feed.ids, ["q0", "r0", "qe", "re"]);
  });

  it("abandons a switch once its signal aborts, reading and showing nothing of it", async () => {
    const { transport, feed } = await connected();
    const controller = new AbortController();

    const switching = transport.switchBranch(feed.find("r2"), "previous", controller.signal);
    controller.abort();
    await switching;

    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r2"]);
    assert.deepEqual(
      router.received.filter((one) => one.method === "GET").map((one) => one.path),
      [],
      "the reads were abandoned before they went",
    );
  });

  it("keeps what a version's message was sent again from when it shows the version", async () => {
    const { transport, feed } = await connected();
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch", {
      body: {
        active: false,
        items: [
          copied("q1", 3, mark("resp_1", "question", 1, 1)),
          copied("r1", 4, { ...mark("resp_1", "reply", 1, 3, "resp_1"), ai_generated: true, resent_from: "r1_first" }, "agent"),
        ],
        has_more: false,
      },
    });

    await transport.switchBranch(feed.find("r2"), "previous");

    assert.equal(feed.find("r1").resentFrom, "r1_first", "else the controller could take it for a reply");
  });

  it("keeps showing a version over the channel's newer messages", async () => {
    const { streamChat, transport, feed } = await connected();
    await transport.switchBranch(feed.find("r2"), "previous");

    arrive(streamChat, chatMessage("q4", 8, mark("resp_4", "question", 1, 1)));
    update(streamChat, { ...r0, text: "r0, revised" });
    await until(() => feed.latest.find((message) => message.id === "r0")?.text === "r0, revised", "the update never showed");

    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r1"]);
  });

  it("brings the version shown back before asking anything on it", async () => {
    const { streamChat, transport, feed } = await connected();
    activating(streamChat as TestChat, 0);
    router.serve("POST", responsesPath, answering(undefined));
    await transport.switchBranch(feed.find("r2"), "previous");

    await transport.send(request({ text: "And Porto?", messageId: "ctl_ask" }));

    assert.deepEqual(writes(), ["/v1/agents/sessions/sess_1/responses/resp_1/activate", responsesPath]);
    assert.deepEqual(feed.ids, ["q0", "r0", "c_q1", "c_r1"], "the version, sent again");
    assert.equal(feed.find("c_r1").resentFrom, "r1", "a copy from Chat says what it copies");
    const asked = router.requestsTo("POST", responsesPath)[0]?.body as Record<string, unknown>;
    assert.equal(asked["text"], "And Porto?");
    assert.equal(asked["request_id"], "ctl_ask");
  });

  it("drops a switch still reading when something is sent, rather than show it over the question", async () => {
    const { streamChat, chat, transport, feed } = await connected();
    // The switch stays on its way to the router until the question has gone and arrived.
    let open = () => {};
    const gate = new Promise<void>((resolve) => {
      open = resolve;
    });
    const session = chat.session.bind(chat);
    let calls = 0;
    chat.session = () => (calls++ === 0 ? gate.then(session) : session());
    router.serve("POST", responsesPath, () => {
      arrive(streamChat, chatMessage("q4", 8, mark("resp_4", "question", 1, 1)));
      return answering(undefined);
    });

    const switching = transport.switchBranch(feed.find("r2"), "previous");
    await transport.send(request({ text: "And Porto?", messageId: "ctl_ask" }));
    await until(() => feed.ids.includes("q4"), "the question never arrived");
    open();
    await switching;

    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r2", "q4"], "the question asked on the conversation shown");
    assert.deepEqual(writes(), [responsesPath], "nothing to bring back: no other version was shown when it was sent");
  });

  it("has the copies an activation sent again in the feed before what is asked resolves", async () => {
    const { streamChat, transport, feed } = await connected();
    // Chat's socket brings the copies a while after the router answered.
    activating(streamChat as TestChat, 100);
    router.serve("POST", responsesPath, answering(undefined));
    await transport.switchBranch(feed.find("r2"), "previous");

    await transport.send(request({ text: "And Porto?", messageId: "ctl_ask" }));

    assert.deepEqual(feed.ids, ["q0", "r0", "c_q1", "c_r1"], "else the controller takes a finished copy for the reply");
  });

  it("reads the channel again when the copies an activation sent again never come", async () => {
    const { streamChat, transport, feed } = await connected();
    const copies = activating(streamChat as TestChat, undefined);
    router.serve("POST", responsesPath, answering(undefined));
    await transport.switchBranch(feed.find("r2"), "previous");
    router.serve("POST", QUERY_PATH, channelPage([q0, r0, ...copies]));

    await transport.send(request({ text: "And Porto?", messageId: "ctl_ask" }));

    assert.equal(router.requestsTo("POST", QUERY_PATH).length, 2, "read again once the wait ran out");
    await until(() => feed.ids.join() === "q0,r0,c_q1,c_r1", `shows ${feed.ids.join()}`);
    assert.deepEqual(writes(), ["/v1/agents/sessions/sess_1/responses/resp_1/activate", responsesPath]);
  });

  it("asks anyway when reading the channel again for the copies hangs", async () => {
    const { streamChat, chat, transport, feed } = await connected();
    activating(streamChat as TestChat, undefined);
    router.serve("POST", responsesPath, answering(undefined));
    await transport.switchBranch(feed.find("r2"), "previous");
    chat.readLatest = () => new Promise<void>(() => {});

    await transport.send(request({ text: "And Porto?", messageId: "ctl_ask" }));

    assert.deepEqual(writes(), ["/v1/agents/sessions/sess_1/responses/resp_1/activate", responsesPath]);
  });

  it("brings the version shown back before regenerating a reply of it", async () => {
    const { streamChat, transport, feed } = await connected();
    router.serve("POST", "/v1/agents/sessions/sess_1/responses/resp_3/activate", () => {
      arrive(streamChat, chatMessage("c_q1", 10, { ...mark("resp_1", "question", 1, 1), resent_from: "q1" }));
      arrive(streamChat, chatMessage("c_r3", 11, { ...mark("resp_3", "reply", 3, 3, "resp_1"), resent_from: "r3" }, "agent"));
      return { body: { leaf_response_id: "resp_3", already_active: false, resent: 2, rewound_to: { message_id: "r0", inclusive: false } } };
    });
    router.serve("POST", responsesPath, answering(undefined));
    await transport.switchBranch(feed.find("r2"), "next");

    await transport.send(request({ kind: "regenerate", message: feed.find("r3") }));

    assert.deepEqual(writes(), ["/v1/agents/sessions/sess_1/responses/resp_3/activate", responsesPath]);
    assert.equal((router.requestsTo("POST", responsesPath)[0]?.body as Record<string, unknown>)["replaces"], "resp_3");
  });

  it("says why the router refused a branch in the person's words", async () => {
    const { transport, feed } = await connected();
    const refusedWith = (code: string) => (error: unknown) =>
      error instanceof AgentBranchError && error.code === code && !error.message.includes("refused");
    router.serve("POST", responsesPath, refusal(409, "branch_limit_reached"));
    router.serve("GET", "/v1/agents/sessions/sess_1/responses/resp_1/branch", refusal(410, "branch_unavailable"));
    router.serve("POST", "/v1/agents/sessions/sess_1/responses/resp_3/activate", refusal(409, "branch_too_long"));

    await assert.rejects(
      transport.send(request({ kind: "regenerate", message: feed.find("r2") })),
      refusedWith("branch_limit_reached"),
    );
    await assert.rejects(transport.switchBranch(feed.find("r2"), "previous"), refusedWith("branch_unavailable"));
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r2"]);

    await transport.switchBranch(feed.find("r2"), "next");
    await assert.rejects(transport.send(request({ text: "And Porto?" })), refusedWith("branch_too_long"));
    assert.deepEqual(feed.ids, ["q0", "r0", "q1", "r3"], "still shown, to continue in a new chat from");
    assert.equal(router.requestsTo("POST", responsesPath).length, 1, "nothing was asked on it");
  });

  it("goes back to the channel's own version when the channel is rewound", async () => {
    const { streamChat, transport, feed } = await connected();
    await transport.switchBranch(feed.find("r2"), "previous");
    router.serve("POST", QUERY_PATH, channelPage([q0, r0, q1]));

    // Another device regenerated the second reply: the question the versions part at stays.
    rewind(streamChat as TestChat, "q1", false, 7);

    await until(() => feed.ids.join() === "q0,r0,q1", `still shows ${feed.ids.join()}`);
  });
});
