import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { ConfigurationError } from "../src/index.js";
import { loadFolder, parsePages, parseSimulations, parseSkill } from "../src/node.js";

/** Writes an agent directory, with an agent.yaml unless told otherwise, and hands back its path. */
async function agentDir(files: Record<string, string | Uint8Array>): Promise<string> {
  const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
  await mkdir(root, { recursive: true });

  for (const [name, content] of Object.entries({ "agent.yaml": "", ...files })) {
    const file = join(root, name);
    await mkdir(join(file, ".."), { recursive: true });
    await writeFile(file, content);
  }
  return root;
}

describe("loadFolder", () => {
  it("reads an agent written down as a directory", async () => {
    const path = await agentDir({
      "instructions.md": "  Be brief.\n",
      "guardrail.md": "---\nkind: llm\n---\nRefuse medical advice.\n",
      "skills/think.md": "---\ndescription: Think it through\n---\nWork step by step.\n",
      "knowledge/pricing.md": "The plans cost money.\n",
      "knowledge/urls.yaml": "- https://example.com/pricing\n",
    });

    const folder = await loadFolder(path);

    assert.equal(folder.name, "jean");
    assert.equal(folder.instructions, "Be brief.");
    assert.match(folder.guardrail, /Refuse medical advice/);
    assert.equal(folder.skills.length, 1);
    assert.equal(folder.skills[0]?.name, "think");
    assert.equal(folder.skills[0]?.instructions, "Work step by step.");
    assert.deepEqual(folder.knowledge, [
      { source: "pricing.md", text: "The plans cost money.\n" },
    ]);
    assert.deepEqual(folder.knowledgeURLs, [{ url: "https://example.com/pricing" }]);
  });

  it("is happy with a directory that only says one thing", async () => {
    const folder = await loadFolder(await agentDir({ "instructions.md": "Be brief." }));

    assert.equal(folder.instructions, "Be brief.");
    assert.deepEqual(folder.skills, []);
    assert.deepEqual(folder.knowledge, []);
    assert.deepEqual(folder.knowledgeURLs, []);
  });

  it("refuses something that is not a directory", async () => {
    await assert.rejects(() => loadFolder(join(tmpdir(), "not-there")), ConfigurationError);
  });

  it("refuses a directory without agent.yaml, since that is what makes it an agent", async () => {
    const path = await agentDir({ "instructions.md": "Be brief." });
    await rm(join(path, "agent.yaml"));

    await assert.rejects(() => loadFolder(path), /agent\.yaml/);
  });

  it("reads what agent.yaml says the agent is called and runs on", async () => {
    const folder = await loadFolder(
      await agentDir({
        "agent.yaml": [
          "# the front desk",
          "name: receptionist",
          'llm: "openai/gpt-5.6"',
          'sts: ""',
          "harness: default",
          "mode: voice",
          "keyterms: [Vision Agents, Stream]",
          "plugins:",
          "  - weather",
          "tags:",
          "  team: support",
          "video:",
          "  source: camera",
          "dispatch:",
          "  text: enabled",
          "greeting:",
          "  text: Hello, how can I help?",
          "  mode: variation",
          "",
        ].join("\n"),
      }),
    );

    assert.equal(folder.name, "receptionist");
    assert.deepEqual(folder.settings, {
      name: "receptionist",
      llm: "openai/gpt-5.6",
      sts: "",
      harness: "default",
      mode: "voice",
      keyterms: ["Vision Agents", "Stream"],
      plugins: ["weather"],
      tags: { team: "support" },
      video: { source: "camera", max_frames: 1 },
      dispatch: { text: "enabled" },
      greeting: { text: "Hello, how can I help?", mode: "variation" },
    });
  });

  it("is called after its directory when agent.yaml names nothing", async () => {
    const folder = await loadFolder(await agentDir({ "agent.yaml": "llm: llm-fast\n" }));

    assert.equal(folder.name, "jean");
  });

  it("refuses a declaration it would otherwise have to guess at", async () => {
    for (const declaration of [
      "lmm: openai/gpt-5.6\n",
      "speed: 0.9\n",
      "greeting: Hello\n",
      "greeting:\n  mode: exact\n",
      "keyterms: Vision Agents\n",
      "llm:\n  - one\n  - two\n",
      "video:\n  max_frames: 9\n",
      "video:\n  frames: 2\n",
      "dispatch:\n  texts: enabled\n",
    ]) {
      await assert.rejects(
        async () => loadFolder(await agentDir({ "agent.yaml": declaration })),
        ConfigurationError,
        declaration,
      );
    }
  });

  it("reads connectors named alone and written out, with tools as a list either way", async () => {
    const folder = await loadFolder(
      await agentDir({
        "agent.yaml": [
          "name: concierge",
          "plugins:",
          "  - linear",
          "connectors:",
          "  - github",
          "  - name: issues",
          "    connector: linear",
          "    tools: [list_*, get_issue]",
          "    connection: work",
          "    required: true",
          "  -",
          "    name: notes",
          "    tools:",
          "      - read_*",
          "    required: false",
          "  - name: crm",
          "    tools: null",
          "",
        ].join("\n"),
      }),
    );

    assert.deepEqual(folder.settings?.connectors, [
      { name: "github" },
      { name: "issues", connector: "linear", tools: ["list_*", "get_issue"], connection: "work", required: true },
      { name: "notes", tools: ["read_*"], required: false },
      { name: "crm" },
    ]);
  });

  it("tells an empty connectors list from none, since one removes the bindings and the other keeps them", async () => {
    const without = await loadFolder(await agentDir({ "agent.yaml": "name: concierge\n" }));
    const empty = await loadFolder(await agentDir({ "agent.yaml": "name: concierge\nconnectors: []\n" }));

    assert.equal(without.settings?.connectors, undefined);
    assert.deepEqual(empty.settings?.connectors, []);
  });

  it("refuses a connector it would otherwise have to guess at", async () => {
    for (const [declaration, reason] of [
      ["connectors:\n  - ~\n", /a connector needs a name/],
      ["connectors:\n  - # linear\n", /a connector needs a name/],
      ["connectors:\n  - connector: linear\n", /a connector needs a name/],
      ['connectors:\n  - name: github\n    connection: ""\n', /connector "github": connection must be a connection id or label/],
      ["connectors:\n  - name: github\n    tools: []\n", /connector "github": tools: \[\] grants nothing/],
      ["connectors:\n  - name: github\n    plugin: github\n", /"plugin" is not something a connector says/],
      ["connectors:\n  - name: github\n    required: yes please\n", /required must be true or false/],
      ["connectors:\n  - linear\n    tools: [a]\n", /belongs to no connector/],
      ["plugins:\n  - linear\nconnectors:\n  - linear\n", /"linear" is named under both connectors and plugins/],
    ] as const) {
      await assert.rejects(
        async () => loadFolder(await agentDir({ "agent.yaml": declaration })),
        reason,
        declaration,
      );
    }
  });

  it("reads mcp_servers and channels", async () => {
    const folder = await loadFolder(
      await agentDir({
        "agent.yaml": [
          "name: concierge",
          "mcp_servers:",
          "  - name: crm # the sales one",
          '    url: "https://crm.example.com/mcp"',
          "    tools:",
          "      - list_*",
          "    scopes: [read, write]",
          "    user: true",
          "  -",
          "    name: notes",
          "    url: 'https://notes.example.com/mcp' # the notes one",
          "    scopes: []",
          "channels:",
          "  # the support line",
          "  whatsapp:",
          '    number: "+15551234567" # the main line',
          "  imessage:",
          "    number: ~",
          "  identity: phone # by phone",
          "",
        ].join("\n"),
      }),
    );

    assert.deepEqual(folder.settings?.mcpServers, [
      {
        name: "crm",
        url: "https://crm.example.com/mcp",
        tools: ["list_*"],
        scopes: ["read", "write"],
        user: true,
      },
      { name: "notes", url: "https://notes.example.com/mcp" },
    ]);
    assert.deepEqual(folder.settings?.channels, {
      whatsapp: { number: "+15551234567" },
      identity: "phone",
    });
  });

  it("reads scheduling, and sends it on a sync", async () => {
    const folder = await loadFolder(
      await agentDir({
        "agent.yaml": [
          "name: concierge",
          "scheduling:",
          "  enabled: true",
          "  max_tasks_per_user: 10",
          "  default_timezone: America/New_York",
          "",
        ].join("\n"),
      }),
    );

    assert.deepEqual(folder.settings?.scheduling, {
      enabled: true,
      max_tasks_per_user: 10,
      default_timezone: "America/New_York",
    });

    const without = await loadFolder(await agentDir({ "agent.yaml": "name: concierge\n" }));
    assert.equal(without.settings?.scheduling, undefined);

    await assert.rejects(
      async () =>
        loadFolder(
          await agentDir({ "agent.yaml": "name: concierge\nscheduling:\n  urgent: true\n" }),
        ),
      /unknown scheduling setting "urgent"/,
    );
    await assert.rejects(
      async () =>
        loadFolder(
          await agentDir({ "agent.yaml": "name: concierge\nscheduling:\n  enabled: yes please\n" }),
        ),
      /scheduling\.enabled must be true or false/,
    );
  });

  it("tells mcp_servers and channels that say nothing from ones that are empty", async () => {
    const without = await loadFolder(await agentDir({ "agent.yaml": "name: concierge\n" }));
    const nothing = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nmcp_servers: ~\nchannels:\n" }),
    );
    const nulled = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nmcp_servers:\nchannels: null\n" }),
    );
    const empty = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nmcp_servers: []\nchannels: {}\n" }),
    );

    assert.deepEqual(without.settings, { name: "concierge" });
    assert.deepEqual(nothing.settings, { name: "concierge" });
    assert.deepEqual(nulled.settings, { name: "concierge" });
    assert.deepEqual(empty.settings?.mcpServers, []);
    assert.deepEqual(empty.settings?.channels, {});
  });

  it("refuses mcp_servers, channels or sandbox_options it would otherwise have to guess at", async () => {
    const server = "mcp_servers:\n  - name: notes\n    url: https://example.com/mcp\n";
    for (const [declaration, reason] of [
      ["mcp_servers:\n  - https://example.com/mcp\n", /an mcp server is a mapping with a name and a url/],
      ["mcp_servers:\n  name: notes\n", /mcp_servers should be a list/],
      ["mcp_servers: notes\n", /mcp_servers should be a list of mappings/],
      [`${server}    branding: x\n`, /"branding" is not something an mcp server says/],
      [`${server}    user: yes please\n`, /mcp server "notes": user must be true or false/],
      [`${server}    tools: list_*\n`, /tools should be a list/],
      ["mcp_servers:\n  - name:\n      - notes\n", /an mcp server's name is one value, not a list/],
      [`connectors:\n  - name: notes\n    connector: linear\n${server}`, /"notes" is named under both connectors and mcp_servers/],
      ["channels: whatsapp\n", /channels should be a mapping/],
      ["channels:\n  telegram:\n    number: x\n", /unknown channels setting "telegram"/],
      ["channels:\n  sms:\n    number: x\n    sender: y\n", /unknown channels\.sms setting "sender"/],
      ['channels:\n  sms: "+15551234567"\n', /channels\.sms should be a mapping with a number/],
      ["channels:\n  identity:\n    - link\n", /channels\.identity is one value, not a list/],
      ["channels:\n    sms:\n      number: x\n  identity: link\n", /belongs to no channel/],
      ["sandbox_options: big\n", /sandbox_options should be a mapping/],
      ["sandbox_options:\n  memory: 4\n", /unknown sandbox_options setting "memory"/],
      ["sandbox_options:\n  timeout: 2h\n", /sandbox_options\.timeout must be a duration up to 30m/],
      ["sandbox_options:\n  cpu: two\n", /sandbox_options\.cpu must be a whole number/],
      ["sandbox_options:\n  network: maybe\n", /sandbox_options\.network must be true or false/],
    ] as const) {
      await assert.rejects(
        async () => loadFolder(await agentDir({ "agent.yaml": declaration })),
        reason,
        declaration,
      );
    }
  });

  it("reads knowledge in path order, with the source a reader would recognise", async () => {
    const path = await agentDir({
      "knowledge/b.md": "second",
      "knowledge/a.md": "first",
      "knowledge/deep/c.md": "third",
    });

    const folder = await loadFolder(path);

    assert.deepEqual(
      folder.knowledge.map((document) => document.source),
      ["a.md", "b.md", "deep/c.md"],
    );
  });

  it("leaves alone what a model cannot look things up in", async () => {
    const path = await agentDir({
      "knowledge/pricing.md": "prose",
      "knowledge/logo.png": "not prose",
      "knowledge/empty.md": "   \n",
    });

    const folder = await loadFolder(path);

    assert.deepEqual(
      folder.knowledge.map((document) => document.source),
      ["pricing.md"],
    );
  });

  it("treats the declaration as a declaration only at the root", async () => {
    const path = await agentDir({
      "knowledge/urls.yaml": "- https://example.com/pricing\n",
      "knowledge/deep/urls.yaml": "just a document\n",
    });

    const folder = await loadFolder(path);

    assert.deepEqual(
      folder.knowledge.map((document) => document.source),
      ["deep/urls.yaml"],
    );
    assert.equal(folder.knowledgeURLs.length, 1);
  });

  it("reads skills in name order", async () => {
    const path = await agentDir({
      "skills/recall.md": "---\ndescription: Remember\n---\nLook back.",
      "skills/explain.md": "---\ndescription: Explain\n---\nSpell it out.",
      "skills/notes.txt": "not a skill",
    });

    const folder = await loadFolder(path);

    assert.deepEqual(
      folder.skills.map((skill) => skill.name),
      ["explain", "recall"],
    );
  });

  it("reads a skill written as a folder, with the files it brings at any depth", async () => {
    const path = await agentDir({
      "skills/pdf/SKILL.md": "---\ndescription: Fill PDF forms\n---\nFill the form.\n",
      "skills/pdf/reference/forms.md": "# Forms\n",
      "skills/pdf/scripts/lib/fill.py": "print('filled')\n",
    });

    const [skill] = (await loadFolder(path)).skills;

    assert.equal(skill?.name, "pdf");
    assert.equal(skill?.instructions, "Fill the form.");
    assert.deepEqual(skill?.files, {
      "reference/forms.md": "# Forms\n",
      "scripts/lib/fill.py": "print('filled')\n",
    });
  });

  it("leaves hidden files and __pycache__ out of a skill folder", async () => {
    const path = await agentDir({
      "skills/pdf/SKILL.md": "---\ndescription: Fill PDF forms\n---\nFill the form.\n",
      "skills/pdf/.DS_Store": new Uint8Array([0x00, 0xff]),
      "skills/pdf/.cache/notes.md": "hidden",
      "skills/pdf/scripts/__pycache__/fill.cpython-313.pyc": new Uint8Array([0xcb, 0x0d, 0xff]),
    });

    const [skill] = (await loadFolder(path)).skills;

    assert.equal(skill?.files, undefined);
  });

  it("refuses a folder in skills/ without SKILL.md, since it is not a skill", async () => {
    const path = await agentDir({ "skills/pdf/fill.py": "print('filled')\n" });

    await assert.rejects(() => loadFolder(path), /pdf has no SKILL\.md, so it is not a skill/);
  });

  it("refuses a skill written both as a file and as a folder", async () => {
    const path = await agentDir({
      "skills/pdf.md": "---\ndescription: Fill\n---\nFill.\n",
      "skills/pdf/SKILL.md": "---\ndescription: Fill\n---\nFill.\n",
    });

    await assert.rejects(() => loadFolder(path), /skill "pdf" is both pdf and pdf\.md/);
  });

  it("refuses a file in a skill folder that is not text", async () => {
    const path = await agentDir({
      "skills/pdf/SKILL.md": "---\ndescription: Fill\n---\nFill.\n",
      "skills/pdf/form.pdf": new Uint8Array([0x25, 0x50, 0xff, 0xfe]),
    });

    await assert.rejects(() => loadFolder(path), /form\.pdf is not text/);
  });

  it("reads each simulations file as a list, in file order", async () => {
    const path = await agentDir({
      "simulations/lunch.yaml": [
        "- name: change of order",
        "  scenario: Order a club, then swap it for a wrap.",
        "  assertion: The final order is one wrap.",
        "  variations: 3",
        "- name: off the menu",
        "  scenario: Ask for a milkshake.",
        "  assertion: The agent says there is no milkshake.",
        "  mode: audio",
        "",
      ].join("\n"),
      "simulations/allergies.yml":
        "- name: peanut allergy\n  scenario: Ask whether the wrap has nuts.\n  assertion: The agent does not guess.\n",
      "simulations/notes.txt": "not a simulation",
    });

    const folder = await loadFolder(path);

    assert.deepEqual(
      folder.simulations?.map((simulation) => simulation.name),
      ["peanut allergy", "change of order", "off the menu"],
    );
    assert.equal(folder.simulations?.[1]?.variations, 3);
    assert.equal(folder.simulations?.[2]?.mode, "audio");
  });

  it("tells an empty simulations/ from none, since one deletes what is stored and the other keeps it", async () => {
    const path = await agentDir({});
    const without = await loadFolder(path);
    await mkdir(join(path, "simulations"));
    const emptied = await loadFolder(path);

    assert.equal(without.simulations, undefined);
    assert.deepEqual(emptied.simulations, []);
  });

  it("refuses a simulation name two files share, since a sync finds one by its name", async () => {
    const simulation = "- name: change of order\n  scenario: Swap the club.\n  assertion: One wrap.\n";
    const path = await agentDir({
      "simulations/a.yaml": simulation,
      "simulations/b.yaml": simulation,
    });

    await assert.rejects(() => loadFolder(path), /also declared in a\.yaml/);
  });
});

describe("parseSimulations", () => {
  it("reads a scenario written as prose over several lines, and the tags", () => {
    const [simulation] = parseSimulations(
      [
        "- name: lunch",
        "  scenario: >",
        "    Order a club,",
        "    then swap it.",
        "  assertion: |-",
        "    One wrap.",
        "    Nothing else.",
        "  max_turns: 6",
        "  caller_voice: amy",
        "  tags:",
        "    area: orders",
      ].join("\n"),
    );

    assert.deepEqual(simulation, {
      name: "lunch",
      scenario: "Order a club, then swap it.\n",
      assertion: "One wrap.\nNothing else.",
      max_turns: 6,
      caller_voice: "amy",
      tags: { area: "orders" },
    });
  });

  it("refuses a simulation it would otherwise have to guess at", () => {
    for (const content of [
      "- name: order\n  scenario: Order.\n  assertion: Ordered.\n  asertion: typo\n",
      "- name: order\n  scenario: Order.\n",
      "- scenario: Order.\n  assertion: Ordered.\n",
      "- name: order\n  scenario: Order.\n  assertion: Ordered.\n  mode: video\n",
      "- name: order\n  scenario: Order.\n  assertion: Ordered.\n  variations: 0\n",
      "name: order\n",
    ]) {
      assert.throws(() => parseSimulations(content), ConfigurationError, content);
    }
  });
});

describe("parseSkill", () => {
  it("takes the name from the frontmatter over the file name", () => {
    const skill = parseSkill(
      "think",
      "---\nname: deep_think\ndescription: Think\n---\nWork it out.",
    );

    assert.equal(skill.name, "deep_think");
  });

  it("reads a deadline written as a duration, and a bare number as seconds", () => {
    const of = (deadline: string) =>
      parseSkill("x", `---\ndescription: D\ndeadline: ${deadline}\n---\nGo.`).deadlineMs;

    assert.equal(of("30s"), 30_000);
    assert.equal(of("2m"), 120_000);
    assert.equal(of("1m30s"), 90_000);
    assert.equal(of("250ms"), 250);
    assert.equal(of("45"), 45_000);
  });

  it("refuses a deadline that is not one", () => {
    assert.throws(
      () => parseSkill("x", "---\ndescription: D\ndeadline: soon\n---\nGo."),
      ConfigurationError,
    );
  });

  it("reads capture_video and refuses anything that is not true or false", () => {
    assert.equal(
      parseSkill("x", "---\ndescription: D\ncapture_video: true\n---\nGo.").captureVideo,
      true,
    );
    assert.throws(
      () => parseSkill("x", "---\ndescription: D\ncapture_video: yes\n---\nGo."),
      ConfigurationError,
    );
  });

  it("refuses a skill with no description, since it is all the fast model sees", () => {
    assert.throws(() => parseSkill("x", "---\nname: x\n---\nGo."), ConfigurationError);
  });

  it("refuses a skill with no instructions, since they are what the subagent answers under", () => {
    assert.throws(() => parseSkill("x", "---\ndescription: D\n---\n"), ConfigurationError);
  });

  it("ignores a key it does not know and a comment", () => {
    const skill = parseSkill(
      "x",
      "---\n# what this is\ndescription: D\nauthor: ana\n---\nGo.",
    );

    assert.equal(skill.description, "D");
  });
});

describe("parsePages", () => {
  it("reads a page written as a url on its own", () => {
    assert.deepEqual(parsePages("- https://example.com/pricing\n"), [
      { url: "https://example.com/pricing" },
    ]);
  });

  it("reads a page written as a mapping naming what it is", () => {
    const pages = parsePages(
      ["- url: https://example.com/plans", "  title: Plans", "  description: What you get"].join(
        "\n",
      ),
    );

    assert.deepEqual(pages, [
      { url: "https://example.com/plans", title: "Plans", description: "What you get" },
    ]);
  });

  it("reads both ways of writing one in the same file", () => {
    const pages = parsePages(
      ["- https://example.com/pricing", "- url: https://example.com/plans", "  title: Plans"].join(
        "\n",
      ),
    );

    assert.equal(pages.length, 2);
    assert.equal(pages[0]?.url, "https://example.com/pricing");
    assert.equal(pages[1]?.title, "Plans");
  });

  it("reads nothing out of nothing", () => {
    assert.deepEqual(parsePages(""), []);
    assert.deepEqual(parsePages("# only a comment\n"), []);
  });

  it("reads how often a page is read again, and refuses one that is not a number of hours", () => {
    assert.deepEqual(parsePages("- url: https://example.com/plans\n  refresh_hours: 24\n"), [
      { url: "https://example.com/plans", refresh_hours: 24 },
    ]);
    for (const hours of ["0", "1.5", "daily"]) {
      assert.throws(
        () => parsePages(`- url: https://example.com\n  refresh_hours: ${hours}\n`),
        ConfigurationError,
        hours,
      );
    }
  });

  it("refuses a key nothing recognises rather than dropping it", () => {
    assert.throws(
      () => parsePages("- url: https://example.com\n  titel: Plans\n"),
      ConfigurationError,
    );
  });

  it("refuses a page that is not reachable over http", () => {
    assert.throws(() => parsePages("- ftp://example.com/pricing\n"), ConfigurationError);
  });
});
