import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { A2UI_BASIC_CATALOG, A2UI_PROTOCOL, A2UI_WRITING } from "../src/generated/a2ui.js";
import { compileUIPrompt, ConfigurationError } from "../src/index.js";
import { loadFolder } from "../src/node.js";

/** sdks/testdata/ui, from build/tests where the suite runs. sdks/go/agents/ui_test.go reads it too. */
const FIXTURE = fileURLToPath(new URL("../../../testdata/ui/", import.meta.url));
/** The Go SDK's copy of the A2UI catalog, which src/generated/a2ui.ts is written from. */
const GO_A2UI = fileURLToPath(new URL("../../../go/agents/a2ui/", import.meta.url));

/** Writes an agent directory and hands back its path. */
async function agentDir(files: Record<string, string>): Promise<string> {
  const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "concierge");
  for (const [name, content] of Object.entries(files)) {
    const file = join(root, name);
    await mkdir(join(file, ".."), { recursive: true });
    await writeFile(file, content);
  }
  return root;
}

describe("loadFolder ui", () => {
  it("compiles the shared fixture to the same instructions the Go SDK writes", async () => {
    const golden = await readFile(join(FIXTURE, "concierge.instructions.txt"), "utf8");

    const folder = await loadFolder(join(FIXTURE, "concierge"));

    assert.equal(`${folder.instructions}\n`, golden);
  });

  it("appends nothing for a bare ui:, which YAML reads as null", async () => {
    const bare = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui:\n", "instructions.md": "You are Juno." }),
    );
    const commented = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui:\n  # nothing yet\n", "instructions.md": "You are Juno." }),
    );

    assert.equal(bare.instructions, "You are Juno.");
    assert.equal(commented.instructions, "You are Juno.");
  });

  it("offers the whole catalog for ui: {}", async () => {
    const folder = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui: {}\n", "instructions.md": "You are Juno." }),
    );

    assert.match(folder.instructions, /^You are Juno\.\n\n## Interactive UI Generation\n/);
    assert.match(folder.instructions, /\n- BarChart\(/);
  });

  it("compiles the shared A2UI fixture to the same instructions the Go SDK writes", async () => {
    const golden = await readFile(join(FIXTURE, "concierge-a2ui.instructions.txt"), "utf8");

    const folder = await loadFolder(join(FIXTURE, "concierge-a2ui"));

    assert.equal(`${folder.instructions}\n`, golden);
  });

  it("carries the A2UI catalog, protocol and writing rules Go's generate.sh last wrote", async () => {
    // Fails when sdks/go/agents/a2ui is regenerated and npm run a2ui is not.
    const catalog = JSON.parse(await readFile(join(GO_A2UI, "basic_catalog.json"), "utf8"));

    assert.deepEqual(A2UI_BASIC_CATALOG, catalog);
    assert.equal(A2UI_PROTOCOL, await readFile(join(GO_A2UI, "protocol.md"), "utf8"));
    assert.equal(A2UI_WRITING, await readFile(join(GO_A2UI, "writing.md"), "utf8"));
  });

  it("refuses patch_mode with format a2ui, written before or after it, since only OpenUI patches are merged", async () => {
    for (const ui of ["  format: a2ui\n  patch_mode: true\n", "  patch_mode: true\n  format: a2ui\n"]) {
      const path = await agentDir({ "agent.yaml": `name: concierge\nui:\n${ui}` });

      await assert.rejects(
        () => loadFolder(path),
        (error: unknown) => error instanceof ConfigurationError && error.message.includes("ui.patch_mode is for openui"),
      );
    }
  });

  it("refuses patch_mode with format a2ui when compileUIPrompt is called directly", () => {
    assert.throws(
      () => compileUIPrompt({ format: "a2ui", patchMode: true }),
      (error: unknown) => error instanceof ConfigurationError && error.message.includes("ui.patch_mode is for openui"),
    );
  });

  it("writes the A2UI catalog's strings as Go does: line separators escaped, a lone surrogate replaced", async () => {
    // components.json as written, escapes and all, so the lone surrogate reaches the compiler.
    const badge = String.raw`[{"name":"loyalty_badge","description":"Tier\u2028shown","parameters":{"type":"object","properties":{"tier":{"type":"string","description":"Lone \ud800 half"}},"x-order":["tier"]}}]`;

    const folder = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: a2ui\n  keep: [Text]\n", "ui/components.json": badge }),
    );

    // The separator as the six characters \u2028, the surrogate as the character U+FFFD.
    assert.ok(
      folder.instructions.includes(
        '"LoyaltyBadge":{"description":"Tier\\u2028shown.","properties":{"component":{"const":"LoyaltyBadge"},"tier":{"description":"Lone \uFFFD half","type":"string"}}',
      ),
    );
  });

  it("refuses an A2UI keep naming a component the basic catalog does not have", async () => {
    const path = await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: a2ui\n  keep: [Text, CardHeader]\n" });

    await assert.rejects(
      () => loadFolder(path),
      (error: unknown) =>
        error instanceof ConfigurationError &&
        error.message.includes(
          'ui.keep names "CardHeader", which is neither a component nor a function of the A2UI basic catalog',
        ),
    );
  });

  it("refuses an app component named like an A2UI component, which OpenUI does not have", async () => {
    const column = JSON.stringify([
      { name: "column", description: "A brand column.", parameters: { type: "object", properties: {}, "x-order": [] } },
    ]);
    const openui = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: openui\n", "ui/components.json": column }),
    );
    const a2ui = await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: a2ui\n", "ui/components.json": column });

    assert.match(openui.instructions, /\n- Column\(\) — A brand column\.$/);
    await assert.rejects(
      () => loadFolder(a2ui),
      (error: unknown) =>
        error instanceof ConfigurationError &&
        error.message.includes('component "column" (Column) collides with a built-in component'),
    );
  });

  it("refuses an A2UI app component property that is not an identifier, which OpenUI allows", async () => {
    const badge = JSON.stringify([
      {
        name: "loyalty_badge",
        description: "The guest's loyalty tier.",
        parameters: { type: "object", properties: { "tier-name": { type: "string" } }, "x-order": ["tier-name"] },
      },
    ]);
    const openui = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: openui\n", "ui/components.json": badge }),
    );
    const a2ui = await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: a2ui\n", "ui/components.json": badge });

    assert.match(openui.instructions, /LoyaltyBadge\(tier-name\?: string\)/);
    await assert.rejects(
      () => loadFolder(a2ui),
      (error: unknown) =>
        error instanceof ConfigurationError &&
        error.message.includes('component "loyalty_badge" property "tier-name" must be an identifier'),
    );
  });

  it("writes an A2UI app component's events into its description, each a sentence, as the renderer does", async () => {
    const badge = JSON.stringify([
      {
        name: "loyalty_badge",
        description: "The guest's loyalty tier",
        parameters: { type: "object", properties: { tier: { type: "string" } }, required: ["tier"], "x-order": ["tier"] },
        events: [{ name: "redeem", description: "Tapping it redeems points" }, { name: "share" }],
      },
    ]);

    const folder = await loadFolder(
      await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: a2ui\n  keep: [Text]\n", "ui/components.json": badge }),
    );

    assert.ok(
      folder.instructions.includes(
        `"LoyaltyBadge":{"description":"The guest's loyalty tier. Tapping it redeems points.","properties":{"component":{"const":"LoyaltyBadge"},"tier":{"type":"string"}},"required":["component","tier"],"type":"object"}`,
      ),
    );
  });

  it("refuses an A2UI example holding a component it never defines", async () => {
    const path = await agentDir({
      "agent.yaml": "name: concierge\nui:\n  format: a2ui\n",
      "ui/blueprints.yaml": `- name: booking
  when: booking a table
  order: [Text]
  example: |
    {"version":"1.0","createSurface":{"surfaceId":"booking","catalogId":"getstream.io:app-catalog","components":[{"id":"root","component":"Column","children":["title","later"]},{"id":"title","component":"Text","text":"## Tonight"}]}}
`,
    });

    await assert.rejects(
      () => loadFolder(path),
      (error: unknown) =>
        error instanceof ConfigurationError &&
        error.message.includes('blueprint "booking" example is invalid: root holds later, which is not defined'),
    );
  });

  it("refuses an A2UI example using a component keep left out", async () => {
    const blueprints = `- name: booking
  when: booking a table
  order: [Text, Button]
  example: |
    {"version":"1.0","createSurface":{"surfaceId":"booking","catalogId":"getstream.io:app-catalog","components":[{"id":"root","component":"Column","children":["book"]},{"id":"book","component":"Button","child":"label","action":{"event":{"name":"book"}}},{"id":"label","component":"Text","text":"Book"}]}}
`;
    const kept = await loadFolder(
      await agentDir({
        "agent.yaml": "name: concierge\nui:\n  format: a2ui\n  keep: [Text, Column, Button]\n",
        "ui/blueprints.yaml": blueprints,
      }),
    );
    const left = await agentDir({
      "agent.yaml": "name: concierge\nui:\n  format: a2ui\n  keep: [Text, Column]\n",
      "ui/blueprints.yaml": blueprints,
    });

    assert.match(kept.instructions, /- \*\*booking\*\* \(when booking a table\): a Column of `Text -> Button`/);
    await assert.rejects(
      () => loadFolder(left),
      (error: unknown) =>
        error instanceof ConfigurationError && error.message.includes("book is a Button, which the catalog doesn't have"),
    );
  });

  it("refuses a field in components.json that Go's decoder refuses", async () => {
    const path = await agentDir({
      "agent.yaml": "name: concierge\nui:\n  format: openui\n",
      "ui/components.json": JSON.stringify([
        {
          name: "loyalty_badge",
          description: "The guest's loyalty tier.",
          parameters: { type: "object", properties: { tier: { type: "string", default: "gold" } }, "x-order": ["tier"] },
        },
      ]),
    });

    await assert.rejects(
      () => loadFolder(path),
      (error: unknown) => error instanceof ConfigurationError && error.message.includes('"default"'),
    );
  });
});
