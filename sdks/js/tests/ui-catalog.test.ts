import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

import { ConfigurationError } from "../src/index.js";
import { loadFolder } from "../src/node.js";

/** sdks/testdata/ui, from build/tests where the suite runs. sdks/go/agents/ui_test.go reads it too. */
const FIXTURE = fileURLToPath(new URL("../../../testdata/ui/", import.meta.url));

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

  it("refuses format a2ui, since the prompt it compiles teaches only OpenUI", async () => {
    const path = await agentDir({ "agent.yaml": "name: concierge\nui:\n  format: a2ui\n" });

    await assert.rejects(
      () => loadFolder(path),
      (error: unknown) =>
        error instanceof ConfigurationError && error.message.endsWith('ui.format must be openui, not "a2ui"'),
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
