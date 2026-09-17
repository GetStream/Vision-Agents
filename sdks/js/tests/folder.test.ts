import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { ConfigurationError } from "../src/index.js";
import { loadFolder, parsePages, parseSkill } from "../src/node.js";

/** Writes an agent directory and hands back its path. */
async function agentDir(files: Record<string, string>): Promise<string> {
  const root = join(await mkdtemp(join(tmpdir(), "vision-agents-")), "jean");
  await mkdir(root, { recursive: true });

  for (const [name, content] of Object.entries(files)) {
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
