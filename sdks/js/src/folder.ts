import { readFile, readdir, stat } from "node:fs/promises";
import { basename, extname, join, posix, relative, sep } from "node:path";

import type { Folder, Skill } from "./agent.js";
import { ConfigurationError } from "./errors.js";

/** What an agent directory calls its system prompt. */
export const INSTRUCTIONS_FILE = "instructions.md";

/** What it calls the policy screening what may be asked of it. */
export const GUARDRAIL_FILE = "guardrail.md";

export const SKILLS_DIR = "skills";
export const KNOWLEDGE_DIR = "knowledge";

/** What a knowledge directory calls the pages it is kept filled from. */
export const KNOWLEDGE_URLS_FILE = "urls.yaml";

/**
 * The extensions a knowledge directory is read from.
 *
 * Anything else in there is left alone: a model looks things up in prose, not in a binary.
 */
const READABLE = new Set([".md", ".mdx", ".txt", ".rst", ".yaml", ".yml"]);

/** What a page in urls.yaml may say. */
const PAGE_KEYS = new Set(["url", "title", "description"]);

/**
 * Reads an agent directory.
 *
 * ```
 * agents/jean/
 *   instructions.md
 *   guardrail.md
 *   skills/think.md
 *   knowledge/pricing.md
 *   knowledge/urls.yaml
 * ```
 *
 * Everything in it is optional: a directory with only instructions.md is a valid agent, and
 * so is one with only skills. A skill is a markdown file with frontmatter naming what the
 * fast model sees; the body is the prompt only the subagent sees.
 */
export async function loadFolder(path: string): Promise<Folder> {
  const info = await stat(path).catch(() => undefined);
  if (!info?.isDirectory()) {
    throw new ConfigurationError(`${path} is not an agent directory`);
  }

  return {
    path,
    name: basename(path.replace(/[/\\]+$/, "")),
    instructions: (await maybeRead(join(path, INSTRUCTIONS_FILE))).trim(),
    guardrail: (await maybeRead(join(path, GUARDRAIL_FILE))).trim(),
    skills: await loadSkills(join(path, SKILLS_DIR)),
    knowledge: await loadKnowledge(join(path, KNOWLEDGE_DIR)),
    knowledgeURLs: parsePages(
      await maybeRead(join(path, KNOWLEDGE_DIR, KNOWLEDGE_URLS_FILE)),
      join(path, KNOWLEDGE_DIR, KNOWLEDGE_URLS_FILE),
    ),
  };
}

async function loadSkills(path: string): Promise<Skill[]> {
  const entries = await readdir(path, { withFileTypes: true }).catch(() => []);
  const skills: Skill[] = [];

  for (const entry of entries.sort((one, other) => one.name.localeCompare(other.name))) {
    if (!entry.isFile() || extname(entry.name) !== ".md") {
      continue;
    }
    const file = join(path, entry.name);
    skills.push(parseSkill(entry.name.replace(/\.md$/, ""), await readFile(file, "utf8"), file));
  }
  return skills;
}

/**
 * Reads a skill file: frontmatter between `---` lines, then the instructions.
 *
 * The recognised keys are name, description, capture_video and deadline. A
 * deadline is a Go duration, so `30s` and `2m` both read the way they look, and a bare
 * number is seconds.
 */
export function parseSkill(name: string, content: string, where = name): Skill {
  const [frontmatter, body] = cutFrontmatter(content);
  const skill: Skill = { name, description: "", instructions: body.trim() };

  for (const line of frontmatter.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const at = trimmed.indexOf(":");
    if (at < 0) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} is not a key and a value`);
    }

    const key = trimmed.slice(0, at).trim();
    const value = trimmed.slice(at + 1).trim().replace(/^["']|["']$/g, "");
    switch (key) {
      case "name":
        skill.name = value;
        break;
      case "description":
        skill.description = value;
        break;
      case "capture_video":
        if (value !== "true" && value !== "false") {
          throw new ConfigurationError(`${where}: capture_video must be true or false`);
        }
        skill.captureVideo = value === "true";
        break;
      case "deadline":
        skill.deadlineMs = parseDeadline(value, where);
        break;
      default:
        break;
    }
  }

  if (!skill.description) {
    throw new ConfigurationError(
      `${where}: a skill needs a description, since it is all the fast model sees`,
    );
  }
  if (!skill.instructions) {
    throw new ConfigurationError(
      `${where}: a skill needs instructions, since they are what the subagent answers under`,
    );
  }
  return skill;
}

/** Takes a Go duration, and a bare number as seconds. */
function parseDeadline(value: string, where: string): number {
  if (/^\d+(\.\d+)?$/.test(value)) {
    return Math.round(Number(value) * 1000);
  }

  const parts = value.matchAll(/(\d+(?:\.\d+)?)(ms|s|m|h)/g);
  const units: Record<string, number> = { ms: 1, s: 1000, m: 60_000, h: 3_600_000 };
  let total = 0;
  let matched = false;
  for (const [, amount, unit] of parts) {
    matched = true;
    total += Number(amount) * (units[unit as string] ?? 0);
  }
  if (!matched) {
    throw new ConfigurationError(`${where}: ${JSON.stringify(value)} is not a deadline`);
  }
  return Math.round(total);
}

/** Separates a leading `---` block from the body. */
function cutFrontmatter(content: string): [frontmatter: string, body: string] {
  const trimmed = content.replace(/^[\uFEFF \t\r\n]+/, "");
  if (!trimmed.startsWith("---")) {
    return ["", content];
  }

  const rest = trimmed.slice(3).replace(/^[\r\n]+/, "");
  const end = rest.indexOf("\n---");
  if (end < 0) {
    return ["", content];
  }
  return [rest.slice(0, end), rest.slice(end + 4).replace(/^[-\r\n]+/, "")];
}

async function loadKnowledge(path: string): Promise<{ source: string; text: string }[]> {
  const info = await stat(path).catch(() => undefined);
  if (!info) {
    return [];
  }
  if (!info.isDirectory()) {
    throw new ConfigurationError(`${path} is not a directory`);
  }

  const documents: { source: string; text: string }[] = [];
  const declaration = join(path, KNOWLEDGE_URLS_FILE);

  const walk = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((one, other) => one.name.localeCompare(other.name))) {
      const file = join(directory, entry.name);
      if (entry.isDirectory()) {
        await walk(file);
        continue;
      }
      if (!READABLE.has(extname(entry.name).toLowerCase())) {
        continue;
      }
      // Only the declaration at the root is a declaration; deeper, urls.yaml is a document
      // like any other.
      if (file === declaration) {
        continue;
      }

      const text = await readFile(file, "utf8");
      if (!text.trim()) {
        continue;
      }
      documents.push({ source: relative(path, file).split(sep).join(posix.sep), text });
    }
  };

  await walk(path);
  return documents;
}

/**
 * Reads the pages a knowledge base is kept filled from.
 *
 * A page is a subscription rather than a copy: what a crawler makes of it is what ends up
 * in the knowledge base. It is written either way — the url on its own, or a mapping naming
 * it alongside what it is:
 *
 * ```yaml
 * - https://example.com/pricing
 * - url: https://example.com/plans
 *   title: Plans
 *   description: What each plan includes.
 * ```
 *
 * The subset is parsed here rather than with a YAML library, because a dependency for one
 * list of urls would be a dependency in every browser bundle that imports this package.
 * A key it does not recognise is refused rather than dropped into a subscription nobody
 * described, and so is a url that is not http.
 */
export function parsePages(
  content: string,
  where = KNOWLEDGE_URLS_FILE,
): { url: string; title?: string; description?: string }[] {
  const pages: { url: string; title?: string; description?: string }[] = [];

  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const item = trimmed.startsWith("- ") || trimmed === "-";
    const body = item ? trimmed.slice(1).trim() : trimmed;

    // A mapping needs whitespace after the colon, which is what tells `title: Plans` from
    // the colon in `https://example.com`.
    const mapping = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(body);
    if (!mapping) {
      // A bare scalar is the url, which is the short way of writing a page.
      if (!item) {
        throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} is not a page`);
      }
      pages.push({ url: body });
      continue;
    }

    const key = mapping[1] as string;
    const value = (mapping[2] ?? "").trim().replace(/^["']|["']$/g, "");
    if (!PAGE_KEYS.has(key)) {
      // Reported rather than dropped into a subscription nobody described.
      throw new ConfigurationError(
        `${where}: ${JSON.stringify(key)} is not something a page says; url, title and description are`,
      );
    }

    if (item) {
      pages.push({ url: "" });
    }
    const page = pages.at(-1);
    if (!page) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(key)} belongs to no page`);
    }
    if (key === "url") {
      page.url = value;
    } else if (key === "title") {
      page.title = value;
    } else {
      page.description = value;
    }
  }

  for (const page of pages) {
    if (!/^https?:\/\//.test(page.url)) {
      throw new ConfigurationError(
        `${where}: ${JSON.stringify(page.url)} is not an http or https url`,
      );
    }
  }
  return pages;
}

async function maybeRead(path: string): Promise<string> {
  return readFile(path, "utf8").catch(() => "");
}
