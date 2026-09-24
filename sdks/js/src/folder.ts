import { readFile, readdir, stat, writeFile } from "node:fs/promises";
import { basename, extname, join, posix, relative, sep } from "node:path";

import type { Declaration, Folder, Skill, SyncStamp } from "./agent.js";
import type { Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** What makes a directory an agent: it names it and says what it runs on. */
export const AGENT_FILE = "agent.yaml";

/** Where a directory records the fingerprint it was last synced under, and when. */
export const AGENT_STAMP = ".agent_sync";

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
 *   agent.yaml
 *   instructions.md
 *   guardrail.md
 *   skills/think.md
 *   knowledge/pricing.md
 *   knowledge/urls.yaml
 * ```
 *
 * agent.yaml is what makes a directory an agent, so it is required. Everything else is
 * optional: a directory with only instructions.md beside it is a valid agent, and so is one
 * with only skills. A skill is a markdown file with frontmatter naming what the fast model
 * sees; the body is the prompt only the subagent sees.
 */
export async function loadFolder(path: string): Promise<Folder> {
  const info = await stat(path).catch(() => undefined);
  if (!info?.isDirectory()) {
    throw new ConfigurationError(`${path} is not an agent directory`);
  }
  const declaration = await readFile(join(path, AGENT_FILE), "utf8").catch(() => undefined);
  if (declaration === undefined) {
    throw new ConfigurationError(`${path} has no ${AGENT_FILE}, so it is not an agent directory`);
  }
  const settings = parseDeclaration(declaration, join(path, AGENT_FILE));

  return {
    path,
    name: settings.name || basename(path.replace(/[/\\]+$/, "")),
    settings,
    stamp: stampAt(join(path, AGENT_STAMP)),
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

const SCALARS = new Set([
  "name",
  "description",
  "mode",
  "stt",
  "tts",
  "sts",
  "voice",
  "llm",
  "subagent",
  "search",
  "greeting",
  "sandbox",
]);
const LISTS = new Set(["plugins", "keyterms"]);

/**
 * Reads agent.yaml: what the agent is called, and what it runs on.
 *
 * ```yaml
 * name: receptionist
 * llm: openai/gpt-5.6
 * keyterms: [Vision Agents, Stream]
 * tags:
 *   team: support
 * video:
 *   source: camera
 *   max_frames: 2
 * ```
 *
 * Parsed by hand for the same reason urls.yaml is: the declaration is flat but for two lists
 * and two small mappings, and a YAML library would land in every browser bundle. A key
 * nobody knows is refused rather than dropped, since a misspelled `llm` that goes quietly is
 * a config running on a model the file does not name.
 */
export function parseDeclaration(content: string, where = AGENT_FILE): Declaration {
  const declared: Declaration = {};
  const lines = content.split("\n");

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] as string;
    if (!line.trim() || line.trim().startsWith("#")) {
      continue;
    }
    if (/^\s/.test(line)) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(line.trim())} belongs to no key`);
    }
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(line.trimEnd());
    if (!entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(line)} is not a key and a value`);
    }
    const key = entry[1] as string;
    const inline = (entry[2] ?? "").replace(/\s+#.*$/, "").trim();

    // What is indented under a key with nothing after it is its list or its mapping.
    const nested: string[] = [];
    while (index + 1 < lines.length && /^\s+\S/.test(lines[index + 1] as string)) {
      nested.push((lines[++index] as string).trim());
    }

    if (SCALARS.has(key)) {
      if (nested.length > 0 || inline.startsWith("[")) {
        throw new ConfigurationError(`${where}: ${key} is one value, not a list`);
      }
      const value = scalar(inline);
      if (value !== undefined && (value || key === "sts")) {
        assign(declared, key, value);
      }
    } else if (LISTS.has(key)) {
      const items = inline ? flowList(inline, key, where) : nested.map((item) => listItem(item, key, where));
      const named = items.filter((item) => item);
      if (named.length > 0) {
        declared[key as "plugins" | "keyterms"] = named;
      }
    } else if (key === "tags") {
      declared.tags = mapping(nested, inline, key, where);
    } else if (key === "video") {
      declared.video = video(mapping(nested, inline, key, where), where);
    } else {
      throw new ConfigurationError(
        `${where} declares ${JSON.stringify(key)}, which is not something an agent has`,
      );
    }
  }
  return declared;
}

function assign(declared: Declaration, key: string, value: string): void {
  switch (key) {
    case "mode":
      declared.mode = value as Schemas["AgentMode"];
      break;
    case "sandbox":
      declared.sandbox = value as Schemas["Sandbox"];
      break;
    default:
      declared[key as "name"] = value;
  }
}

/** One value, unquoted. Nothing, `~` and `null` read as unset. */
function scalar(value: string): string | undefined {
  if (value === "" || value === "~" || value === "null") {
    return undefined;
  }
  const quoted = /^(["'])(.*)\1$/.exec(value);
  return quoted ? (quoted[2] as string) : value;
}

function flowList(value: string, key: string, where: string): string[] {
  const listed = /^\[(.*)\]$/.exec(value);
  if (!listed) {
    throw new ConfigurationError(`${where}: ${key} should be a list`);
  }
  return (listed[1] as string).split(",").map((item) => scalar(item.trim()) ?? "");
}

function listItem(line: string, key: string, where: string): string {
  if (!line.startsWith("-")) {
    throw new ConfigurationError(`${where}: ${key} should be a list`);
  }
  return scalar(line.slice(1).trim()) ?? "";
}

function mapping(nested: string[], inline: string, key: string, where: string): Record<string, string> {
  if (inline) {
    throw new ConfigurationError(`${where}: ${key} should be a mapping`);
  }
  const mapped: Record<string, string> = {};
  for (const line of nested) {
    const entry = /^([^:\s][^:]*):(?:\s+(.*))?$/.exec(line);
    if (!entry) {
      throw new ConfigurationError(`${where}: ${key} should be a mapping`);
    }
    mapped[(entry[1] as string).trim()] = scalar((entry[2] ?? "").trim()) ?? "";
  }
  return mapped;
}

function video(declared: Record<string, string>, where: string): Schemas["SessionVideo"] {
  for (const key of Object.keys(declared)) {
    if (key !== "source" && key !== "max_frames") {
      throw new ConfigurationError(`${where}: unknown video setting ${JSON.stringify(key)}`);
    }
  }
  const frames = declared["max_frames"] === undefined ? 1 : Number(declared["max_frames"]);
  if (!Number.isInteger(frames) || frames < 1 || frames > 8) {
    throw new ConfigurationError(`${where}: video.max_frames must be an integer from 1 to 8`);
  }
  return { ...(declared["source"] ? { source: declared["source"] } : {}), max_frames: frames };
}

/** `.agent_sync`: the fingerprint a directory was last synced under, and when. */
function stampAt(path: string): SyncStamp {
  return {
    async read() {
      const recorded: unknown = await readFile(path, "utf8")
        .then((text) => JSON.parse(text) as unknown)
        .catch(() => undefined);
      const hash = recorded && typeof recorded === "object" ? (recorded as { hash?: unknown }).hash : "";
      return typeof hash === "string" ? hash : "";
    },
    async write(hash) {
      const syncedAt = new Date().toISOString().replace(/\.\d+Z$/, "+00:00");
      await writeFile(path, `${JSON.stringify({ hash, synced_at: syncedAt })}\n`);
    },
  };
}

async function maybeRead(path: string): Promise<string> {
  return readFile(path, "utf8").catch(() => "");
}
