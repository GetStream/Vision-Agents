import { readFile, readdir, realpath, stat, writeFile } from "node:fs/promises";
import { basename, extname, isAbsolute, join, posix, relative, sep } from "node:path";

import type { ConnectorDeclaration, Declaration, Folder, Skill, SyncStamp } from "./agent.js";
import type { Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";
import {
  compileUIPrompt,
  type UIBlueprint,
  type UIDeclaration,
  type UIPortableComponent,
} from "./ui-catalog.js";

/** What makes a directory an agent: it names it and says what it runs on. */
export const AGENT_FILE = "agent.yaml";

/** Where a directory records the fingerprint it was last synced under, and when. */
export const AGENT_STAMP = ".agent_sync";

/** What an agent directory calls its system prompt. */
export const INSTRUCTIONS_FILE = "instructions.md";

/** What it calls the policy screening what may be asked of it. */
export const GUARDRAIL_FILE = "guardrail.md";

export const SKILLS_DIR = "skills";
/** What a skill written as a folder calls the file its frontmatter and instructions are in. */
export const SKILL_FILE = "SKILL.md";
export const KNOWLEDGE_DIR = "knowledge";
export const SIMULATIONS_DIR = "simulations";

/** What a knowledge directory calls the pages it is kept filled from. */
export const KNOWLEDGE_URLS_FILE = "urls.yaml";

/**
 * The extensions a knowledge directory is read from.
 *
 * Anything else in there is left alone: a model looks things up in prose, not in a binary.
 */
const READABLE = new Set([".md", ".mdx", ".txt", ".rst", ".yaml", ".yml"]);

/** What a page in urls.yaml may say. */
const PAGE_KEYS = new Set(["url", "title", "description", "refresh_hours"]);

/** What a simulation may say, as text and as a count. Anything else is refused. */
const SIMULATION_TEXT = new Set([
  "name",
  "scenario",
  "assertion",
  "mode",
  "caller_target",
  "judge_target",
  "caller_stt",
  "caller_tts",
  "caller_voice",
]);
const SIMULATION_COUNTS = new Set(["variations", "max_turns"]);

/**
 * Reads an agent directory.
 *
 * ```
 * agents/jean/
 *   agent.yaml
 *   instructions.md
 *   guardrail.md
 *   skills/think.md
 *   skills/pdf/SKILL.md
 *   skills/pdf/scripts/fill.py
 *   knowledge/pricing.md
 *   knowledge/urls.yaml
 *   simulations/lunch.yaml
 * ```
 *
 * agent.yaml is what makes a directory an agent, so it is required. Everything else is
 * optional: a directory with only instructions.md beside it is a valid agent, and so is one
 * with only skills. A skill is a markdown file with frontmatter naming what the fast model
 * sees; the body is the prompt only the subagent sees.
 */
export async function loadFolder(path: string): Promise<Folder> {
  const info = await stat(path).catch(missing);
  if (!info?.isDirectory()) {
    throw new ConfigurationError(`${path} is not an agent directory`);
  }
  const declaration = await readFile(join(path, AGENT_FILE), "utf8").catch(missing);
  if (declaration === undefined) {
    throw new ConfigurationError(`${path} has no ${AGENT_FILE}, so it is not an agent directory`);
  }
  const settings = parseDeclaration(declaration, join(path, AGENT_FILE));
  const simulations = await loadSimulations(join(path, SIMULATIONS_DIR));
  const baseInstructions = (await maybeRead(join(path, INSTRUCTIONS_FILE))).trim();
  const uiPrompt = settings.ui ? await loadUIPrompt(path, settings.ui) : "";
  const instructions = uiPrompt
    ? baseInstructions
      ? `${baseInstructions}\n\n${uiPrompt}`
      : uiPrompt
    : baseInstructions;

  return {
    path,
    name: settings.name || basename(path.replace(/[/\\]+$/, "")),
    settings,
    stamp: stampAt(join(path, AGENT_STAMP)),
    instructions,
    guardrail: (await maybeRead(join(path, GUARDRAIL_FILE))).trim(),
    skills: await loadSkills(join(path, SKILLS_DIR)),
    knowledge: await loadKnowledge(join(path, KNOWLEDGE_DIR)),
    knowledgeURLs: parsePages(
      await maybeRead(join(path, KNOWLEDGE_DIR, KNOWLEDGE_URLS_FILE)),
      join(path, KNOWLEDGE_DIR, KNOWLEDGE_URLS_FILE),
    ),
    ...(simulations ? { simulations } : {}),
  };
}

/**
 * Reads every .yaml and .yml file in simulations/, by file name, each a list of simulations.
 *
 * Undefined when there is no simulations/, which is different from one that declares none:
 * the first leaves what is stored alone and the second deletes it. A name two simulations
 * share is refused, since a sync finds a simulation by its name.
 */
async function loadSimulations(
  path: string,
): Promise<Schemas["SimulationDeclaration"][] | undefined> {
  const info = await stat(path).catch(missing);
  if (!info) {
    return undefined;
  }
  if (!info.isDirectory()) {
    throw new ConfigurationError(`${path} is not a directory`);
  }

  const simulations: Schemas["SimulationDeclaration"][] = [];
  const named = new Map<string, string>();
  const entries = await readdir(path, { withFileTypes: true });
  for (const entry of entries.sort((one, other) => one.name.localeCompare(other.name))) {
    const extension = extname(entry.name).toLowerCase();
    if (!entry.isFile() || (extension !== ".yaml" && extension !== ".yml")) {
      continue;
    }
    const file = join(path, entry.name);
    for (const simulation of parseSimulations(await readFile(file, "utf8"), file)) {
      const first = named.get(simulation.name);
      if (first) {
        throw new ConfigurationError(
          `${file}: simulation ${JSON.stringify(simulation.name)} is also declared in ${first}`,
        );
      }
      named.set(simulation.name, entry.name);
      simulations.push(simulation);
    }
  }
  return simulations;
}

/**
 * Reads skills/: a skill is either a markdown file, or a folder holding a SKILL.md and
 * whatever files it comes with, nested as deep as they are.
 */
async function loadSkills(path: string): Promise<Skill[]> {
  const entries = (await readdir(path, { withFileTypes: true }).catch(missing)) ?? [];
  const skills: Skill[] = [];
  const declared = new Map<string, string>();

  for (const entry of entries.sort((one, other) => one.name.localeCompare(other.name))) {
    let name = entry.name;
    let file = join(path, name);
    if (entry.isDirectory()) {
      file = join(file, SKILL_FILE);
    } else if (entry.isFile() && extname(name) === ".md") {
      name = name.replace(/\.md$/, "");
    } else {
      continue;
    }

    const content = await readFile(file, "utf8").catch(missing);
    if (content === undefined) {
      throw new ConfigurationError(
        `${join(path, entry.name)} has no ${SKILL_FILE}, so it is not a skill`,
      );
    }
    const first = declared.get(name);
    if (first) {
      throw new ConfigurationError(
        `skill ${JSON.stringify(name)} is both ${first} and ${entry.name}`,
      );
    }
    declared.set(name, entry.name);

    const skill = parseSkill(name, content, file);
    if (entry.isDirectory()) {
      const files = await loadSkillFiles(join(path, entry.name));
      if (Object.keys(files).length > 0) {
        skill.files = files;
      }
    }
    skills.push(skill);
  }
  return skills;
}

/** Reads every file in a skill folder but its SKILL.md, by slash path relative to the folder. */
async function loadSkillFiles(path: string): Promise<Record<string, string>> {
  const files: Record<string, string> = {};
  const text = new TextDecoder("utf-8", { fatal: true });

  const walk = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((one, other) => one.name.localeCompare(other.name))) {
      const file = join(directory, entry.name);
      if (skipped(entry.name) || file === join(path, SKILL_FILE)) {
        continue;
      }
      if (entry.isDirectory()) {
        await walk(file);
        continue;
      }
      const raw = await readFile(file);
      let content: string;
      try {
        content = text.decode(raw);
      } catch {
        throw new ConfigurationError(`${file} is not text, and a skill brings only text files`);
      }
      files[relative(path, file).split(sep).join(posix.sep)] = content;
    }
  };

  await walk(path);
  return files;
}

/**
 * Whether a file or folder in a skill folder is left behind: hidden ones such as .DS_Store,
 * and the __pycache__ that running a skill's Python scripts locally leaves.
 */
function skipped(name: string): boolean {
  return name.startsWith(".") || name === "__pycache__";
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

  if (!/^(?:\d+(?:\.\d+)?(?:ms|s|m|h))+$/.test(value)) {
    throw new ConfigurationError(`${where}: ${JSON.stringify(value)} is not a deadline`);
  }
  const parts = value.matchAll(/(\d+(?:\.\d+)?)(ms|s|m|h)/g);
  const units: Record<string, number> = { ms: 1, s: 1000, m: 60_000, h: 3_600_000 };
  let total = 0;
  for (const [, amount, unit] of parts) {
    total += Number(amount) * (units[unit as string] ?? 0);
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
  const info = await stat(path).catch(missing);
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
 *   refresh_hours: 24
 * ```
 *
 * `refresh_hours` is how often the backend reads the page again on its own, at least 1;
 * left out it is read when it is synced and never on a schedule.
 *
 * The subset is parsed here rather than with a YAML library, because a dependency for one
 * list of urls would be a dependency in every browser bundle that imports this package.
 * A key it does not recognise is refused rather than dropped into a subscription nobody
 * described, and so is a url that is not http.
 */
export function parsePages(
  content: string,
  where = KNOWLEDGE_URLS_FILE,
): { url: string; title?: string; description?: string; refresh_hours?: number }[] {
  const pages: { url: string; title?: string; description?: string; refresh_hours?: number }[] =
    [];

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
        `${where}: ${JSON.stringify(key)} is not something a page says; url, title, description and refresh_hours are`,
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
    } else if (key === "refresh_hours") {
      page.refresh_hours = count(value, key, where);
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

/**
 * Reads one simulations/*.yaml file: a list of conversations to run against the agent.
 *
 * ```yaml
 * - name: lunch order with a change
 *   scenario: >
 *     Order a turkey club, then swap it
 *     for a veggie wrap.
 *   assertion: The final order is one veggie wrap.
 *   variations: 3
 *   tags:
 *     area: orders
 * ```
 *
 * Parsed by hand for the same reason agent.yaml is. A scenario is prose, so a value may be
 * a `|` or `>` block as well as one line. A key nobody knows is refused, as in agent.yaml,
 * and so is a simulation missing its name, scenario or assertion.
 */
export function parseSimulations(
  content: string,
  where = SIMULATIONS_DIR,
): Schemas["SimulationDeclaration"][] {
  const lines = content.split("\n").map((line) => line.replace(/\r$/, ""));
  const listed: Record<string, string | number | Record<string, string>>[] = [];
  let current: Record<string, string | number | Record<string, string>> | undefined;

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index] as string;
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    let body = trimmed;
    let column = line.length - line.trimStart().length;
    if (trimmed === "-" || trimmed.startsWith("- ")) {
      current = {};
      listed.push(current);
      body = trimmed.slice(1).trim();
      column += trimmed.length - body.length;
      if (!body) {
        continue;
      }
    }
    if (!current) {
      throw new ConfigurationError(`${where}: a simulations file is a list, each item starting "- "`);
    }

    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(body);
    if (!entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} is not a key and a value`);
    }
    const key = entry[1] as string;
    const inline = (entry[2] ?? "").trim();

    // What is indented deeper than the key belongs to it: a block of prose, or the tags.
    const nested: string[] = [];
    while (index + 1 < lines.length) {
      const next = lines[index + 1] as string;
      if (next.trim() && next.length - next.trimStart().length <= column) {
        break;
      }
      nested.push(next);
      index++;
    }
    const written = nested.filter((one) => one.trim());

    if (SIMULATION_TEXT.has(key)) {
      if (/^[|>]-?$/.test(inline)) {
        current[key] = block(nested, inline);
      } else if (written.length > 0) {
        throw new ConfigurationError(`${where}: ${key} is one value, not a list`);
      } else {
        current[key] = scalar(uncomment(inline)) ?? "";
      }
    } else if (SIMULATION_COUNTS.has(key)) {
      current[key] = count(scalar(inline) ?? "", key, where);
    } else if (key === "tags") {
      current[key] = mapping(written.map((one) => one.trim()), inline, key, where);
    } else {
      throw new ConfigurationError(
        `${where}: ${JSON.stringify(key)} is not something a simulation says`,
      );
    }
  }

  return listed.map((fields) => {
    const name = fields["name"];
    if (!name) {
      throw new ConfigurationError(`${where}: a simulation needs a name`);
    }
    for (const required of ["scenario", "assertion"]) {
      if (!fields[required]) {
        throw new ConfigurationError(`${where}: simulation ${JSON.stringify(name)} needs a ${required}`);
      }
    }
    const mode = fields["mode"];
    if (mode !== undefined && mode !== "text" && mode !== "audio") {
      throw new ConfigurationError(
        `${where}: simulation ${JSON.stringify(name)} is text or audio, not ${JSON.stringify(mode)}`,
      );
    }
    return fields as Schemas["SimulationDeclaration"];
  });
}

const SCALARS = new Set([
  "name",
  "description",
  "mode",
  "region",
  "stt",
  "tts",
  "sts",
  "voice",
  "llm",
  "harness",
  "subagent",
  "search",
  "sandbox",
]);
const LISTS = new Set(["keyterms"]);

// What each key connectors replaced is refused with, since a file that names one is asking
// for tools it would no longer get.
const REPLACED = new Map([
  ["plugins", "plugins were replaced by connectors, so name the provider under connectors"],
  [
    "mcp_servers",
    "a server named by URL is now a custom connector, so create one for its URL and name it under connectors",
  ],
]);

/**
 * Reads agent.yaml: what the agent is called, and what it runs on.
 *
 * ```yaml
 * name: receptionist
 * llm: openai/gpt-5.6
 * harness: default
 * keyterms: [Vision Agents, Stream]
 * tags:
 *   team: support
 * video:
 *   source: camera
 *   max_frames: 2
 * dispatch:
 *   text: enabled
 * greeting:
 *   text: Hello, how can I help?
 *   mode: variation
 * floor_control:
 *   model: stream/chorus-asr
 *   max_wait_ms: 800
 * ```
 *
 * Parsed by hand for the same reason urls.yaml is: the declaration is flat but for one list
 * and five small mappings, and a YAML library would land in every browser bundle. A key
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
    const inline = uncomment(entry[2] ?? "").trim();

    // What is indented under a key with nothing after it is its list or its mapping.
    const indented: string[] = [];
    while (index + 1 < lines.length && /^\s+\S/.test(lines[index + 1] as string)) {
      indented.push(lines[++index] as string);
    }
    const nested = indented.map((one) => one.trim());

    if (SCALARS.has(key)) {
      if (nested.length > 0 || inline.startsWith("[")) {
        throw new ConfigurationError(`${where}: ${key} is one value, not a list`);
      }
      const value = scalar(inline);
      if (value !== undefined && (value || key === "sts")) {
        assign(declared, key, value, where);
      }
    } else if (LISTS.has(key)) {
      const items = inline ? flowList(inline, key, where) : nested.map((item) => listItem(item, key, where));
      const named = items.filter((item) => item);
      if (named.length > 0) {
        declared[key as "keyterms"] = named;
      }
    } else if (REPLACED.has(key)) {
      // An empty or null key declares nothing, and nothing was ever sent for it.
      const listed = scalar(inline) !== undefined && !/^\[\s*\]$/.test(inline);
      if (listed || nested.some((one) => !one.startsWith("#"))) {
        throw new ConfigurationError(`${where}: ${key} is no longer read: ${REPLACED.get(key)}`);
      }
    } else if (key === "tags") {
      declared.tags = mapping(nested, inline, key, where);
    } else if (key === "video") {
      declared.video = video(mapping(nested, inline, key, where), where);
    } else if (key === "dispatch") {
      declared.dispatch = dispatch(mapping(nested, inline, key, where), where);
    } else if (key === "greeting") {
      declared.greeting = greeting(mapping(nested, inline, key, where), where);
    } else if (key === "connectors") {
      declared.connectors = connectors(indented, inline, where);
    } else if (key === "channels") {
      const lines = channels(indented, inline, where);
      if (lines) {
        declared.channels = lines;
      }
    } else if (key === "scheduling") {
      declared.scheduling = scheduling(mapping(nested, inline, key, where), where);
    } else if (key === "sandbox_options") {
      declared.sandboxOptions = sandboxOptions(indented, inline, where);
    } else if (key === "floor_control") {
      const chosen = floorControl(indented, inline, where);
      if (chosen) {
        declared.floorControl = chosen;
      }
    } else if (key === "ui") {
      declared.ui = parseUIDeclaration(indented, inline, where);
    } else {
      throw new ConfigurationError(
        `${where} declares ${JSON.stringify(key)}, which is not something an agent has`,
      );
    }
  }
  return declared;
}

const CONNECTOR_KEYS = new Set([
  "name",
  "connector",
  "tools",
  "connection",
  "required",
  "sandbox",
  "require_approval",
]);

/**
 * Reads `connectors:`, each the alias alone or a mapping:
 *
 * ```yaml
 * connectors:
 *   - linear
 *   - name: issues
 *     connector: linear
 *     tools: [list_*, get_issue]
 *     connection: work
 *     required: true
 *   - name: code
 *     connector: github
 *     sandbox: true
 *     require_approval: [create_*]
 * ```
 */
function connectors(lines: string[], inline: string, where: string): ConnectorDeclaration[] {
  if (inline) {
    if (/^\[\s*\]$/.test(inline)) {
      return [];
    }
    return flowList(inline, "connectors", where).map((name) => connectorOf(new Map(), name, where));
  }
  return listed(lines, "connectors", "connector", where).map((item) =>
    connectorOf(item.keys, item.alias, where),
  );
}

type Written = { inline: string; nested: string[] };

/** A block list whose items are each a scalar alias or a mapping of keys to what they say. */
function listed(
  lines: string[],
  key: string,
  noun: string,
  where: string,
): { alias?: string; keys: Map<string, Written> }[] {
  const items: { alias?: string; keys: Map<string, Written> }[] = [];
  let itemColumn = -1;
  let keyColumn = -1;
  let value: { inline: string; nested: string[] } | undefined;
  for (const line of lines) {
    const trimmed = uncomment(line).trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const column = line.length - line.trimStart().length;
    let body = trimmed;
    let at = column;
    if (itemColumn < 0 || column <= itemColumn) {
      if (trimmed !== "-" && !trimmed.startsWith("- ")) {
        throw new ConfigurationError(`${where}: ${key} should be a list`);
      }
      itemColumn = column;
      body = trimmed.slice(1).trim();
      items.push({ keys: new Map() });
      value = undefined;
      keyColumn = -1;
      if (!body) {
        continue;
      }
      if (!/^[A-Za-z_][\w-]*:(\s|$)/.test(body)) {
        (items.at(-1) as { alias?: string }).alias = body;
        continue;
      }
      at = column + trimmed.length - body.length;
    }

    const item = items.at(-1) as (typeof items)[number];
    if (keyColumn < 0) {
      keyColumn = at;
    }
    if (at > keyColumn && value && item.alias === undefined) {
      value.nested.push(body);
      continue;
    }
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(body);
    if (at !== keyColumn || !entry || item.alias !== undefined) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} belongs to no ${noun}`);
    }
    value = { inline: (entry[2] ?? "").trim(), nested: [] };
    item.keys.set(entry[1] as string, value);
  }
  return items;
}

const CHANNEL_LINES = new Set(["whatsapp", "sms", "imessage"]);

/**
 * Reads `channels:`, the lines the agent answers on outside Stream Chat:
 *
 * ```yaml
 * channels:
 *   whatsapp:
 *     number: "+15551234567"
 *   sms:
 *     number: "+15557654321"
 *   identity: link
 * ```
 *
 * Undefined when the key says nothing. A line without a number is left out, which is what
 * the sync sends.
 */
function channels(
  lines: string[],
  inline: string,
  where: string,
): Schemas["AgentChannels"] | undefined {
  if (scalar(inline) !== undefined) {
    if (/^\{\s*\}$/.test(inline)) {
      return {};
    }
    throw new ConfigurationError(`${where}: channels should be a mapping`);
  }

  const keys = new Map<string, Written>();
  let keyColumn = -1;
  let value: Written | undefined;
  for (const line of lines) {
    const trimmed = uncomment(line).trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const column = line.length - line.trimStart().length;
    if (keyColumn < 0) {
      keyColumn = column;
    }
    if (column > keyColumn && value) {
      value.nested.push(trimmed);
      continue;
    }
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(trimmed);
    if (column !== keyColumn || !entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} belongs to no channel`);
    }
    value = { inline: (entry[2] ?? "").trim(), nested: [] };
    keys.set(entry[1] as string, value);
  }
  if (keyColumn < 0) {
    return undefined;
  }

  const declared: { -readonly [key in keyof Schemas["AgentChannels"]]: Schemas["AgentChannels"][key] } =
    {};
  for (const [key, written] of keys) {
    if (key === "identity") {
      if (written.nested.length) {
        throw new ConfigurationError(`${where}: channels.identity is one value, not a list`);
      }
      const identity = scalar(written.inline);
      if (identity) {
        declared.identity = identity as Schemas["ChannelIdentity"];
      }
      continue;
    }
    if (!CHANNEL_LINES.has(key)) {
      throw new ConfigurationError(`${where}: unknown channels setting ${JSON.stringify(key)}`);
    }
    if (scalar(written.inline) !== undefined) {
      throw new ConfigurationError(`${where}: channels.${key} should be a mapping with a number`);
    }
    const line = mapping(written.nested, "", `channels.${key}`, where);
    for (const setting of Object.keys(line)) {
      if (setting !== "number") {
        throw new ConfigurationError(
          `${where}: unknown channels.${key} setting ${JSON.stringify(setting)}`,
        );
      }
    }
    if (line["number"]) {
      declared[key as "whatsapp" | "sms" | "imessage"] = { number: line["number"] };
    }
  }
  return declared;
}

/**
 * Reads `scheduling:`, whether the agent may schedule tasks for its users:
 *
 * ```yaml
 * scheduling:
 *   enabled: true
 *   max_tasks_per_user: 10
 *   default_timezone: America/New_York
 * ```
 */
function scheduling(declared: Record<string, string>, where: string): Schemas["AgentScheduling"] {
  for (const key of Object.keys(declared)) {
    if (key !== "enabled" && key !== "max_tasks_per_user" && key !== "default_timezone") {
      throw new ConfigurationError(`${where}: unknown scheduling setting ${JSON.stringify(key)}`);
    }
  }
  const read: { -readonly [key in keyof Schemas["AgentScheduling"]]: Schemas["AgentScheduling"][key] } = {};
  if (declared["enabled"]) {
    if (declared["enabled"] !== "true" && declared["enabled"] !== "false") {
      throw new ConfigurationError(`${where}: scheduling.enabled must be true or false`);
    }
    read.enabled = declared["enabled"] === "true";
  }
  if (declared["max_tasks_per_user"]) {
    const maximum = Number(declared["max_tasks_per_user"]);
    if (!Number.isInteger(maximum) || maximum < 1) {
      throw new ConfigurationError(`${where}: scheduling.max_tasks_per_user must be a whole number`);
    }
    read.max_tasks_per_user = maximum;
  }
  if (declared["default_timezone"]) {
    read.default_timezone = declared["default_timezone"];
  }
  return read;
}

const SANDBOX_LISTS = new Set(["setup", "allowed_domains", "network_allow_list"]);
const SANDBOX_SIZES = new Set(["cpu", "memory_gb", "disk_gb"]);
const SANDBOX_TIMEOUT_MS = 30 * 60_000;

/**
 * Reads `sandbox_options:`, how the sandbox is built and what code in it may reach:
 *
 * ```yaml
 * sandbox_options:
 *   image: python:3.13-slim-bookworm
 *   setup: [pip install bpy==5.2.2]
 *   timeout: 5m
 *   memory_gb: 4
 *   network: true
 *   allowed_domains: [api.example.com]
 * ```
 */
function sandboxOptions(lines: string[], inline: string, where: string): Schemas["SandboxOptions"] {
  if (scalar(inline) !== undefined) {
    throw new ConfigurationError(`${where}: sandbox_options should be a mapping`);
  }

  const keys = new Map<string, Written>();
  let keyColumn = -1;
  let value: Written | undefined;
  for (const line of lines) {
    const trimmed = uncomment(line).trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const column = line.length - line.trimStart().length;
    if (keyColumn < 0) {
      keyColumn = column;
    }
    if (value && (column > keyColumn || trimmed.startsWith("-"))) {
      value.nested.push(trimmed);
      continue;
    }
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(trimmed);
    if (column !== keyColumn || !entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} belongs to no sandbox_options setting`);
    }
    value = { inline: (entry[2] ?? "").trim(), nested: [] };
    keys.set(entry[1] as string, value);
  }

  const options: { -readonly [key in keyof Schemas["SandboxOptions"]]: Schemas["SandboxOptions"][key] } = {};
  for (const [key, written] of keys) {
    const name = `sandbox_options.${key}`;
    if (SANDBOX_LISTS.has(key)) {
      const items = written.inline
        ? flowList(written.inline, name, where)
        : written.nested.map((item) => listItem(item, name, where));
      options[key as "setup"] = items.filter((item) => item);
      continue;
    }
    if (written.nested.length) {
      throw new ConfigurationError(`${where}: ${name} is one value, not a list`);
    }
    const given = scalar(written.inline);
    if (given === undefined) {
      continue;
    }
    if (key === "image") {
      options.image = given;
    } else if (key === "network") {
      if (given !== "true" && given !== "false") {
        throw new ConfigurationError(`${where}: ${name} must be true or false`);
      }
      options.network = given === "true";
    } else if (key === "timeout") {
      const timeout = parseDeadline(given, where);
      if (timeout <= 0 || timeout > SANDBOX_TIMEOUT_MS) {
        throw new ConfigurationError(`${where}: ${name} must be a duration up to 30m`);
      }
      options.timeout_ms = timeout;
    } else if (SANDBOX_SIZES.has(key)) {
      if (!/^\d+$/.test(given)) {
        throw new ConfigurationError(`${where}: ${name} must be a whole number`);
      }
      options[key as "cpu"] = Number(given);
    } else {
      throw new ConfigurationError(`${where}: unknown sandbox_options setting ${JSON.stringify(key)}`);
    }
  }
  return options;
}

/** A value as agent.yaml writes it: a scalar, undefined for none, a list or a mapping. */
type Tree = string | undefined | Tree[] | Map<string, Tree>;

const FLOOR_CONTROL_KEYS = new Set(["model", "max_wait_ms"]);

/**
 * Reads `floor_control:`, the model that controls the agent's floor and how long a caller may
 * wait once they stop, as a block or on one line:
 *
 * ```yaml
 * floor_control:
 *   model: stream/chorus-asr
 *   max_wait_ms: 800
 * floor_control: {model: stream/chorus-asr}
 * ```
 *
 * Undefined when the key says nothing, which leaves the stored choice; `{}` is the default
 * flow controller again. Each key is sent as written: which models there are, and whether a
 * wait needs one, is the router's to say, and it refuses stream/chorus-asr when it has no
 * chorus-asr server.
 */
function floorControl(
  lines: string[],
  inline: string,
  where: string,
): Schemas["FloorControl"] | undefined {
  const notMapping = (): ConfigurationError =>
    new ConfigurationError(`${where}: floor_control is a mapping of model and, optionally, max_wait_ms`);
  let declared: Tree;
  if (scalar(inline) !== undefined) {
    if (lines.some((line) => !line.trim().startsWith("#"))) {
      throw notMapping();
    }
    declared = flow(inline, "floor_control", where);
  } else {
    declared = floorControlBlock(lines, where);
  }
  if (declared === undefined) {
    return undefined;
  }
  if (!(declared instanceof Map)) {
    throw notMapping();
  }
  refuseUnknown(declared.keys(), FLOOR_CONTROL_KEYS, "floor_control", where);
  const read: { -readonly [key in keyof Schemas["FloorControl"]]: Schemas["FloorControl"][key] } = {};
  const model = declared.get("model");
  if (model !== undefined) {
    if (typeof model !== "string") {
      throw new ConfigurationError(`${where}: floor_control.model must be a string`);
    }
    read.model = model;
  }
  const wait = declared.get("max_wait_ms");
  if (wait !== undefined) {
    read.max_wait_ms = whole(wait, "floor_control.max_wait_ms", where);
  }
  return read;
}

/** The block under `floor_control:`, each key's value read as written. */
function floorControlBlock(lines: string[], where: string): Map<string, Tree> | undefined {
  const declared = new Map<string, Tree>();
  let keyColumn = -1;
  let last: string | undefined;
  for (const line of lines) {
    const trimmed = uncomment(line).trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const column = line.length - line.trimStart().length;
    if (keyColumn < 0) {
      if (trimmed.startsWith("-")) {
        throw new ConfigurationError(`${where}: floor_control is a mapping of model and, optionally, max_wait_ms`);
      }
      keyColumn = column;
    }
    // A list or a mapping under model or max_wait_ms is not the one value either one is.
    if (last !== undefined && (column > keyColumn || trimmed.startsWith("-"))) {
      throw new ConfigurationError(`${where}: floor_control.${last} is one value, not a list or a mapping`);
    }
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(trimmed);
    if (column !== keyColumn || !entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} belongs to no floor_control setting`);
    }
    last = entry[1] as string;
    refuseUnknown([last], FLOOR_CONTROL_KEYS, "floor_control", where);
    declared.set(last, flow((entry[2] ?? "").trim(), `floor_control.${last}`, where));
  }
  return keyColumn < 0 ? undefined : declared;
}

/**
 * Reads a value written in YAML's flow style, `{from_ms: 0, bar: 0.97}` or `[one, two]`,
 * nested as deep as it goes, or a scalar on its own.
 */
function flow(text: string, key: string, where: string): Tree {
  let at = 0;
  const unreadable = (): ConfigurationError =>
    new ConfigurationError(`${where}: ${key} cannot be read from ${JSON.stringify(text)}`);
  const space = (): void => {
    while (/\s/.test(text.charAt(at))) {
      at++;
    }
  };
  // A scalar runs to whatever closes what it is in, unless it is quoted.
  const word = (ends: string): string => {
    space();
    const start = at;
    const quote = text.charAt(at);
    if (quote === '"' || quote === "'") {
      const close = text.indexOf(quote, at + 1);
      if (close < 0) {
        throw unreadable();
      }
      at = close + 1;
      return text.slice(start, at);
    }
    while (at < text.length && !ends.includes(text.charAt(at))) {
      at++;
    }
    return text.slice(start, at).trim();
  };
  const value = (ends: string): Tree => {
    space();
    const open = text.charAt(at);
    if (open !== "{" && open !== "[") {
      return scalar(word(ends));
    }
    at++;
    const close = open === "{" ? "}" : "]";
    const mapped = new Map<string, Tree>();
    const items: Tree[] = [];
    space();
    while (text.charAt(at) !== close) {
      if (open === "[") {
        items.push(value(",]"));
      } else {
        const name = scalar(word(":,}"));
        if (!name || text.charAt(at) !== ":") {
          throw unreadable();
        }
        at++;
        mapped.set(name, value(",}"));
      }
      space();
      if (text.charAt(at) === ",") {
        at++;
        space();
      } else if (text.charAt(at) !== close) {
        throw unreadable();
      }
    }
    at++;
    return open === "{" ? mapped : items;
  };

  const read = value("");
  space();
  if (at < text.length) {
    throw unreadable();
  }
  return read;
}

function refuseUnknown(keys: Iterable<string>, known: Set<string>, name: string, where: string): void {
  for (const key of keys) {
    if (!known.has(key)) {
      throw new ConfigurationError(`${where}: unknown ${name} setting ${JSON.stringify(key)}`);
    }
  }
}

/** A whole number, as written: whether it is in range is the router's to say. */
function whole(value: Tree, name: string, where: string): number {
  if (typeof value !== "string" || !/^-?\d+$/.test(value)) {
    throw new ConfigurationError(`${where}: ${name} must be a whole number`);
  }
  return Number(value);
}

function connectorOf(
  keys: Map<string, { inline: string; nested: string[] }>,
  alias: string | undefined,
  where: string,
): ConnectorDeclaration {
  for (const key of keys.keys()) {
    if (!CONNECTOR_KEYS.has(key)) {
      throw new ConfigurationError(
        `${where}: ${JSON.stringify(key)} is not something a connector says; name, connector, tools, connection, required, sandbox and require_approval are`,
      );
    }
  }
  const one = (key: string): string | undefined => {
    const written = keys.get(key);
    if (written?.nested.length) {
      throw new ConfigurationError(`${where}: a connector's ${key} is one value, not a list`);
    }
    return written ? scalar(written.inline) : undefined;
  };

  const name = alias === undefined ? one("name") : scalar(alias);
  if (!name) {
    throw new ConfigurationError(`${where}: a connector needs a name`);
  }
  const declared: ConnectorDeclaration = { name };
  const connector = one("connector");
  if (connector) {
    declared.connector = connector;
  }

  const tools = keys.get("tools");
  if (tools && (tools.nested.length > 0 || scalar(tools.inline) !== undefined)) {
    const listed = tools.nested.length
      ? tools.nested.map((item) => listItem(item, "tools", where))
      : flowList(tools.inline, "tools", where);
    declared.tools = listed.filter((item) => item);
    if (declared.tools.length === 0) {
      throw new ConfigurationError(
        `${where}: connector ${JSON.stringify(name)}: tools: [] grants nothing; leave tools out for every tool`,
      );
    }
  }

  const connection = one("connection");
  if (connection !== undefined) {
    if (!connection) {
      throw new ConfigurationError(
        `${where}: connector ${JSON.stringify(name)}: connection must be a connection id or label`,
      );
    }
    declared.connection = connection;
  }

  const required = one("required");
  if (required !== undefined) {
    if (required !== "true" && required !== "false") {
      throw new ConfigurationError(
        `${where}: connector ${JSON.stringify(name)}: required must be true or false`,
      );
    }
    declared.required = required === "true";
  }

  const sandbox = one("sandbox");
  if (sandbox !== undefined && sandbox !== "true" && sandbox !== "false") {
    throw new ConfigurationError(`${where}: connector ${JSON.stringify(name)}: sandbox must be true or false`);
  }
  if (sandbox === "true") {
    declared.sandbox = true;
  }

  const approval = keys.get("require_approval");
  if (approval && (approval.nested.length > 0 || scalar(approval.inline) !== undefined)) {
    const listed = approval.nested.length
      ? approval.nested.map((item) => listItem(item, "require_approval", where))
      : flowList(approval.inline, "require_approval", where);
    declared.requireApproval = listed.filter((item) => item);
  }
  return declared;
}

function assign(declared: Declaration, key: string, value: string, where: string): void {
  switch (key) {
    case "mode":
      declared.mode = value as Schemas["AgentMode"];
      break;
    case "sandbox":
      declared.sandbox = value as Schemas["Sandbox"];
      break;
    case "harness":
      declared.harness = value as Schemas["Harness"];
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

/** A whole number of at least 1, which is every count a declaration takes. */
function count(value: string, key: string, where: string): number {
  const counted = Number(value);
  if (!/^\d+$/.test(value) || counted < 1) {
    throw new ConfigurationError(`${where}: ${key} is a whole number of at least 1`);
  }
  return counted;
}

/**
 * A `|` block keeps its line breaks and a `>` block folds them into spaces, a blank line
 * still breaking. Either ends in one newline, or none when written `|-` or `>-`.
 */
function block(lines: string[], style: string): string {
  const written = [...lines];
  while (written.length > 0 && !(written.at(-1) as string).trim()) {
    written.pop();
  }
  const indent = Math.min(
    ...written.filter((one) => one.trim()).map((one) => one.length - one.trimStart().length),
  );
  const body = written.map((one) => (one.trim() ? one.slice(indent) : ""));

  let text = body.join("\n");
  if (style.startsWith(">")) {
    text = body.reduce(
      (folded, one, at) =>
        at === 0 ? one : one === "" ? `${folded}\n` : `${folded}${body[at - 1] === "" ? "" : " "}${one}`,
      "",
    );
  }
  return style.endsWith("-") || !text ? text : `${text}\n`;
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

/** Each setting is `enabled` or `disabled`, passed through for the router to check. */
function dispatch(declared: Record<string, string>, where: string): Schemas["AgentDispatch"] {
  for (const key of Object.keys(declared)) {
    if (key !== "incoming_call" && key !== "text") {
      throw new ConfigurationError(`${where}: unknown dispatch setting ${JSON.stringify(key)}`);
    }
  }
  const incoming = declared["incoming_call"] as Schemas["DispatchSetting"] | "" | undefined;
  const text = declared["text"] as Schemas["DispatchSetting"] | "" | undefined;
  return { ...(incoming ? { incoming_call: incoming } : {}), ...(text ? { text } : {}) };
}

/** What the agent says as it joins: `text`, and a `mode` of `exact` or `variation`. */
function greeting(declared: Record<string, string>, where: string): Schemas["Greeting"] {
  for (const key of Object.keys(declared)) {
    if (key !== "text" && key !== "mode") {
      throw new ConfigurationError(`${where}: unknown greeting setting ${JSON.stringify(key)}`);
    }
  }
  if (!declared["text"]) {
    throw new ConfigurationError(`${where}: a greeting needs text`);
  }
  const mode = declared["mode"] as Schemas["GreetingMode"] | "" | undefined;
  return { text: declared["text"], ...(mode ? { mode } : {}) };
}

const DEFAULT_UI_COMPONENTS = "ui/components.json";
const DEFAULT_UI_BLUEPRINTS = "ui/blueprints.yaml";
const MAX_UI_FILE_BYTES = 512 << 10;
const UI_KEYS = new Set([
  "format",
  "keep",
  "components",
  "blueprints",
  "patch_mode",
  "voice_ui",
  "rules",
]);

function parseUIDeclaration(indented: string[], inline: string, where: string): UIDeclaration {
  if (inline) {
    throw new ConfigurationError(`${where}: ui should be a mapping`);
  }
  const out: UIDeclaration = {};
  const lines = indented.map((line) => line.replace(/\r$/, ""));

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i]!;
    const trimmed = raw.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const column = raw.length - raw.trimStart().length;
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(trimmed);
    if (!entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} is not a ui setting`);
    }
    const key = entry[1]!;
    const val = uncomment(entry[2] ?? "").trim();
    if (!UI_KEYS.has(key)) {
      throw new ConfigurationError(
        `${where}: ${JSON.stringify(key)} is not something ui says; ${[...UI_KEYS].join(", ")} are`,
      );
    }
    const children: string[] = [];
    while (i + 1 < lines.length) {
      const next = lines[i + 1]!;
      if (next.trim() && next.length - next.trimStart().length <= column) {
        break;
      }
      children.push(next.trim());
      i++;
    }
    const items = children.filter((c) => c && !c.startsWith("#"));

    switch (key) {
      case "format": {
        const fmt = scalar(val);
        if (fmt !== undefined) {
          if (fmt !== "openui" && fmt !== "a2ui") {
            throw new ConfigurationError(`${where}: ui.format must be openui or a2ui, not ${JSON.stringify(fmt)}`);
          }
          out.format = fmt;
        }
        break;
      }
      case "components": {
        const comp = scalar(val);
        if (comp !== undefined) {
          out.components = comp;
        }
        break;
      }
      case "blueprints": {
        const bp = scalar(val);
        if (bp !== undefined) {
          out.blueprints = bp;
        }
        break;
      }
      case "patch_mode":
        out.patchMode = uiBool(val, "ui.patch_mode", where);
        break;
      case "voice_ui":
        out.voiceUI = uiBool(val, "ui.voice_ui", where);
        break;
      case "keep": {
        const list = val ? flowList(val, "ui.keep", where) : items.map((c) => listItem(c, "ui.keep", where));
        out.keep = list.filter(Boolean);
        break;
      }
      case "rules": {
        const list = val ? flowList(val, "ui.rules", where) : items.map((c) => listItem(c, "ui.rules", where));
        out.rules = list.filter(Boolean);
        break;
      }
    }
  }
  return out;
}

function uiBool(raw: string, key: string, where: string): boolean {
  const val = scalar(raw);
  if (val !== "true" && val !== "false") {
    throw new ConfigurationError(`${where}: ${key} must be true or false`);
  }
  return val === "true";
}

async function loadUIPrompt(root: string, ui: UIDeclaration): Promise<string> {
  let components: UIPortableComponent[] = [];
  const compRel = ui.components || ((await existsInside(root, DEFAULT_UI_COMPONENTS)) ? DEFAULT_UI_COMPONENTS : "");
  if (compRel) {
    const { text, full } = await readInsideFolder(root, compRel);
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (cause) {
      throw new ConfigurationError(`${full}: invalid JSON: ${cause instanceof Error ? cause.message : String(cause)}`);
    }
    if (!Array.isArray(parsed)) {
      throw new ConfigurationError(`${full}: expected an array of portable UI components`);
    }
    components = parsed as UIPortableComponent[];
  }

  let blueprints: UIBlueprint[] = [];
  const bpRel = ui.blueprints || ((await existsInside(root, DEFAULT_UI_BLUEPRINTS)) ? DEFAULT_UI_BLUEPRINTS : "");
  if (bpRel) {
    const { text, full } = await readInsideFolder(root, bpRel);
    blueprints = parseBlueprints(text, full);
  }

  return compileUIPrompt(ui, components, blueprints, join(root, AGENT_FILE));
}

async function readInsideFolder(root: string, rel: string): Promise<{ text: string; full: string }> {
  if (isAbsolute(rel)) {
    throw new ConfigurationError(`ui path ${JSON.stringify(rel)} must be relative to the agent directory`);
  }
  const rootReal = await realpath(root);
  const full = join(root, ...rel.split("/"));
  const targetReal = await realpath(full).catch(missing);
  if (!targetReal) {
    throw new ConfigurationError(`${full} does not exist`);
  }
  const relCheck = relative(rootReal, targetReal);
  if (relCheck === ".." || relCheck.startsWith(`..${sep}`) || isAbsolute(relCheck)) {
    throw new ConfigurationError(`ui path ${JSON.stringify(rel)} leaves the agent directory`);
  }
  const info = await stat(targetReal);
  if (info.size > MAX_UI_FILE_BYTES) {
    throw new ConfigurationError(`${full} exceeds ${MAX_UI_FILE_BYTES} bytes`);
  }
  return { text: await readFile(targetReal, "utf8"), full };
}

async function existsInside(root: string, rel: string): Promise<boolean> {
  try {
    await readInsideFolder(root, rel);
    return true;
  } catch {
    return false;
  }
}

export function parseBlueprints(content: string, where = DEFAULT_UI_BLUEPRINTS): UIBlueprint[] {
  const lines = content.split("\n").map((l) => l.replace(/\r$/, ""));
  const out: UIBlueprint[] = [];
  const seen = new Set<string>();
  let current: Partial<UIBlueprint> | undefined;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    let body = trimmed;
    let column = line.length - line.trimStart().length;
    if (trimmed === "-" || trimmed.startsWith("- ")) {
      current = {};
      out.push(current as UIBlueprint);
      body = trimmed.slice(1).trim();
      column += trimmed.length - body.length;
      if (!body) {
        continue;
      }
    }
    if (!current) {
      throw new ConfigurationError(`${where}: blueprints must be a YAML list starting with "- "`);
    }
    const entry = /^([A-Za-z_][\w-]*):(?:\s+(.*))?$/.exec(body);
    if (!entry) {
      throw new ConfigurationError(`${where}: ${JSON.stringify(trimmed)} is not a key and value`);
    }
    const key = entry[1]!;
    const inline = (entry[2] ?? "").trim();
    const nested: string[] = [];
    while (i + 1 < lines.length) {
      const next = lines[i + 1]!;
      if (next.trim() && next.length - next.trimStart().length <= column) {
        break;
      }
      nested.push(next);
      i++;
    }
    const written = nested.map((n) => n.trim()).filter((n) => n && !n.startsWith("#"));

    switch (key) {
      case "name":
        current.name = scalar(uncomment(inline)) ?? "";
        break;
      case "when":
        current.when = /^[|>]-?$/.test(inline) ? block(nested, inline).trim() : (scalar(uncomment(inline)) ?? "");
        break;
      case "example":
        current.example = /^[|>]-?$/.test(inline) ? block(nested, inline).trim() : (scalar(uncomment(inline)) ?? "");
        break;
      case "order":
        current.order = inline ? flowList(inline, "order", where) : written.map((w) => listItem(w, "order", where));
        break;
      case "rules":
        current.rules = inline ? flowList(inline, "rules", where) : written.map((w) => listItem(w, "rules", where));
        break;
      default:
        throw new ConfigurationError(`${where}: unknown blueprint key ${JSON.stringify(key)}`);
    }
  }

  return out.map((bp) => {
    if (!bp.name) {
      throw new ConfigurationError(`${where}: a blueprint needs a name`);
    }
    if (seen.has(bp.name)) {
      throw new ConfigurationError(`${where}: duplicate blueprint ${JSON.stringify(bp.name)}`);
    }
    seen.add(bp.name);
    if (!bp.when) {
      throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} needs a when condition`);
    }
    if (!bp.order || bp.order.length === 0) {
      throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} needs an order list`);
    }
    return bp as UIBlueprint;
  });
}

/** `.agent_sync`: the fingerprint a directory was last synced under, and when. */
function stampAt(path: string): SyncStamp {
  return {
    async read() {
      const content = await readFile(path, "utf8").catch(missing);
      let recorded: unknown;
      try {
        recorded = content === undefined ? undefined : JSON.parse(content);
      } catch {
        return "";
      }
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
  return (await readFile(path, "utf8").catch(missing)) ?? "";
}

function missing(cause: unknown): undefined {
  if (cause instanceof Error && "code" in cause && cause.code === "ENOENT") {
    return undefined;
  }
  throw cause;
}

/** A hash in a quoted scalar is text, not the start of a YAML comment. */
function uncomment(line: string): string {
  let quote = "";
  for (let index = 0; index < line.length; index++) {
    const char = line[index];
    if (quote) {
      if (char === "\\" && quote === '"') {
        index++;
      } else if (char === quote) {
        if (quote === "'" && line[index + 1] === "'") {
          index++;
        } else {
          quote = "";
        }
      }
    } else if ((char === '"' || char === "'") &&
      (index === 0 || /\s/.test(line[index - 1]!) || "[{,:".includes(line[index - 1]!))) {
      quote = char;
    } else if (char === "#" && (index === 0 || /\s/.test(line[index - 1]!))) {
      return line.slice(0, index);
    }
  }
  return line;
}
