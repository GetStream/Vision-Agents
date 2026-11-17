import { ConfigurationError } from "./errors.js";
import { A2UI_BASIC_CATALOG, A2UI_PROTOCOL, A2UI_WRITING, type A2UISchema } from "./generated/a2ui.js";
import type { UIBlueprint, UIDeclaration, UIPortableComponent } from "./ui-catalog.js";
import { checkBlueprint, extractFence, uiOpening, uiPascalName, uiRules, UI_BLUEPRINTS_HEADING } from "./ui-prompt.js";

// The compiler for `ui.format: a2ui`, the same as the Go SDK's (sdks/go/agents/ui_a2ui.go): the
// prompt stream-chat-swift's A2UI renderer writes for itself, so the agent is told what the apps
// draw. Both are held to sdks/testdata/ui/concierge-a2ui.instructions.txt.

const A2UI_FENCE = "ui:a2ui@1.0";
/**
 * The id the apps' A2UI renderers give the basic catalog with the app's components beside its own
 * (A2UIRenderer.defaultCatalogId in stream-chat-swift).
 */
const A2UI_CATALOG_ID = "getstream.io:app-catalog";
/** How deep a renderer draws components (A2UIProtocol.maxNestingDepth). */
const A2UI_MAX_NESTING = 50;

export const A2UI_IDENTIFIER = /^[\p{L}_][\p{L}\p{N}_]*$/u;

const A2UI_MESSAGE_KINDS = [
  "createSurface",
  "updateComponents",
  "updateDataModel",
  "deleteSurface",
  "callRendererFunction",
  "agentFunctionResponse",
];

type JSONObject = { [key: string]: unknown };
type Components = Map<string, JSONObject>;

/**
 * What the model is told to write UI in A2UI with: the renderer's protocol and guidelines, then the
 * catalog it draws with, the basic catalog's components and functions that keep names beside the
 * app's own, as the renderer serializes it.
 */
export function compileA2UIPrompt(
  settings: UIDeclaration,
  components: readonly UIPortableComponent[],
  blueprints: readonly UIBlueprint[],
  where: string,
): string {
  const custom = new Map<string, A2UISchema>();
  for (const comp of components) {
    custom.set(uiPascalName(comp.name), a2uiComponentSchema(comp));
  }

  const basicComponents = new Map(Object.entries(A2UI_BASIC_CATALOG.components));
  const basicFunctions = new Map(Object.entries(A2UI_BASIC_CATALOG.functions));
  const catalogComponents = new Map<string, A2UISchema>();
  const catalogFunctions = new Map<string, A2UISchema>();
  if (!settings.keep || settings.keep.length === 0) {
    basicComponents.forEach((schema, name) => catalogComponents.set(name, schema));
    basicFunctions.forEach((schema, name) => catalogFunctions.set(name, schema));
  }
  for (const name of settings.keep ?? []) {
    const component = basicComponents.get(name);
    const fn = basicFunctions.get(name);
    if (component) {
      catalogComponents.set(name, component);
    } else if (fn) {
      catalogFunctions.set(name, fn);
    } else if (!custom.has(name)) {
      throw new ConfigurationError(
        `${where}: ui.keep names ${JSON.stringify(name)}, which is neither a component nor a function of the A2UI basic catalog`,
      );
    }
  }
  custom.forEach((schema, name) => catalogComponents.set(name, schema));

  for (const bp of blueprints) {
    checkBlueprint(bp, where);
    if (!bp.example) {
      continue;
    }
    const problems = checkA2UIProgram(extractFence(bp.example, A2UI_FENCE), catalogComponents);
    if (problems.length > 0) {
      throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} example is invalid: ${problems.join("; ")}`);
    }
  }

  let out = `${uiOpening(A2UI_FENCE, settings.voiceUI).join("\n")}\n`;
  out += `${A2UI_WRITING.trim()}\n\n`;
  out += `${a2uiSubsection(A2UI_PROTOCOL)}\n\n`;
  out += `${a2uiSubsection(A2UI_BASIC_CATALOG.instructions)}\n\n`;
  out += `### The catalog \`${A2UI_CATALOG_ID}\`\n\n\`\`\`json\n${a2uiCatalogDocument(catalogComponents, catalogFunctions)}\n\`\`\`\n`;

  if (blueprints.length > 0) {
    out += `${UI_BLUEPRINTS_HEADING.join("\n")}\n`;
    for (const bp of blueprints) {
      out += `- **${bp.name}** (when ${bp.when.trim()}): a Column of \`${bp.order.join(" -> ")}\``;
      if (bp.rules && bp.rules.length > 0) {
        out += ` — ${bp.rules.join("; ")}`;
      }
      out += "\n";
      if (bp.example) {
        out += `\n\`\`\`${A2UI_FENCE}\n${extractFence(bp.example, A2UI_FENCE).trim()}\n\`\`\`\n`;
      }
    }
  }

  out += uiRules(settings.rules).join("\n");

  return out.trim();
}

/** Nests the renderer's text under the section's own heading. */
function a2uiSubsection(text: string): string {
  return text
    .trim()
    .split("\n")
    .map((line) => {
      if (line.startsWith("## ")) {
        return `### ${line.slice(3)}`;
      }
      if (line.startsWith("# ")) {
        return `### ${line.slice(2)}`;
      }
      return line;
    })
    .join("\n");
}

/**
 * An app component as the renderer has it in its catalog (PortableComponent in stream-chat-swift):
 * its parameters are its properties, and what its events do joins its description, as the
 * component reports them itself.
 */
function a2uiComponentSchema(comp: UIPortableComponent): A2UISchema {
  const properties: JSONObject = { component: { const: uiPascalName(comp.name) } };
  for (const [key, prop] of Object.entries(comp.parameters?.properties ?? {})) {
    const schema: JSONObject = {};
    if (prop.type) {
      schema.type = prop.type;
    }
    if (prop.format) {
      schema.format = prop.format;
    }
    if (prop.description) {
      schema.description = prop.description;
    }
    if (prop.enum && prop.enum.length > 0) {
      schema.enum = prop.enum;
    }
    // Go writes the numbers in items as they were written (1.0 stays 1.0); JSON.parse has already
    // read them as numbers, so 1.0 is written 1.
    if (prop.items !== undefined) {
      schema.items = prop.items;
    }
    properties[key] = schema;
  }
  const required = ["component", ...(comp.parameters?.required ?? [])].sort();

  const notes = [comp.description, ...(comp.events ?? []).map((event) => event.description ?? "")]
    .map(a2uiSentence)
    .filter(Boolean);
  return { description: notes.join(" "), properties, required, type: "object" };
}

function a2uiSentence(text: string): string {
  const trimmed = text.trim();
  if (trimmed === "" || ".!?".includes(trimmed.at(-1)!)) {
    return trimmed;
  }
  return `${trimmed}.`;
}

/** The catalog as the renderer serializes it, keys sorted, on one line. */
function a2uiCatalogDocument(components: ReadonlyMap<string, A2UISchema>, functions: ReadonlyMap<string, A2UISchema>): string {
  const document: JSONObject = {
    $defs: {
      anyComponent: {
        discriminator: { propertyName: "component" },
        oneOf: [...components.keys()].sort().map((name) => ({ $ref: `#/components/${name}` })),
      },
      anyFunction: { oneOf: [...functions.keys()].sort().map((name) => ({ $ref: `#/functions/${name}` })) },
    },
    $id: A2UI_CATALOG_ID,
    $schema: A2UI_BASIC_CATALOG.$schema,
    catalogId: A2UI_CATALOG_ID,
    description: A2UI_BASIC_CATALOG.description,
    protocolVersion: "1.0",
    title: A2UI_BASIC_CATALOG.title,
  };
  if (components.size > 0) {
    document.components = Object.fromEntries(components);
  }
  if (functions.size > 0) {
    document.functions = Object.fromEntries(functions);
  }
  return stringifySorted(document);
}

/**
 * JSON on one line with every object's keys sorted, as Go writes a map. Go also reads a lone
 * surrogate as U+FFFD and escapes U+2028 and U+2029, which JSON.stringify writes as they are.
 */
function stringifySorted(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) => {
    if (typeof item === "string") {
      return wellFormed(item);
    }
    if (isObject(item)) {
      const entries = Object.keys(item).map((key): [string, unknown] => [wellFormed(key), item[key]]);
      return Object.fromEntries(entries.sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
    }
    return item;
  }).replace(/[\u2028\u2029]/g, (separator) => (separator === "\u2028" ? "\\u2028" : "\\u2029"));
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

function wellFormed(text: string): string {
  return text.replace(LONE_SURROGATE, "\uFFFD");
}

/**
 * What is wrong with the A2UI messages of an example for a renderer with catalog, as
 * stream-chat-swift's A2UIProgram finds it: messages it can't read, components it doesn't have,
 * and references that lead nowhere. An example teaches the model, so it has to render as written.
 */
function checkA2UIProgram(text: string, catalog: ReadonlyMap<string, A2UISchema>): string[] {
  const problems: string[] = [];
  const surfaces = new Map<string, Components>();
  const order: string[] = [];
  for (const raw of a2uiMessages(text)) {
    let message: unknown;
    try {
      message = JSON.parse(raw);
    } catch {
      problems.push(`this isn't JSON: ${truncate(raw, 80)}`);
      continue;
    }
    const read = a2uiMessageOf(message as JSONObject);
    if (typeof read === "string") {
      problems.push(read);
      continue;
    }
    const { kind, body } = read;
    const surfaceId = typeof body.surfaceId === "string" ? body.surfaceId : "";
    switch (kind) {
      case "createSurface": {
        const { components, problems: more } = a2uiComponentsOf(body, surfaceId);
        problems.push(...more);
        if (typeof body.catalogId === "string" && body.catalogId !== A2UI_CATALOG_ID) {
          problems.push(`surface ${surfaceId} names the catalog ${body.catalogId}, not ${A2UI_CATALOG_ID}`);
        }
        surfaces.set(surfaceId, components);
        order.push(surfaceId);
        break;
      }
      case "updateComponents": {
        const existing = surfaces.get(surfaceId);
        if (!existing) {
          problems.push(`surface ${surfaceId} gets components before it is created`);
          continue;
        }
        const { components, problems: more } = a2uiComponentsOf(body, surfaceId);
        problems.push(...more);
        components.forEach((component, id) => existing.set(id, component));
        break;
      }
      case "updateDataModel":
        if (!surfaces.has(surfaceId)) {
          problems.push(`surface ${surfaceId} gets data before it is created`);
        }
        break;
      case "deleteSurface":
        break;
      default:
        problems.push("a reply's UI calls no functions");
    }
  }
  if (order.length === 0) {
    problems.push("nothing renders: there is no createSurface");
  }
  for (const surfaceId of order) {
    problems.push(...a2uiSurfaceProblems(surfaces.get(surfaceId)!, surfaceId, catalog));
  }
  return problems;
}

/** The JSON objects in text, as the renderer reads a reply's block. */
function a2uiMessages(text: string): string[] {
  const messages: string[] = [];
  let depth = 0;
  let start = 0;
  let inString = false;
  let escaped = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c === "\\") {
        escaped = true;
      } else if (c === '"') {
        inString = false;
      }
      continue;
    }
    if (c === '"') {
      inString = depth > 0;
    } else if (c === "{") {
      if (depth === 0) {
        start = i;
      }
      depth++;
    } else if (c === "}" && depth > 0) {
      depth--;
      if (depth === 0) {
        messages.push(text.slice(start, i + 1));
      }
    }
  }
  return messages;
}

function a2uiMessageOf(message: JSONObject): { kind: string; body: JSONObject } | string {
  const exactlyOne = `a message has exactly one of ${A2UI_MESSAGE_KINDS.join(", ")}`;
  if (message.version !== "1.0") {
    return 'a message needs "version": "1.0"';
  }
  let kind = "";
  for (const key of Object.keys(message)) {
    if (key === "version") {
      continue;
    }
    if (!A2UI_MESSAGE_KINDS.includes(key) || kind !== "") {
      return exactlyOne;
    }
    kind = key;
  }
  if (kind === "") {
    return exactlyOne;
  }
  const body = message[kind];
  if (!isObject(body)) {
    return `${kind} is an object`;
  }
  const surfaceId = typeof body.surfaceId === "string" ? body.surfaceId : "";
  if (surfaceId === "" && kind !== "callRendererFunction" && kind !== "agentFunctionResponse") {
    return `${kind} needs a surfaceId`;
  }
  return { kind, body };
}

/** A message's components by id, and the problems with them. */
function a2uiComponentsOf(body: JSONObject, surfaceId: string): { components: Components; problems: string[] } {
  const listed = Array.isArray(body.components) ? body.components : [];
  const components: Components = new Map();
  const problems: string[] = [];
  const repeated: string[] = [];
  for (const item of listed) {
    const component = isObject(item) ? item : {};
    const id = typeof component.id === "string" ? component.id : "";
    if (id === "" || typeof component.component !== "string") {
      problems.push(`surface ${surfaceId} has a component without an id and a component type`);
      continue;
    }
    if (components.has(id) && !repeated.includes(id)) {
      repeated.push(id);
    }
    components.set(id, component);
  }
  for (const id of repeated) {
    problems.push(`surface ${surfaceId} defines ${id} more than once in one message`);
  }
  return { components, problems };
}

function a2uiSurfaceProblems(
  components: Components,
  surfaceId: string,
  catalog: ReadonlyMap<string, A2UISchema>,
): string[] {
  const problems: string[] = [];
  const references = new Map<string, string[]>();
  for (const id of [...components.keys()].sort()) {
    const component = components.get(id)!;
    const kind = component.component as string;
    if (!A2UI_IDENTIFIER.test(id)) {
      problems.push(`${id} isn't an identifier: use letters, digits and _, not starting with a digit`);
    }
    const schema = catalog.get(kind);
    if (!schema) {
      problems.push(`${id} is a ${kind}, which the catalog doesn't have`);
      continue;
    }
    const properties = (isObject(schema.properties) ? schema.properties : {}) as Record<string, JSONObject>;
    const required = Array.isArray(schema.required) ? (schema.required as string[]) : [];
    for (const name of required) {
      if (!Object.hasOwn(component, name) && name !== "component") {
        problems.push(`${id} is a ${kind} without its ${name}`);
      }
    }
    for (const key of Object.keys(component).sort()) {
      if (["id", "component", "catalogId", "checks", "weight"].includes(key)) {
        continue;
      }
      if (!Object.hasOwn(properties, key)) {
        problems.push(`${id} is a ${kind}, which has no ${key}`);
      }
    }
    const held: string[] = [];
    for (const key of Object.keys(properties).sort()) {
      held.push(...a2uiReferences(component[key], properties[key]!));
    }
    references.set(id, held);
  }
  for (const id of [...references.keys()].sort()) {
    for (const child of references.get(id)!) {
      if (!components.has(child)) {
        problems.push(`${id} holds ${child}, which is not defined`);
      }
    }
  }
  if (!components.has("root")) {
    problems.push(`surface ${surfaceId} has no component with the id root`);
    return problems;
  }
  const reached = new Set<string>();
  const next = ["root"];
  while (next.length > 0) {
    const id = next.pop()!;
    if (reached.has(id)) {
      continue;
    }
    reached.add(id);
    next.push(...(references.get(id) ?? []));
  }
  for (const id of [...components.keys()].sort()) {
    if (!reached.has(id)) {
      problems.push(`${id} is defined but nothing holds it`);
    }
  }
  problems.push(...a2uiNestingProblems(references, surfaceId));
  return problems;
}

/**
 * The ids of the components value holds, by what schema says it is: a ComponentId, a ChildList of
 * ids or a template, or an array or object of them.
 */
function a2uiReferences(value: unknown, schema: JSONObject): string[] {
  if (typeof schema.$ref === "string") {
    if (schema.$ref.endsWith("/ComponentId")) {
      return typeof value === "string" ? [value] : [];
    }
    if (schema.$ref.endsWith("/ChildList")) {
      if (Array.isArray(value)) {
        return value.every((item) => typeof item === "string") ? (value as string[]) : [];
      }
      if (isObject(value) && typeof value.componentId === "string") {
        return [value.componentId];
      }
      return [];
    }
  }
  if (isObject(schema.items) && Array.isArray(value)) {
    const items = schema.items;
    return value.flatMap((item) => a2uiReferences(item, items));
  }
  if (isObject(schema.properties) && isObject(value)) {
    const properties = schema.properties;
    return Object.keys(properties)
      .sort()
      .flatMap((key) => {
        const property = properties[key];
        return isObject(property) ? a2uiReferences(value[key], property) : [];
      });
  }
  return [];
}

/** A component that holds itself, which never finishes rendering, and a tree deeper than a renderer draws. */
function a2uiNestingProblems(references: ReadonlyMap<string, string[]>, surfaceId: string): string[] {
  const problems: string[] = [];
  const depths = new Map<string, number>();
  const visiting = new Set<string>();
  let reportedCycle = false;
  const depth = (id: string): number => {
    const known = depths.get(id);
    if (known !== undefined) {
      return known;
    }
    if (visiting.has(id)) {
      if (!reportedCycle) {
        problems.push(`${id} holds itself through its children, so it never finishes rendering`);
        reportedCycle = true;
      }
      return 0;
    }
    visiting.add(id);
    let deepest = 0;
    for (const child of references.get(id) ?? []) {
      if (references.has(child)) {
        deepest = Math.max(deepest, depth(child));
      }
    }
    visiting.delete(id);
    depths.set(id, deepest + 1);
    return deepest + 1;
  };
  if (references.has("root") && depth("root") > A2UI_MAX_NESTING) {
    problems.push(`surface ${surfaceId} nests components deeper than ${A2UI_MAX_NESTING} levels`);
  }
  for (const id of [...references.keys()].sort()) {
    depth(id);
  }
  return problems;
}

function isObject(value: unknown): value is JSONObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function truncate(text: string, limit: number): string {
  const chars = Array.from(text);
  return chars.length <= limit ? text : chars.slice(0, limit).join("");
}
