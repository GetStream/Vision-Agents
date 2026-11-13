import { ConfigurationError } from "./errors.js";

/** Declarative `ui:` configuration in `agent.yaml`. */
export interface UIDeclaration {
  format?: "openui" | "a2ui";
  keep?: string[];
  components?: string;
  blueprints?: string;
  patchMode?: boolean;
  voiceUI?: boolean;
  rules?: string[];
}

/** Portable cross-SDK component definition (`ui/components.json`). */
export interface UIPortableComponent {
  name: string;
  description: string;
  parameters: {
    type: string;
    properties?: Record<string, { type: string; format?: string; description?: string; enum?: string[] }>;
    required?: string[];
    "x-order"?: string[];
  };
  events?: { name: string; description?: string }[];
  slots?: string[];
}

/** Canonical layout recipe (`ui/blueprints.yaml`). */
export interface UIBlueprint {
  name: string;
  when: string;
  order: string[];
  rules?: string[];
  example?: string;
}

interface BuiltinComponent {
  readonly name: string;
  readonly signature: string;
  readonly maxArgs: number;
  readonly deps?: readonly string[];
}

const BUILTINS: readonly BuiltinComponent[] = [
  { name: "Card", signature: "Card(children: Component[], sources?: {title: string, sourceName: string, url?: string}[]) — Root vertical card container.", maxArgs: 2 },
  { name: "CardHeader", signature: "CardHeader(title?: string, subtitle?: string) — Primary card heading and subtitle.", maxArgs: 2 },
  { name: "InlineHeader", signature: "InlineHeader(heading: string, description?: string) — Compact section label.", maxArgs: 2 },
  { name: "TextContent", signature: 'TextContent(text: string, size?: "small" | "default" | "large" | "small-heavy" | "large-heavy") — Markdown-capable text block.', maxArgs: 2 },
  { name: "Callout", signature: 'Callout(variant: "info" | "warning" | "error" | "success" | "neutral", title: string, description: string) — Highlighted status callout.', maxArgs: 4 },
  { name: "Separator", signature: 'Separator(orientation?: "horizontal" | "vertical") — Hairline divider.', maxArgs: 2 },
  { name: "TagBlock", signature: 'TagBlock(tags: string[], size?: "sm" | "md" | "lg") — Compact pill row (at most 3 tags).', maxArgs: 2 },
  { name: "EntityList", signature: 'EntityList(rows: {left: string, right: string}[], size?: "small" | "default") — Key/value summary rows.', maxArgs: 4 },
  { name: "Table", signature: "Table(columns: Col[]) — Column-oriented read-only data table.", maxArgs: 1, deps: ["Col"] },
  { name: "Col", signature: 'Col(label: string, data: (string | number)[], type?: "string" | "number") — Table column.', maxArgs: 3 },
  { name: "BarChart", signature: 'BarChart(labels: string[], series: Series[], variant?: "grouped" | "stacked", xLabel?: string, yLabel?: string) — Categorical bar chart.', maxArgs: 6, deps: ["Series"] },
  { name: "Series", signature: "Series(category: string, values: number[]) — Named chart data series.", maxArgs: 2 },
  { name: "PieChart", signature: 'PieChart(labels: string[], values: number[], variant?: "pie" | "donut", appearance?: "circular" | "semiCircular") — Proportional or semicircular gauge chart.', maxArgs: 4 },
  { name: "Tabs", signature: "Tabs(items: TabItem[]) — Multi-view tab container.", maxArgs: 1, deps: ["TabItem"] },
  { name: "TabItem", signature: "TabItem(value: string, trigger: string, content: Component[]) — One tab panel.", maxArgs: 3 },
  { name: "Steps", signature: "Steps(items: StepsItem[]) — Sequential timeline steps.", maxArgs: 1, deps: ["StepsItem"] },
  { name: "StepsItem", signature: "StepsItem(title: string, details: string) — Single step entry.", maxArgs: 2 },
  { name: "ListBlock", signature: 'ListBlock(items: ListItem[], variant?: "number" | "image", size?: "default" | "small") — Structured list.', maxArgs: 3, deps: ["ListItem"] },
  { name: "ListItem", signature: "ListItem(title: string, subtitle?: string, image?: {src: string, alt: string}, actionLabel?: string, action?: ActionExpression) — Clickable list row.", maxArgs: 5 },
  { name: "Form", signature: "Form(name: string, buttons: Buttons, fields?: FormControl[]) — Interactive form with explicit Buttons.", maxArgs: 3, deps: ["FormControl", "Buttons", "Button"] },
  { name: "FormControl", signature: "FormControl(label: string, input: Component, hint?: string) — Labeled form field.", maxArgs: 3 },
  { name: "Select", signature: "Select(name: string, items: SelectItem[], placeholder?: string, rules?: object, value?: $binding) — Dropdown selector.", maxArgs: 6, deps: ["SelectItem"] },
  { name: "SelectItem", signature: "SelectItem(value: string, label: string) — Option inside Select.", maxArgs: 2 },
  { name: "Chips", signature: 'Chips(name: string, type?: "single" | "multiple", items?: ChipItem[], rules?: object, defaultValue?: string | string[]) — Selectable pill group.', maxArgs: 5, deps: ["ChipItem"] },
  { name: "ChipItem", signature: "ChipItem(value: string, label: string) — Single chip option.", maxArgs: 4 },
  { name: "OptionCards", signature: 'OptionCards(name: string, type?: "single" | "multiple", items?: OptionCard[], rules?: object, defaultValue?: string | string[]) — Selectable card grid.', maxArgs: 5, deps: ["OptionCard"] },
  { name: "OptionCard", signature: "OptionCard(value: string, title: string, subtitle?: string) — Single option card.", maxArgs: 5 },
  { name: "DatePicker", signature: 'DatePicker(name: string, mode?: "single" | "range", rules?: object, value?: $binding) — Date selector.', maxArgs: 4 },
  { name: "Slider", signature: 'Slider(name: string, variant: "continuous" | "discrete", min: number, max: number, step?: number, defaultValue?: number[]) — Numeric range slider.', maxArgs: 9 },
  { name: "Buttons", signature: 'Buttons(buttons: Button[], direction?: "row" | "column") — Action button group.', maxArgs: 2, deps: ["Button"] },
  { name: "Button", signature: 'Button(label: string, action?: ActionExpression, variant?: "primary" | "secondary" | "tertiary") — Interactive trigger button.', maxArgs: 5 },
  { name: "FollowUpBlock", signature: "FollowUpBlock(items: FollowUpItem[]) — Suggested next turns at the bottom of a card.", maxArgs: 1, deps: ["FollowUpItem"] },
  { name: "FollowUpItem", signature: "FollowUpItem(text: string) — Single tappable follow-up prompt.", maxArgs: 1 },
];

const PORTABLE_NAME = /^[a-z][a-z0-9_]{0,63}$/;
const IDENT_PATTERN = /^\$?[A-Za-z_][A-Za-z0-9_]*$/;

/** Converts `snake_case` component names (`restaurant_card`) to `PascalCase` (`RestaurantCard`). */
export function uiPascalName(snake: string): string {
  return snake
    .split("_")
    .filter(Boolean)
    .map((part) => part[0]!.toUpperCase() + part.slice(1))
    .join("");
}

/** Validates portable components and compiles a deterministic, prefix-cache-safe system prompt section. */
export function compileUIPrompt(
  settings: UIDeclaration,
  components: readonly UIPortableComponent[] = [],
  blueprints: readonly UIBlueprint[] = [],
  where = "ui",
): string {
  const format = settings.format ?? "openui";
  if (format !== "openui" && format !== "a2ui") {
    throw new ConfigurationError(`${where}: ui.format must be openui or a2ui, not ${JSON.stringify(format)}`);
  }
  validatePortableComponents(components, where);

  const byName = new Map(BUILTINS.map((b) => [b.name, b]));
  const customArity = new Map<string, number>();
  for (const comp of components) {
    const order = comp.parameters["x-order"] ?? [];
    customArity.set(uiPascalName(comp.name), order.length + (comp.slots?.length ?? 0));
  }

  const included = new Set<string>(["Card"]);
  if (!settings.keep || settings.keep.length === 0) {
    for (const b of BUILTINS) {
      included.add(b.name);
    }
  } else {
    const queue: string[] = [];
    for (const name of settings.keep) {
      if (customArity.has(name)) {
        continue;
      }
      if (!byName.has(name)) {
        throw new ConfigurationError(`${where}: ui.keep names unknown component ${JSON.stringify(name)}`);
      }
      if (!included.has(name)) {
        included.add(name);
        queue.push(name);
      }
    }
    while (queue.length > 0) {
      const current = queue.shift()!;
      for (const dep of byName.get(current)?.deps ?? []) {
        if (!included.has(dep)) {
          included.add(dep);
          queue.push(dep);
        }
      }
    }
  }

  const knownArity = new Map<string, number>();
  const selected: BuiltinComponent[] = [];
  for (const b of BUILTINS) {
    if (included.has(b.name)) {
      selected.push(b);
      knownArity.set(b.name, b.maxArgs);
    }
  }
  for (const [k, v] of customArity) {
    knownArity.set(k, v);
  }

  for (const bp of blueprints) {
    if (!bp.name.trim()) {
      throw new ConfigurationError(`${where}: a blueprint needs a name`);
    }
    if (!bp.when.trim()) {
      throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} needs a when condition`);
    }
    if (bp.order.length === 0) {
      throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} needs an order list`);
    }
    if (bp.example && format === "openui") {
      const problems = validateOpenUIProgram(extractFence(bp.example), knownArity);
      if (problems.length > 0) {
        throw new ConfigurationError(
          `${where}: blueprint ${JSON.stringify(bp.name)} example is invalid: ${problems.join("; ")}`,
        );
      }
    }
  }

  const lines: string[] = [];
  lines.push("## Interactive UI Generation", "");
  if (settings.voiceUI) {
    lines.push(
      "When an answer benefits from interactive UI, first speak or write one concise sentence summarizing the key takeaway, then emit the UI in a single ```openui-lang fenced block. Never read component names or UI syntax aloud.",
      "",
    );
  } else {
    lines.push(
      "When an answer benefits from interactive UI, first write one short sentence summarizing what the card shows, then write the whole UI in one ```openui-lang fenced block. On a voice call without a screen, speak concisely with no UI block.",
      "",
    );
  }

  lines.push(
    "### Syntax & Streaming Invariants",
    "1. Put `root = Card([...])` on the FIRST line so the card shell streams immediately.",
    "2. Every statement is `identifier = Expression` on its own line; arguments are strictly POSITIONAL (`TypeName(arg1, arg2)` — never `key: value`).",
    "3. Every defined identifier other than `root` and `$state` variables must be referenced from `root`.",
    "4. Never fabricate image URLs; use custom components or icons when no verified URL is in context.",
  );
  if (settings.patchMode) {
    lines.push(
      "5. When refining an existing UI from the immediately preceding turn (e.g. updating a time slot, party size, or filter), emit only the changed statements inside a ```openui-patch fenced block (`$party = 4`, or `oldSection = null` to remove).",
    );
  }

  lines.push("", "### Allowed Component Signatures");
  for (const def of selected) {
    lines.push(`- ${def.signature}`);
  }
  for (const comp of components) {
    lines.push(`- ${portableSignature(comp)}`);
  }

  if (blueprints.length > 0) {
    lines.push(
      "",
      "### Canonical Layout Blueprints (Predictability Rules)",
      "Match similar prompts to the exact component sequence below so repeated asks produce a consistent layout:",
    );
    for (const bp of blueprints) {
      const suffix = bp.rules && bp.rules.length > 0 ? ` — ${bp.rules.join("; ")}` : "";
      lines.push(`- **${bp.name}** (when ${bp.when}): \`Card([${bp.order.join(" -> ")}])\`${suffix}`);
      if (bp.example) {
        lines.push("", "```openui-lang", extractFence(bp.example).trim(), "```");
      }
    }
  }

  if (settings.rules && settings.rules.length > 0) {
    lines.push("", "### Brand & Design Rules");
    for (const rule of settings.rules) {
      lines.push(`- ${rule.trim()}`);
    }
  }

  return lines.join("\n").trim();
}

function validatePortableComponents(components: readonly UIPortableComponent[], where: string): void {
  const builtins = new Set(BUILTINS.map((b) => b.name));
  const seen = new Set<string>();
  for (const comp of components) {
    if (!PORTABLE_NAME.test(comp.name)) {
      throw new ConfigurationError(`${where}: component ${JSON.stringify(comp.name)} must be snake_case`);
    }
    const pascal = uiPascalName(comp.name);
    if (builtins.has(pascal)) {
      throw new ConfigurationError(`${where}: component ${JSON.stringify(comp.name)} (${pascal}) collides with a built-in component`);
    }
    if (seen.has(pascal)) {
      throw new ConfigurationError(`${where}: duplicate component ${JSON.stringify(comp.name)}`);
    }
    seen.add(pascal);
    if (!comp.description?.trim()) {
      throw new ConfigurationError(`${where}: component ${JSON.stringify(comp.name)} needs a description`);
    }
    const props = comp.parameters?.properties ?? {};
    const propKeys = Object.keys(props);
    const order = comp.parameters?.["x-order"] ?? [];
    if (propKeys.length > 0 && order.length !== propKeys.length) {
      throw new ConfigurationError(
        `${where}: component ${JSON.stringify(comp.name)} parameters.x-order must list all ${propKeys.length} properties`,
      );
    }
    for (const key of order) {
      if (!Object.hasOwn(props, key)) {
        throw new ConfigurationError(`${where}: component ${JSON.stringify(comp.name)} x-order names unknown property ${JSON.stringify(key)}`);
      }
    }
  }
}

function portableSignature(comp: UIPortableComponent): string {
  const required = new Set(comp.parameters.required ?? []);
  const props = comp.parameters.properties ?? {};
  const order = comp.parameters["x-order"] ?? [];
  const params: string[] = [];
  for (const key of order) {
    const prop = props[key];
    if (!prop) {
      continue;
    }
    const opt = required.has(key) ? "" : "?";
    const typ = prop.enum && prop.enum.length > 0 ? prop.enum.map((e) => JSON.stringify(e)).join(" | ") : prop.type;
    params.push(`${key}${opt}: ${typ}`);
  }
  for (const slot of comp.slots ?? []) {
    params.push(`${slot}?: Component[]`);
  }
  const details = [comp.description.trim()];
  for (const ev of comp.events ?? []) {
    if (ev.description?.trim()) {
      details.push(ev.description.trim());
    }
  }
  return `${uiPascalName(comp.name)}(${params.join(", ")}) — ${details.join(" ")}`;
}

function extractFence(example: string): string {
  const marker = "```openui-lang\n";
  const start = example.indexOf(marker);
  if (start >= 0) {
    const after = example.slice(start + marker.length);
    const close = after.indexOf("\n```");
    if (close >= 0) {
      return after.slice(0, close);
    }
  }
  return example;
}

/** Validates balanced delimiters, root assignment, component arity, and reference reachability in OpenUI Lang. */
export function validateOpenUIProgram(program: string, knownArity?: ReadonlyMap<string, number>): string[] {
  const trimmed = program.trim();
  if (!trimmed) {
    return ["program is empty"];
  }
  const stmts: string[] = [];
  let current = "";
  let quote = "";
  let escaped = false;
  let depth = 0;

  for (const line of trimmed.split("\n")) {
    if (!quote && depth === 0) {
      const clean = line.trim();
      if (!clean || clean.startsWith("//") || clean.startsWith("#")) {
        continue;
      }
    }
    current = current ? `${current}\n${line}` : line;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]!;
      if (escaped) {
        escaped = false;
        continue;
      }
      if (quote) {
        if (ch === "\\") {
          escaped = true;
        } else if (ch === quote) {
          quote = "";
        }
        continue;
      }
      if (ch === "/" && line[i + 1] === "/") {
        break;
      }
      if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === "(" || ch === "[" || ch === "{") {
        depth++;
      } else if (ch === ")" || ch === "]" || ch === "}") {
        depth--;
        if (depth < 0) {
          return ["unmatched closing bracket"];
        }
      }
    }
    if (!quote && depth === 0) {
      const s = current.trim();
      if (s) {
        stmts.push(s);
      }
      current = "";
    }
  }
  if (quote || depth !== 0) {
    return ["unclosed string or bracket"];
  }

  const defined = new Map<string, string>();
  const order: string[] = [];
  for (const stmt of stmts) {
    const split = splitAssign(stmt);
    if (!split) {
      return [`invalid statement ${JSON.stringify(stmt)}`];
    }
    defined.set(split.lhs, split.rhs);
    order.push(split.lhs);
  }
  if (!defined.has("root")) {
    return ["root is not defined"];
  }

  const problems: string[] = [];
  const used = new Set<string>(["root"]);
  const tokenRegex = /\$?[A-Za-z_][A-Za-z0-9_]*/g;
  const reserved = new Set(["true", "false", "null", "Action", "Query", "Mutation"]);

  for (const lhs of order) {
    const stripped = stripLiterals(defined.get(lhs) ?? "");
    let match: RegExpExecArray | null;
    tokenRegex.lastIndex = 0;
    while ((match = tokenRegex.exec(stripped)) !== null) {
      const tok = match[0];
      const start = match.index;
      const end = start + tok.length;
      const prev = start > 0 ? stripped[start - 1] : "";
      if (prev === "@" || prev === ".") {
        continue;
      }
      if (stripped.slice(end).trimStart().startsWith(":")) {
        continue;
      }
      if (reserved.has(tok)) {
        continue;
      }
      if (stripped[end] === "(") {
        if (knownArity && knownArity.size > 0) {
          const maxArgs = knownArity.get(tok);
          if (maxArgs === undefined) {
            problems.push(`unknown component ${tok}`);
          } else {
            const count = countCallArgs(stripped.slice(end));
            if (count > maxArgs) {
              problems.push(`${tok} passed ${count} arguments, max is ${maxArgs}`);
            }
          }
        }
        continue;
      }
      if (defined.has(tok)) {
        used.add(tok);
      } else {
        problems.push(`${tok} is referenced but not defined`);
      }
    }
  }

  for (const lhs of order) {
    if (lhs !== "root" && !lhs.startsWith("$") && !used.has(lhs)) {
      problems.push(`${lhs} is defined but never referenced`);
    }
  }
  return problems;
}

function splitAssign(stmt: string): { lhs: string; rhs: string } | undefined {
  for (let i = 0; i < stmt.length; i++) {
    if (stmt[i] !== "=") {
      continue;
    }
    const prev = i > 0 ? stmt[i - 1] : "";
    if (prev === "!" || prev === "<" || prev === ">" || prev === "=") {
      continue;
    }
    if (stmt[i + 1] === "=") {
      i++;
      continue;
    }
    const lhs = stmt.slice(0, i).trim();
    if (!IDENT_PATTERN.test(lhs)) {
      return undefined;
    }
    return { lhs, rhs: stmt.slice(i + 1).trim() };
  }
  return undefined;
}

function stripLiterals(s: string): string {
  let out = "";
  let quote = "";
  let escaped = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i]!;
    if (escaped) {
      escaped = false;
      continue;
    }
    if (quote) {
      if (ch === "\\") {
        escaped = true;
      } else if (ch === quote) {
        quote = "";
        out += '"';
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      out += '"';
      continue;
    }
    out += ch;
  }
  return out;
}

function countCallArgs(suffix: string): number {
  if (!suffix.startsWith("(")) {
    return 0;
  }
  let depth = 0;
  let commas = 0;
  let hasContent = false;
  for (let i = 0; i < suffix.length; i++) {
    const ch = suffix[i]!;
    if (ch === "(" || ch === "[" || ch === "{") {
      if (depth === 1) {
        hasContent = true;
      }
      depth++;
    } else if (ch === ")" || ch === "]" || ch === "}") {
      depth--;
      if (depth === 0) {
        return hasContent ? commas + 1 : 0;
      }
    } else if (ch === "," && depth === 1) {
      commas++;
    } else if (depth === 1 && !/\s/.test(ch)) {
      hasContent = true;
    }
  }
  return 0;
}
