import { ConfigurationError } from "./errors.js";
import type { UIBlueprint } from "./ui-catalog.js";

// What the OpenUI and A2UI prompts share, apart from both so neither imports the other.

/** Converts `snake_case` component names (`restaurant_card`) to `PascalCase` (`RestaurantCard`). */
export function uiPascalName(snake: string): string {
  return snake
    .split("_")
    .filter(Boolean)
    .map((part) => part[0]!.toUpperCase() + part.slice(1))
    .join("");
}

export function checkBlueprint(bp: UIBlueprint, where: string): void {
  if (!bp.name.trim()) {
    throw new ConfigurationError(`${where}: a blueprint needs a name`);
  }
  if (!bp.when.trim()) {
    throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} needs a when condition`);
  }
  if (bp.order.length === 0) {
    throw new ConfigurationError(`${where}: blueprint ${JSON.stringify(bp.name)} needs an order list`);
  }
}

export const UI_BLUEPRINTS_HEADING = [
  "",
  "### Canonical Layout Blueprints (Predictability Rules)",
  "Match similar prompts to the exact component sequence below so repeated asks produce a consistent layout:",
] as const;

/** The section's first lines: when to answer with UI, in a block fenced with `fence`. */
export function uiOpening(fence: string, voiceUI: boolean | undefined): string[] {
  const lines = ["## Interactive UI Generation", ""];
  if (voiceUI) {
    lines.push(
      `When an answer benefits from interactive UI, first speak or write one concise sentence summarizing the key takeaway, then emit the UI in a single \`\`\`${fence} fenced block. Never read component names or UI syntax aloud.`,
      "",
    );
  } else {
    lines.push(
      `When an answer benefits from interactive UI, first write one short sentence summarizing what the card shows, then write the whole UI in one \`\`\`${fence} fenced block. On a voice call without a screen, speak concisely with no UI block.`,
      "",
    );
  }
  return lines;
}

export function uiRules(rules: readonly string[] | undefined): string[] {
  if (!rules || rules.length === 0) {
    return [];
  }
  return ["", "### Brand & Design Rules", ...rules.map((rule) => `- ${rule.trim()}`)];
}

/** The program in an example's block fenced with `fence`, or the example itself when it has none. */
export function extractFence(example: string, fence: string): string {
  const marker = `\`\`\`${fence}\n`;
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
