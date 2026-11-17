#!/usr/bin/env node
/**
 * Writes src/generated/a2ui.ts from sdks/go/agents/a2ui: the A2UI basic catalog, protocol and
 * writing rules the Go SDK compiles `ui:` with when its format is a2ui. Go's a2ui/generate.sh
 * writes those files from stream-chat-swift's renderer; run this after it, so both SDKs tell the
 * agent what the apps draw. A test fails while the two are out of step.
 *
 * The output is a module, not the files themselves, since the package also runs in a browser.
 *
 *     npm run a2ui
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const source = join(pkg, "..", "go", "agents", "a2ui");
const output = join(pkg, "src", "generated", "a2ui.ts");

const catalog = JSON.parse(readFileSync(join(source, "basic_catalog.json"), "utf8"));
const protocol = readFileSync(join(source, "protocol.md"), "utf8");
const writing = readFileSync(join(source, "writing.md"), "utf8");

const generated = `/**
 * Generated from sdks/go/agents/a2ui. Do not edit.
 *
 * Regenerate with \`npm run a2ui\` in sdks/js, after sdks/go/agents/a2ui/generate.sh.
 */

/** A JSON Schema as the A2UI catalog has it. */
export type A2UISchema = { readonly [key: string]: unknown };

/** The A2UI basic catalog as stream-chat-swift's renderer serializes it. */
export interface A2UIBasicCatalog {
  readonly $schema: string;
  readonly title: string;
  readonly description: string;
  readonly instructions: string;
  readonly components: Readonly<Record<string, A2UISchema>>;
  readonly functions: Readonly<Record<string, A2UISchema>>;
  readonly [key: string]: unknown;
}

export const A2UI_BASIC_CATALOG: A2UIBasicCatalog = ${JSON.stringify(catalog, null, 2)};

/** What the renderer tells a model about A2UI's messages (protocol.md). */
export const A2UI_PROTOCOL = ${JSON.stringify(protocol)};

/** How a reply carries its UI, the renderer's own A2UIInstructions (writing.md). */
export const A2UI_WRITING = ${JSON.stringify(writing)};
`;

writeFileSync(output, generated);
console.log(`regenerated ${relative(join(pkg, "..", ".."), output)}`);
