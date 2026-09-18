#!/usr/bin/env node
/**
 * Regenerates this package's types from the acceleration OpenAPI spec.
 *
 * What comes out is types only: openapi-typescript emits a `paths` map and a `components`
 * map and no runtime at all, so the request code stays hand-written in `src/client.ts` and
 * is typed against the spec rather than generated from it.
 *
 * Types only is what makes one package work in a browser and on a server. A generated
 * client brings a runtime with it, and the runtime is the part that has to decide how a
 * token is minted and which global fetch to use — which is the part that differs between
 * the two.
 *
 * WebSockets are not generated either, because OpenAPI stops at the upgrade. The session,
 * dispatch and modality sockets are hand-written in `src/socket.ts` and its callers.
 *
 * The output is committed, so installing the package needs no code generation.
 *
 *     npm run types
 *     npm run types -- --check
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, "..");
const spec = join(pkg, "..", "..", "acceleration", "api", "openapi.yaml");
const output = join(pkg, "src", "generated", "api.ts");

const header = `/**
 * Generated from acceleration/api/openapi.yaml. Do not edit.
 *
 * Regenerate with \`npm run types\` in sdks/js.
 */

`;

const generated =
  header +
  execFileSync(
    "npx",
    [
      "--no-install",
      "openapi-typescript",
      spec,
      "--export-type",
      "--immutable",
      "--alphabetize",
      // A field the spec gives a default is one the caller may leave out, and the router
      // fills it in. Generated non-nullable it would be required on the way in, so every
      // session request would have to spell out the defaults it wanted.
      "--default-non-nullable",
      "false",
    ],
    { cwd: pkg, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] },
  );

const shown = relative(join(pkg, "..", ".."), output);

if (process.argv.includes("--check")) {
  const stored = readFileSync(output, "utf8");
  if (stored !== generated) {
    console.error(`${shown} is out of step with the spec; run npm run types in sdks/js`);
    process.exit(1);
  }
  console.log(`${shown} is in step with the spec`);
} else {
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, generated);
  console.log(`regenerated ${shown}`);
}
