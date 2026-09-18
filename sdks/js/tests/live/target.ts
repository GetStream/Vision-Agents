import { Client, RouterError } from "../../src/index.js";

/**
 * What the live suites need that a unit test does not: a deployment that is actually there.
 *
 * Neither suite is part of `npm test`. They talk to a real router, so they are opt-in and
 * they skip rather than fail when there is nothing to talk to — a laptop without the local
 * router up, or a checkout without credentials, is not a broken build.
 */

/**
 * Why a deployment cannot be tested, or `false` if it can. Pass it straight to `skip`.
 *
 * It answers with the reason rather than a yes or no so that a skip says which it was. A
 * deployment part way through a bad deploy answers 503 and one nobody started refuses the
 * connection, and told apart they are different days: the first is worth waiting out and
 * the second is worth starting something. Reported as one silent skip they are neither.
 */
export async function unreachable(url: string): Promise<string | false> {
  try {
    const response = await fetch(`${url}/health`, {
      signal: AbortSignal.timeout(5_000),
    });
    return response.ok ? false : `${url} answered ${response.status}`;
  } catch (raised) {
    return `${url} could not be reached: ${(raised as Error).message}`;
  }
}

/**
 * A conversation model this deployment knows.
 *
 * The routes are a property of the deployment rather than of the spec: a name is resolved
 * against that deployment's own catalogue, so the default a session would otherwise fall
 * to is not guaranteed to exist. Asking means a suite runs against a router configured
 * any which way instead of against one particular catalogue.
 */
export async function conversationModel(api: Client): Promise<string> {
  for (const target of ["llm-fast", "llm-thinking", "llm-flow"]) {
    try {
      await api.get("/v1/{modality}/routes/{target}", {
        path: { modality: "llm", target },
      });
      return target;
    } catch (raised) {
      if (raised instanceof RouterError && raised.status === 404) {
        continue;
      }
      throw raised;
    }
  }
  throw new Error("this deployment resolves none of the conversation models");
}

/**
 * Whether a refusal is the deployment having nothing left rather than something wrong.
 *
 * Holding a real conversation spends the app's daily token allowance, so a suite run often
 * enough will spend it. That is worth saying and stepping around; it is not the deployment
 * being broken, and reporting it as a failure sends somebody looking for a bug in the SDK.
 */
export function exhausted(reason: string): boolean {
  return reason.includes("quota");
}

/** Closes a session, so a suite that opened one does not leave it running. */
export async function close(api: Client, id: string): Promise<void> {
  await api.delete("/v1/agents/sessions/{id}", { path: { id } }).catch(() => undefined);
}

/** A conversation id no other run will pick, since one names one conversation at a time. */
export function uniqueId(what: string): string {
  return `test-${what}-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
}
