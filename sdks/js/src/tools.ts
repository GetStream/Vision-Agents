import { ConfigurationError } from "./errors.js";
import type { Schemas } from "./client.js";

/** A JSON Schema object describing what a tool takes. */
export type ParameterSchema = Readonly<Record<string, unknown>>;

/** One of the caller's own functions, as the model is offered it and as it runs. */
export interface Tool<In = Readonly<Record<string, unknown>>> {
  /** How the model asks for it. */
  name: string;
  /** What the model is told it does, which is the whole of how it decides to reach for one. */
  description: string;
  /**
   * A JSON Schema object describing the arguments.
   *
   * Written out rather than derived from the type, because a type is gone by the time this
   * runs and a schema guessed from one would be a schema the model was told wrongly.
   */
  parameters?: ParameterSchema;
  /** Runs it. What it returns is rendered for the model; a string is used as it is. */
  run: (input: In, signal: AbortSignal) => unknown;
}

/**
 * The functions a session offers, in the order they were registered.
 *
 * The functions stay here while the conversation runs in the backend, because the
 * functions are here: the model asks over the session socket and the answer goes back the
 * same way.
 */
export class Tools {
  private readonly registered = new Map<string, Tool<never>>();

  /** Adds a function. Registering one name twice is refused rather than silently winning. */
  register<In>(tool: Tool<In>): this {
    if (!tool.name) {
      throw new ConfigurationError("a tool needs a name");
    }
    if (!tool.description) {
      throw new ConfigurationError(
        `${tool.name} needs a description, since it is all the model has to choose by`,
      );
    }
    if (this.registered.has(tool.name)) {
      throw new ConfigurationError(`${tool.name} is registered twice`);
    }
    this.registered.set(tool.name, tool as Tool<never>);
    return this;
  }

  /** How many functions are registered. */
  get size(): number {
    return this.registered.size;
  }

  /** The functions as the session spec declares them. */
  declared(): Schemas["SessionTool"][] {
    return [...this.registered.values()].map((tool) => ({
      name: tool.name,
      description: tool.description,
      ...(tool.parameters ? { parameters: tool.parameters } : {}),
    }));
  }

  /**
   * Runs one function and renders what it returned in words the model can use.
   *
   * Arguments are the JSON object the model wrote. Empty is treated as no arguments, since
   * a model calling a function that takes none often sends nothing at all.
   */
  async call(name: string, args: string, signal: AbortSignal): Promise<string> {
    const tool = this.registered.get(name);
    if (!tool) {
      throw new ConfigurationError(`nothing is registered as ${name}`);
    }

    let input: unknown = {};
    if (args.trim()) {
      try {
        input = JSON.parse(args);
      } catch (cause) {
        throw new ConfigurationError(
          `${name} was asked for with arguments it cannot take: ${String(cause)}`,
        );
      }
    }
    return render(await tool.run(input as never, signal));
  }
}

/**
 * Turns what a function returned into words the model can use.
 *
 * A string is already that; everything else becomes JSON.
 */
export function render(output: unknown): string {
  if (typeof output === "string") {
    return output;
  }
  if (output === undefined) {
    return "";
  }
  try {
    return JSON.stringify(output) ?? String(output);
  } catch {
    return String(output);
  }
}
