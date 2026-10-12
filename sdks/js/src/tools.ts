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
  /**
   * Who runs it, as the conversation shows it: `server`, what the router assumes, or
   * `client`, a person's device, whose call is shown with its arguments to every channel
   * member. Either way this process answers it.
   */
  executor?: "server" | "client";
  /**
   * What a call is doing, in words for the people in the conversation, such as "Checking
   * your location". At most 80 characters, shown on the reply's tool call attachment.
   */
  displayTitle?: string;
  /**
   * What a person is asked before each call runs. A call that asks waits in
   * `session.approvals` until `session.decide` answers it. A tool hosted through dispatch is
   * never asked about: its calls run straight away.
   */
  approval?: ToolApproval;
  /** Runs it. What it returns is rendered for the model; a string is used as it is. */
  run: (input: In, signal: AbortSignal) => unknown;
}

/**
 * A tool's question to the person whose message a call answers.
 *
 * In a conversation kept in Stream Chat the call's step carries it as `awaiting_approval`,
 * and every channel member can read it: keep anything private out of it. A client tool is
 * asked about only once the router knows the install to address it to; until then, declare
 * it with the server executor.
 */
export interface ToolApproval {
  /** The question, such as "Share your location?". At most 80 characters. */
  title: string;
  /** What allowing it shares or does, such as "Only your city is shared." */
  message?: string;
  /**
   * The string argument in which the model says why it wants the call, shown beside the
   * question. Declare it in `parameters` so the model fills it in.
   */
  reasonArgument?: string;
  /** The label of the button that allows the call. */
  allowTitle?: string;
  /** The label of the button that declines it. */
  declineTitle?: string;
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
      ...(tool.executor ? { executor: tool.executor } : {}),
      ...(tool.displayTitle ? { display_title: tool.displayTitle } : {}),
      ...(tool.approval ? { approval: approvalOf(tool.approval) } : {}),
    }));
  }

  /** The question the function named asks before it runs, if it asks one. */
  approval(name: string): ToolApproval | undefined {
    return this.registered.get(name)?.approval;
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

function approvalOf(approval: ToolApproval): Schemas["SessionToolApproval"] {
  return {
    title: approval.title,
    ...(approval.message ? { message: approval.message } : {}),
    ...(approval.reasonArgument ? { reason_argument: approval.reasonArgument } : {}),
    ...(approval.allowTitle ? { allow_title: approval.allowTitle } : {}),
    ...(approval.declineTitle ? { decline_title: approval.declineTitle } : {}),
  };
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
