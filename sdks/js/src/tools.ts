import { ConfigurationError } from "./errors.js";
import type { Schemas } from "./client.js";

/** A JSON Schema object describing what a tool takes. */
export type ParameterSchema = Readonly<Record<string, unknown>>;

/** One problem a {@link StandardToolSchema} found with a value. */
export interface StandardIssue {
  readonly message: string;
  readonly path?: ReadonlyArray<PropertyKey | { readonly key: PropertyKey }> | undefined;
}

type StandardResult<Output> =
  | { readonly value: Output; readonly issues?: undefined }
  | { readonly issues: ReadonlyArray<StandardIssue> };

/**
 * A schema that validates a value and says what it takes as JSON Schema, in the shape of
 * Standard Schema and Standard JSON Schema (https://standardschema.dev). A Zod 4 schema is
 * one, as are ArkType's and Valibot's.
 */
export interface StandardToolSchema<Input = unknown, Output = Input> {
  readonly "~standard": {
    readonly version: 1;
    readonly vendor: string;
    readonly validate: (
      value: unknown,
    ) => StandardResult<Output> | Promise<StandardResult<Output>>;
    readonly jsonSchema: {
      readonly input: (options: { readonly target: "draft-2020-12" }) => Record<string, unknown>;
    };
    readonly types?: { readonly input: Input; readonly output: Output } | undefined;
  };
}

/** One of the caller's own functions, as the model is offered it and as it runs. */
export interface Tool<In = Readonly<Record<string, unknown>>> {
  /** How the model asks for it. */
  name: string;
  /** What the model is told it does, which is the whole of how it decides to reach for one. */
  description: string;
  /**
   * What the arguments are: a JSON Schema object, or a schema such as a Zod 4 `z.object`.
   *
   * A JSON Schema is told to the model as it is, and `run` is handed the arguments
   * unchecked. A Zod schema is told to the model as the JSON Schema it describes, and each
   * call's arguments are parsed by it first: `run` is handed what it parsed, typed from it,
   * and arguments it refuses fail the call with what was wrong, for the model to correct.
   * A rule JSON Schema cannot say, such as a `refine`, still holds; the model is just not
   * told it.
   */
  parameters?: ParameterSchema | StandardToolSchema<unknown, In>;
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
   * `session.approvals` until `session.decide` answers it. On a voice call the agent asks
   * the caller out loud instead, with each `{name}` in the title filled in from the call's
   * arguments, and runs the call on a yes within 15 seconds (`toolCallContext.approved`). A
   * tool hosted through dispatch is asked about only on a voice call.
   */
  approval?: ToolApproval;
  /**
   * Whether the caller talking over the agent cancels a call in flight. Off, the default,
   * the call keeps running and the model is told it is still going.
   */
  interruptible?: boolean;
  /**
   * How long one call may take, in milliseconds, in place of the session's. A call that takes
   * longer is aborted and fails. Zero or omitted takes the session's.
   */
  timeoutMs?: number;
  /**
   * Runs it. What it returns is rendered for the model; a string is used as it is.
   *
   * `signal` aborts when the call is cancelled or times out. `toolCallContext` is the
   * call being answered, and how the function tells the conversation how it is going.
   */
  run: (input: In, signal: AbortSignal, toolCallContext: ToolCall) => unknown;
}

/** The person whose turn asked for a call. */
export interface ToolUser {
  id: string;
  name: string;
  custom: Readonly<Record<string, unknown>>;
}

/** The Chat channel a call was made in, and the session's own labels. */
export interface ToolConversation {
  cid: string;
  custom: Readonly<Record<string, unknown>>;
}

/** What the router said about a call. */
export interface ToolCallFields {
  /**
   * The id the model gave the call. A call sent again keeps it, so it is the key to make a
   * side effect happen once.
   */
  id: string;
  sessionId: string;
  turnId: string;
  /** The person whose turn asked for it. Empty when the router did not say. */
  user: ToolUser;
  conversation: ToolConversation;
  /** The caller already said yes to this call out loud, on a voice call. */
  approved: boolean;
}

/**
 * How a running call reaches the session it runs for. The socket that delivered the call
 * provides one; a test of a function's progress can too.
 */
export interface ToolLine {
  /** Says how the call is going, or with `verbatim`, words to say as written. */
  update(id: string, text: string, verbatim: boolean): void;
  /** Asks for the call to keep going when the caller talks over the agent; resolves with whether it was still running to be held. */
  hold(id: string): Promise<boolean>;
}

/** Why the signal `disallowInterruptions` returns is aborted when the call had already been stopped. */
export class CallNotHeldError extends Error {
  constructor(id: string) {
    super(`the call ${id} was stopped before it could be held`);
    this.name = "CallNotHeldError";
  }
}

/**
 * The call a function is running for.
 *
 * Built without a line, as in a unit test of the function, its methods do nothing.
 */
export class ToolCall implements ToolCallFields {
  readonly id: string;
  readonly sessionId: string;
  readonly turnId: string;
  readonly user: ToolUser;
  readonly conversation: ToolConversation;
  readonly approved: boolean;
  private readonly line: ToolLine | undefined;
  private readonly lasting: AbortSignal;

  /**
   * `lasting` is what still stops the call once it is held: its socket ending, and its
   * timeout.
   */
  constructor(fields: Partial<ToolCallFields> = {}, line?: ToolLine, lasting?: AbortSignal) {
    this.id = fields.id ?? "";
    this.sessionId = fields.sessionId ?? "";
    this.turnId = fields.turnId ?? "";
    this.user = fields.user ?? { id: "", name: "", custom: {} };
    this.conversation = fields.conversation ?? { cid: "", custom: {} };
    this.approved = fields.approved ?? false;
    this.line = line;
    this.lasting = lasting ?? new AbortController().signal;
  }

  /**
   * Says how the call is going. A voice agent tells the caller in its own words once nobody
   * is talking, and a conversation shows it on the call's step until the call ends. A newer
   * update replaces one not yet said, and the result replaces both.
   */
  update(text: string): void {
    if (this.line && this.id) {
      this.line.update(this.id, text, false);
    }
  }

  /**
   * Has the agent say `text` as written while the call runs, or post it in a text
   * conversation. A speech-to-speech model is asked for these words and may paraphrase them.
   */
  say(text: string): void {
    if (this.line && this.id) {
      this.line.update(this.id, text, true);
    }
  }

  /**
   * Keeps the call going from now on when the caller talks over the agent, which is what a
   * write that must not be left halfway wants. Resolves with the signal to do that write on:
   * only the call's timeout and its socket ending abort it. When the call was stopped before
   * it could be held, that signal is already aborted, with a `CallNotHeldError`.
   */
  async disallowInterruptions(): Promise<AbortSignal> {
    if (!this.line || !this.id) {
      return this.lasting;
    }
    if (!(await this.line.hold(this.id))) {
      return AbortSignal.abort(new CallNotHeldError(this.id));
    }
    return this.lasting;
  }

  /** The same call, which also stops when `signal` aborts once it is held. */
  lastingUntil(signal: AbortSignal): ToolCall {
    return new ToolCall(this, this.line, AbortSignal.any([this.lasting, signal]));
  }
}

/** A {name} in an approval title, filled in from the call's arguments. */
const PLACEHOLDER = /\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

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
  private readonly registered = new Map<string, Registered>();

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
    if (tool.timeoutMs !== undefined && !(Number.isFinite(tool.timeoutMs) && tool.timeoutMs >= 0)) {
      throw new ConfigurationError(`${tool.name} has a timeout that is not a duration: ${tool.timeoutMs}`);
    }
    const parameters = parametersOf(tool.name, tool.parameters);
    if (tool.approval) {
      const properties = (parameters?.["properties"] ?? {}) as Record<string, unknown>;
      for (const [, argument] of tool.approval.title.matchAll(PLACEHOLDER)) {
        if (!(argument! in properties)) {
          throw new ConfigurationError(
            `${tool.name}'s approval names {${argument}}, which is not one of its parameters`,
          );
        }
      }
    }
    this.registered.set(tool.name, {
      tool: tool as unknown as Tool<never>,
      parameters,
      parse: isStandard(tool.parameters) ? tool.parameters["~standard"].validate : undefined,
    });
    return this;
  }

  /** How many functions are registered. */
  get size(): number {
    return this.registered.size;
  }

  /** The functions as the session spec declares them. */
  declared(): Schemas["SessionTool"][] {
    return [...this.registered.values()].map(({ tool, parameters }) => ({
      name: tool.name,
      description: tool.description,
      ...(parameters ? { parameters } : {}),
      ...(tool.executor ? { executor: tool.executor } : {}),
      ...(tool.displayTitle ? { display_title: tool.displayTitle } : {}),
      ...(tool.approval ? { approval: approvalOf(tool.approval) } : {}),
      ...(tool.interruptible ? { interruptible: true } : {}),
      ...(tool.timeoutMs ? { timeout_ms: Math.max(1, Math.round(tool.timeoutMs)) } : {}),
    }));
  }

  /** Whether a function is registered under this name. */
  has(name: string): boolean {
    return this.registered.has(name);
  }

  /** The question the function named asks before it runs, if it asks one. */
  approval(name: string): ToolApproval | undefined {
    return this.registered.get(name)?.tool.approval;
  }

  /**
   * Runs one function and renders what it returned in words the model can use.
   *
   * Arguments are the JSON object the model wrote. Empty is treated as no arguments, since
   * a model calling a function that takes none often sends nothing at all. A function with
   * a timeout that takes longer is aborted and fails.
   */
  async call(
    name: string,
    args: string,
    signal: AbortSignal,
    toolCallContext: ToolCall = new ToolCall(),
  ): Promise<string> {
    const registered = this.registered.get(name);
    if (!registered) {
      throw new ConfigurationError(`nothing is registered as ${name}`);
    }
    const { tool, parse } = registered;

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
    if (parse) {
      const parsed = await parse(input);
      if (parsed.issues) {
        throw new ConfigurationError(
          `${name} was asked for with arguments it cannot take: ${describeIssues(parsed.issues)}`,
        );
      }
      input = parsed.value;
    }
    if (!tool.timeoutMs) {
      return render(await tool.run(input as never, signal, toolCallContext));
    }

    const timeout = tool.timeoutMs;
    const late = new AbortController();
    const timer = setTimeout(
      () => late.abort(new Error(`${name} took longer than ${timeout} ms`)),
      timeout,
    );
    try {
      const running = Promise.resolve(
        tool.run(
          input as never,
          AbortSignal.any([signal, late.signal]),
          toolCallContext.lastingUntil(late.signal),
        ),
      );
      const timedOut = new Promise<never>((_, reject) => {
        late.signal.addEventListener("abort", () => reject(late.signal.reason as Error), { once: true });
      });
      return render(await Promise.race([running, timedOut]));
    } finally {
      clearTimeout(timer);
    }
  }
}

interface Registered {
  tool: Tool<never>;
  /** What the model is told the tool takes. */
  parameters: ParameterSchema | undefined;
  /** What a call's arguments are parsed by first, for a tool described by a schema library. */
  parse: StandardToolSchema["~standard"]["validate"] | undefined;
}

function isStandard(parameters: unknown): parameters is StandardToolSchema {
  return typeof parameters === "object" && parameters !== null && "~standard" in parameters;
}

/** The JSON Schema a tool's parameters tell the model, refusing one that is not an object. */
function parametersOf(name: string, parameters: Tool<unknown>["parameters"]): ParameterSchema | undefined {
  if (!isStandard(parameters)) {
    return parameters;
  }
  const standard = parameters["~standard"];
  if (typeof standard.jsonSchema?.input !== "function") {
    throw new ConfigurationError(
      `${name}'s parameters are a Standard Schema with no JSON Schema to tell the model, ` +
        `such as a recent Zod 4 schema has (this one is ${standard.vendor}'s)`,
    );
  }
  const { $schema: _, ...schema } = standard.jsonSchema.input({ target: "draft-2020-12" });
  if (schema["type"] !== "object") {
    throw new ConfigurationError(
      `${name}'s parameters describe ${String(schema["type"] ?? "something else")}, and a tool takes an object`,
    );
  }
  return schema;
}

/** What a schema found wrong with a call's arguments, in words the model can correct. */
function describeIssues(issues: ReadonlyArray<StandardIssue>): string {
  return issues
    .map((issue) => {
      const path = (issue.path ?? [])
        .map((segment) => String(typeof segment === "object" ? segment.key : segment))
        .join(".");
      return path ? `${path}: ${issue.message}` : issue.message;
    })
    .join("; ");
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
