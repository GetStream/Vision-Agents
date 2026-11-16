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
   * How many calls to it run at once. A call over the cap waits its turn, and its timeout keeps
   * counting while it waits. Omitted, its calls are capped only by how many its `Tools` runs at
   * once in all, 16 unless `setConcurrency` says otherwise.
   */
  concurrency?: number;
  /**
   * Runs it, with the arguments and then the call it runs for, which a function that needs
   * neither who asked nor how to tell them how it is going leaves out. What it returns is
   * rendered for the model: a string is used as it is, a `ToolOutput` also shows the model its
   * images, and anything else becomes JSON.
   */
  run: (input: In, toolCallContext: ToolCallContext) => unknown;
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
  hold(id: string, signal?: AbortSignal): Promise<boolean>;
}

/** What `disallowInterruptions` rejects with when the call had already been stopped. */
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
export class ToolCallContext implements ToolCallFields {
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
   * write that must not be left halfway wants: from then on only the call's timeout and its
   * socket ending stop it. Rejects with a `CallNotHeldError` when the call was stopped before
   * it could be held, so the write should not start.
   */
  async disallowInterruptions(): Promise<void> {
    if (!this.line || !this.id) {
      return;
    }
    const held = await this.line.hold(this.id, this.lasting);
    if (!held || this.lasting.aborted) {
      throw new CallNotHeldError(this.id);
    }
  }

  /** The same call, which also stops when `signal` aborts once it is held. */
  lastingUntil(signal: AbortSignal): ToolCallContext {
    return new ToolCallContext(this, this.line, AbortSignal.any([this.lasting, signal]));
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
  private readonly slots = new Slots(DEFAULT_CONCURRENCY);

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
    if (tool.concurrency !== undefined && !(Number.isInteger(tool.concurrency) && tool.concurrency >= 1)) {
      throw new ConfigurationError(
        `${tool.name} has a concurrency that is not a number of calls: ${tool.concurrency}`,
      );
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
      slots: tool.concurrency ? new Slots(tool.concurrency) : undefined,
    });
    return this;
  }

  /**
   * Caps how many calls to all its functions run at once, in place of 16. A `Dispatch` hosting
   * an agent shares its `tools` across every session it hosts, so this caps the worker's calls.
   * Calls already waiting take a place as soon as the cap allows.
   */
  setConcurrency(calls: number): this {
    if (!(Number.isInteger(calls) && calls >= 1)) {
      throw new ConfigurationError(`a concurrency that is not a number of calls: ${calls}`);
    }
    this.slots.resize(calls);
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
   * a timeout that takes longer is aborted and fails, its wait for a place to run included.
   */
  async call(
    name: string,
    args: string,
    signal: AbortSignal,
    toolCallContext: ToolCallContext = new ToolCallContext(),
  ): Promise<string | ToolOutputPart[]> {
    const registered = this.registered.get(name);
    if (!registered) {
      throw new ConfigurationError(`nothing is registered as ${name}`);
    }
    const { tool, parse, slots } = registered;

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
    // The function cannot be stopped from here, so a cancelled call stops being waited on.
    let stop = (): void => undefined;
    const cancelled = new Promise<never>((_, reject) => {
      stop = () => reject(new Error(`${name} was cancelled`));
    });
    cancelled.catch(() => undefined);
    signal.addEventListener("abort", stop, { once: true });
    if (signal.aborted) {
      stop();
    }

    const timeout = tool.timeoutMs;
    const late = new AbortController();
    const timer = timeout
      ? setTimeout(() => late.abort(new Error(`${name} took longer than ${timeout} ms`)), timeout)
      : undefined;
    const timedOut = new Promise<never>((_, reject) => {
      late.signal.addEventListener("abort", () => reject(late.signal.reason as Error), { once: true });
    });
    timedOut.catch(() => undefined);
    const stopped = Promise.race([timedOut, cancelled]);
    stopped.catch(() => undefined);
    const held: Slots[] = [];
    let running: Promise<unknown> | undefined;
    try {
      // The function's own place first, so its calls waiting their turn hold none of the
      // places every other function needs.
      for (const places of [slots, this.slots]) {
        if (places) {
          await places.take(stopped);
          held.unshift(places);
        }
      }
      const call = timeout ? toolCallContext.lastingUntil(late.signal) : toolCallContext;
      running = Promise.resolve(tool.run(input as never, call));
      return render(await Promise.race([running, timedOut, cancelled]));
    } finally {
      clearTimeout(timer);
      signal.removeEventListener("abort", stop);
      // A call no longer waited on may still be running, and keeps its places until it returns.
      const release = (): void => held.forEach((places) => places.give());
      if (running) {
        running.then(release, release);
      } else {
        release();
      }
    }
  }
}

/** How many calls a `Tools` runs at once unless told otherwise. */
const DEFAULT_CONCURRENCY = 16;

/** Places for calls to run in, handed to the calls waiting for one in the order they came. */
class Slots {
  private size: number;
  private taken = 0;
  private readonly waiting: (() => void)[] = [];

  constructor(size: number) {
    this.size = size;
  }

  /** Takes a place, waiting for one unless `stopped` rejects first, which leaves the queue. */
  async take(stopped: Promise<never>): Promise<void> {
    if (this.taken < this.size && this.waiting.length === 0) {
      this.taken += 1;
      return;
    }
    let grant = (): void => undefined;
    const granted = new Promise<void>((resolve) => {
      grant = resolve;
    });
    this.waiting.push(grant);
    try {
      await Promise.race([granted, stopped]);
    } catch (reason) {
      const at = this.waiting.indexOf(grant);
      if (at >= 0) {
        this.waiting.splice(at, 1);
      } else {
        // Handed a place just as it stopped waiting, so it passes the place on.
        this.give();
      }
      throw reason;
    }
  }

  /** Gives a place back, to the call that has waited longest. */
  give(): void {
    this.taken -= 1;
    this.handOut();
  }

  /** Makes it `size` places. Calls holding more than that keep them until they give them back. */
  resize(size: number): void {
    this.size = size;
    this.handOut();
  }

  private handOut(): void {
    while (this.taken < this.size && this.waiting.length > 0) {
      this.taken += 1;
      this.waiting.shift()!();
    }
  }
}

interface Registered {
  tool: Tool<never>;
  /** What the model is told the tool takes. */
  parameters: ParameterSchema | undefined;
  /** What a call's arguments are parsed by first, for a tool described by a schema library. */
  parse: StandardToolSchema["~standard"]["validate"] | undefined;
  /** The places its calls run in, for a tool that caps them. */
  slots: Slots | undefined;
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
 * What a function says and shows: words, and pictures the model looks at with them.
 *
 * Return one from `run` when the answer is something to see, such as a photo the device
 * took. Anything else a function returns is words only.
 */
export class ToolOutput {
  /** What the function says. */
  readonly text: string;
  /** What the function shows: an image URL, or a data URI. */
  readonly images: readonly Schemas["ImageSource"][];

  constructor(output: { text?: string; images?: readonly Schemas["ImageSource"][] } = {}) {
    this.text = output.text ?? "";
    this.images = output.images ?? [];
  }
}

/** One part of a result that shows pictures, as the router takes it. */
export type ToolOutputPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } };

/**
 * Turns what a function returned into what the model is told.
 *
 * A string is already that; a `ToolOutput` with images is its words and then its pictures, as
 * parts; everything else becomes JSON.
 */
export function render(output: unknown): string | ToolOutputPart[] {
  if (output instanceof ToolOutput) {
    if (output.images.length === 0) {
      return output.text;
    }
    return [
      ...(output.text ? [{ type: "text" as const, text: output.text }] : []),
      ...output.images.map((image) => ({
        type: "image_url" as const,
        image_url: { url: image.url, ...(image.detail ? { detail: image.detail } : {}) },
      })),
    ];
  }
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
