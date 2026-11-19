import type { ConversationRecord, ReasoningStep, ToolStep, Turn } from "./conversation.js";
import type { LiveReasoning } from "./reasoning.js";
import type { ToolApprovalRequest } from "./session.js";
import { chatKeepsCustom } from "./stream.js";

/**
 * The turns of a conversation kept in no channel, an incognito session's, as the Stream Chat
 * messages a kept one is written into, so the AI components' `fromStreamMessage` reads both
 * alike.
 *
 * Each of the person's turns is a message, and the agent's turns between two of them are one
 * reply, as the router writes a reply whose tools it waited on: its text, its UI as `ai_ui`,
 * and each round of thinking and each tool call as an `ai_reasoning` or `ai_tool_call` step,
 * `generating` until the agent is done with it. A call waiting in `approvals` asks the
 * session's user, as its step does in a kept conversation. The thinking in full goes to
 * `reasoning`, by the reply and the step, as a kept reply's live updates do.
 */
export class TurnMessages {
  private readonly userId: string;
  private readonly reasoning: LiveReasoning;
  /** Each message as last made, by its first turn's id, with what it was made of. */
  private made = new Map<string, Made>();

  constructor(userId: string, reasoning: LiveReasoning) {
    this.userId = userId;
    this.reasoning = reasoning;
  }

  /** The messages of `record`, oldest first: the same object for each that didn't change. */
  messages(record: ConversationRecord): readonly Record<string, unknown>[] {
    const groups: Turn[][] = [];
    for (const turn of record.turns) {
      const last = groups[groups.length - 1];
      if (turn.speaker === "agent" && last?.[0]?.speaker === "agent") {
        last.push(turn);
      } else {
        groups.push([turn]);
      }
    }
    const custom = chatKeepsCustom();
    const made = new Map<string, Made>();
    const messages = groups.map((turns, index) => {
      // Only the latest reply can still be written: one the person asked over is as it was left.
      const generating = index === groups.length - 1 && record.connected && turns[turns.length - 1]?.done !== true;
      const id = turns[0]!.id;
      const before = this.made.get(id);
      const same =
        before !== undefined &&
        before.custom === custom &&
        before.generating === generating &&
        before.approvals === record.approvals &&
        before.turns.length === turns.length &&
        before.turns.every((turn, at) => turn === turns[at]);
      const one = same ? before : { turns, generating, approvals: record.approvals, custom, message: this.make(turns, generating, record.approvals, custom) };
      made.set(id, one);
      return one.message;
    });
    this.made = made;
    return messages;
  }

  private make(
    turns: readonly Turn[],
    generating: boolean,
    approvals: readonly ToolApprovalRequest[],
    custom: boolean,
  ): Record<string, unknown> {
    const first = turns[0]!;
    if (first.speaker === "participant") {
      return messageOf(first.id, first.text, [], first.at, { user: { id: first.participant?.userId || this.userId } }, {}, custom);
    }
    let rounds = 0;
    const steps: Record<string, unknown>[] = [];
    for (const step of turns.flatMap((turn) => turn.steps ?? [])) {
      if (step.kind === "reasoning") {
        // Numbered across the reply, which a tool's result carries on in a turn of its own.
        const id = `r${++rounds}`;
        this.reasoning.read({ id: first.id, reasoning: { id, offset: 0, text: step.text } });
        steps.push(reasoningStep(id, step));
      } else {
        steps.push(toolStep(step, approvals.find((request) => request.id === step.id), this.userId));
      }
    }
    const ui = turns.findLast((turn) => turn.ui)?.ui;
    const fields = {
      ai_generated: true,
      generating,
      ...(ui
        ? {
            ai_ui: {
              v: 1,
              format: ui.format,
              ...(ui.version ? { version: ui.version } : {}),
              status: ui.status,
              content: ui.content,
            },
          }
        : {}),
    };
    const text = turns
      .map((turn) => turn.text)
      .filter((said) => said)
      .join("\n\n");
    return messageOf(first.id, text, steps, first.at, {}, fields, custom);
  }
}

interface Made {
  readonly turns: readonly Turn[];
  readonly generating: boolean;
  readonly approvals: readonly ToolApprovalRequest[];
  readonly custom: boolean;
  readonly message: Record<string, unknown>;
}

/**
 * A message as the installed stream-chat holds one: the app's fields top level and times as
 * Dates in 9; under `custom` and in nanoseconds in 10.
 */
function messageOf(
  id: string,
  text: string,
  attachments: Record<string, unknown>[],
  at: number,
  extra: Record<string, unknown>,
  fields: Record<string, unknown>,
  custom: boolean,
): Record<string, unknown> {
  const common = { id, text, type: "regular", status: "received", attachments, ...extra };
  if (custom) {
    return Object.freeze({ ...common, custom: fields, created_at: at * 1e6, updated_at: at * 1e6 });
  }
  return Object.freeze({ ...fields, ...common, created_at: new Date(at), updated_at: new Date(at) });
}

function reasoningStep(id: string, step: ReasoningStep): Record<string, unknown> {
  return {
    type: "ai_reasoning",
    id,
    status: step.status,
    ...(step.durationMs !== undefined ? { duration_ms: step.durationMs } : {}),
  };
}

function toolStep(
  step: ToolStep,
  waiting: ToolApprovalRequest | undefined,
  userId: string,
): Record<string, unknown> {
  const asking = waiting !== undefined && step.status === "running";
  const { approval } = waiting ?? {};
  return {
    type: "ai_tool_call",
    id: step.id,
    name: step.name,
    status: asking ? "awaiting_approval" : step.status,
    ...(step.summary ? { summary: step.summary } : {}),
    ...(step.durationMs !== undefined ? { duration_ms: step.durationMs } : {}),
    ...(asking && approval
      ? {
          target_user_id: userId,
          approval: {
            title: approval.title,
            ...(approval.message ? { message: approval.message } : {}),
            ...(waiting.reason ? { reason: waiting.reason } : {}),
            ...(approval.allowTitle ? { allow_title: approval.allowTitle } : {}),
            ...(approval.declineTitle ? { decline_title: approval.declineTitle } : {}),
          },
        }
      : {}),
  };
}
