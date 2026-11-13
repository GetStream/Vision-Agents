import type { Turn, TurnUI } from "./conversation.js";
import { messageField } from "./message-field.js";

/**
 * Overlays 0ms WebSocket `ui_delta` / `ui_completed` state from `session.conversation`
 * onto Stream Chat assistant messages while they stream, handing off without a flicker
 * once Stream Chat's durable `custom.ai_ui` settles (`status: "complete"`).
 */
export class LiveUI {
  private readonly byTurn = new Map<string, TurnUI>();
  private latestTurnId: string | undefined;
  private cache = new WeakMap<object, { readonly ui: TurnUI; readonly value: unknown }>();

  /** The live UI recorded for `turnId`, or undefined. */
  get(turnId: string): TurnUI | undefined {
    return this.byTurn.get(turnId);
  }

  /** The latest agent turn's live UI, for a voice screen or an in-flight reply before its turn id lands in Chat. */
  latest(): TurnUI | undefined {
    return this.latestTurnId !== undefined ? this.byTurn.get(this.latestTurnId) : undefined;
  }

  /**
   * @internal Folds the current `session.conversation` turns in and reports whether any
   * turn's UI changed.
   */
  observeTurns(turns: readonly Turn[] | undefined): boolean {
    if (!turns || turns.length === 0) {
      return false;
    }
    let changed = false;
    for (const turn of turns) {
      if (turn.speaker !== "agent" || !turn.ui) {
        continue;
      }
      const previous = this.byTurn.get(turn.id);
      if (previous !== turn.ui) {
        this.byTurn.set(turn.id, turn.ui);
        this.latestTurnId = turn.id;
        changed = true;
      }
    }
    return changed;
  }

  /**
   * Returns `messages` with live socket UI overlaid onto any assistant reply whose durable
   * Stream Chat `ai_ui` has not yet reached `status: "complete"`. Preserves array and item
   * references whenever nothing changed.
   */
  overlay<M extends { readonly id: string }>(messages: readonly M[]): readonly M[] {
    if (this.byTurn.size === 0 || messages.length === 0) {
      return messages;
    }
    const latestAssistantIndex = findLatestGeneratingAssistant(messages);
    let mutated: M[] | undefined;

    for (let i = 0; i < messages.length; i++) {
      const message = messages[i]!;
      if (!isAssistantMessage(message)) {
        continue;
      }
      const settled = existingUIStatus(message);
      if (settled === "complete") {
        continue;
      }
      const turnId = turnIdOfMessage(message);
      const ui =
        (turnId ? this.byTurn.get(turnId) : undefined) ??
        (i === latestAssistantIndex ? this.latest() : undefined);
      if (!ui) {
        continue;
      }
      const next = this.withUI(message, ui);
      if (next !== message) {
        mutated ??= messages.slice();
        mutated[i] = next;
      }
    }
    return mutated ?? messages;
  }

  private withUI<M extends object>(message: M, ui: TurnUI): M {
    const cached = this.cache.get(message);
    if (cached?.ui === ui) {
      return cached.value as M;
    }
    const wire = Object.freeze({
      v: 1,
      format: ui.format,
      ...(ui.version ? { version: ui.version } : {}),
      status: ui.status,
      content: ui.content,
    });
    const record = message as Record<string, unknown>;
    const existingCustom = record["custom"];
    const custom =
      typeof existingCustom === "object" && existingCustom !== null && !Array.isArray(existingCustom)
        ? { ...(existingCustom as Record<string, unknown>), ai_ui: wire }
        : { ai_ui: wire };
    const overlaid = {
      ...message,
      ai_ui: wire,
      custom,
    } as M;
    this.cache.set(message, { ui, value: overlaid });
    return overlaid;
  }
}

function isAssistantMessage(message: object): boolean {
  if (messageField(message, "ai_generated") === true) {
    return true;
  }
  const runtime = messageField(message, "support_runtime");
  if (typeof runtime === "object" && runtime !== null && (runtime as Record<string, unknown>)["role"] === "assistant") {
    return true;
  }
  return false;
}

function existingUIStatus(message: object): string | undefined {
  const ui = messageField(message, "ai_ui");
  if (typeof ui !== "object" || ui === null) {
    return undefined;
  }
  const record = ui as Record<string, unknown>;
  if (record["overflow"] === true) {
    return undefined;
  }
  const status = record["status"];
  return typeof status === "string" ? status : undefined;
}

function turnIdOfMessage(message: object): string | undefined {
  const direct = messageField(message, "turn_id");
  if (typeof direct === "string" && direct) {
    return direct;
  }
  const runtime = messageField(message, "support_runtime");
  if (typeof runtime === "object" && runtime !== null) {
    const turnId = (runtime as Record<string, unknown>)["turn_id"];
    if (typeof turnId === "string" && turnId) {
      return turnId;
    }
  }
  return undefined;
}

function findLatestGeneratingAssistant(messages: readonly object[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (isAssistantMessage(msg) && messageField(msg, "generating") === true) {
      return i;
    }
  }
  return -1;
}
