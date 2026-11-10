import type { Turn } from "@stream-io/stream-agents";

/**
 * Closed captions for a spoken conversation: what is being said, a sentence at a time, gone a
 * moment after the talking stops.
 *
 * The agent's reply is written before it is spoken, so its words are revealed at the pace of its
 * voice, counting the time its audio is heard, rather than as they arrive. What the person said
 * appears once it has been transcribed. A port of the iOS and Android apps' `Captions`.
 */
export interface Caption {
  readonly text: string;
  readonly isPerson: boolean;
}

/** About how fast the agent speaks, in characters a second. */
const PACE = 15;
/** How long a caption stays up once nothing new is being said, in milliseconds. */
const LINGER = 2_500;
/** How long the agent has to be quiet, with its reply written, for the reply to be over. */
const FINISHED = 1_500;

export class Captions {
  caption: Caption | undefined;
  isVisible = false;

  private turnId: string | undefined;
  private spoken = 0;
  private lastAudible = Number.NEGATIVE_INFINITY;
  private lastChange = Number.NEGATIVE_INFINITY;

  /** Moves the captions on by `elapsed` milliseconds, at `now`. */
  update(input: {
    turn: Turn | undefined;
    agentIsAudible: boolean;
    agentIsAnswering: boolean;
    now: number;
    elapsed: number;
  }): void {
    const { turn, agentIsAudible, agentIsAnswering, now, elapsed } = input;
    let next: Caption | undefined;
    if (turn?.speaker === "agent") {
      if (turn.id !== this.turnId) {
        this.turnId = turn.id;
        this.spoken = 0;
      }
      if (agentIsAudible) {
        this.spoken += elapsed;
        this.lastAudible = now;
      }
      // Once the agent has stopped for good, whatever the pacing has not reached yet is shown,
      // so a reply never ends half captioned.
      const length = [...turn.text].length;
      const isOver = this.spoken > 0 && !agentIsAnswering && now - this.lastAudible > FINISHED;
      const count = isOver ? length : Math.min(length, Math.floor((this.spoken / 1000) * PACE));
      const sentence = sentenceIn(turn.text, count);
      next = sentence ? { text: sentence, isPerson: false } : undefined;
    } else if (turn) {
      const sentence = sentenceIn(turn.text, [...turn.text].length);
      next = sentence ? { text: sentence, isPerson: true } : undefined;
    }
    if (next && (next.text !== this.caption?.text || next.isPerson !== this.caption.isPerson)) {
      this.caption = next;
      this.lastChange = now;
    }
    this.isVisible = this.caption !== undefined && (agentIsAudible || now - this.lastChange < LINGER);
  }
}

/**
 * The sentence being said once the first `count` characters of `text` have been: whole words
 * only, starting after the last sentence that has ended.
 */
export function sentenceIn(text: string, count: number): string {
  const characters = [...text];
  let said = characters.slice(0, count).join("");
  if (count < characters.length) {
    // Hold back a word until all of it has been said.
    const space = said.search(/\s\S*$/);
    said = space >= 0 ? said.slice(0, space) : "";
  }
  let start = 0;
  for (let index = 0; index < said.length - 1; index++) {
    const character = said[index]!;
    if ((".!?".includes(character) || character === "\n") && /\s/.test(said[index + 1]!)) {
      start = index + 1;
    }
  }
  return said.slice(start).trim();
}
