"use client";

import { useEffect, useRef, useState } from "react";

import { sessionSocket, type DecisionKind } from "./router";

/** A frame off the session socket. Every one carries a type and that type's fields. */
export type Frame = { type: string } & Record<string, unknown>;

export type Participant = { id?: string; user_id?: string; name?: string };

/** What the caller is saying right now, before anything has been decided about it. */
export type Hearing = {
  participant: Participant;
  text: string;
};

/** One judgement the conversation made, as it happens. */
export type Decision = {
  at: string;
  kind: DecisionKind;
  reason: string;
  turn_id: string;
  participant: Participant;
  said: string;
  latency_ms: number;
};

/** A settled line, from either side of the call. */
export type Line = {
  at: number;
  agent: boolean;
  speaker: string;
  text: string;
};

export type Timing = {
  turn_id: string;
  started_at: string;
  stt_latency_ms: number;
  llm_ttft_ms: number;
  tts_ttfb_ms: number;
  roundtrip_ms: number;
  interrupted: boolean;
};

export type Live = {
  /** connected is false while the socket is opening, and again once the call ends. */
  connected: boolean;
  /** ended is true once the agent has left, which is a finished call rather than a lost socket. */
  ended: boolean;
  hearing: Hearing | null;
  decisions: Decision[];
  lines: Line[];
  timings: Timing[];
};

const empty: Live = {
  connected: false,
  ended: false,
  hearing: null,
  decisions: [],
  lines: [],
  timings: [],
};

/** How much of a live call is kept in the browser. An hour of one is thousands of rows. */
const keep = 500;

function tail<T>(existing: T[], added: T): T[] {
  const next = [...existing, added];
  return next.length > keep ? next.slice(next.length - keep) : next;
}

function speakerOf(participant: Participant | undefined): string {
  return participant?.name || participant?.user_id || participant?.id || "caller";
}

/**
 * useSession watches a call as it happens.
 *
 * A call's id is its session's id, so a running call needs no endpoint of its own: this is
 * the same socket the SDK holds a conversation over, read rather than answered. It asks
 * for interim transcripts, which the SDK does not, because watching an agent hear is the
 * point of watching at all.
 *
 * `live` is false for a call that has already ended, which reconnects nothing and leaves
 * the page to the stored history.
 */
export function useSession(callID: string, live: boolean): Live {
  // The call the state belongs to travels with it, so switching calls shows the new call's
  // silence rather than the old call's transcript until the first frame arrives.
  const [state, setState] = useState<{ of: string; live: Live }>({
    of: callID,
    live: empty,
  });
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!live) {
      return;
    }

    const connection = new WebSocket(sessionSocket(callID));
    socket.current = connection;

    const update = (change: (current: Live) => Live) =>
      setState((current) => ({
        of: callID,
        live: change(current.of === callID ? current.live : empty),
      }));

    connection.onopen = () =>
      update((current) => ({ ...current, connected: true }));
    connection.onclose = () =>
      update((current) => ({ ...current, connected: false }));

    connection.onmessage = (message) => {
      let frame: Frame;
      try {
        frame = JSON.parse(message.data as string) as Frame;
      } catch {
        return;
      }
      update((current) => apply(current, frame));
    };

    return () => {
      connection.onmessage = null;
      connection.close();
      socket.current = null;
    };
  }, [callID, live]);

  return live && state.of === callID ? state.live : empty;
}

function apply(current: Live, frame: Frame): Live {
  switch (frame.type) {
    case "hearing":
      return {
        ...current,
        hearing: {
          participant: frame.participant as Participant,
          text: String(frame.text ?? ""),
        },
      };

    case "heard":
      return {
        ...current,
        hearing: null,
        lines: tail(current.lines, {
          at: Date.now(),
          agent: false,
          speaker: speakerOf(frame.participant as Participant),
          text: String(frame.text ?? ""),
        }),
      };

    case "responded": {
      const said = String(frame.text ?? "");
      if (said === "") {
        return current;
      }
      return {
        ...current,
        lines: tail(current.lines, {
          at: Date.now(),
          agent: true,
          speaker: "agent",
          text: said,
        }),
      };
    }

    case "decision":
      return {
        ...current,
        decisions: tail(current.decisions, frame as unknown as Decision),
      };

    case "turn":
      return {
        ...current,
        timings: tail(current.timings, frame as unknown as Timing),
      };

    case "left":
      return { ...current, ended: true, hearing: null };

    default:
      return current;
  }
}
