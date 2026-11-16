import { useEffect, useRef, useState } from 'react';

import * as Speech from 'expo-speech';

import { useStableCallback } from 'stream-chat-expo';

import {
  dictationStore,
  useDictation,
  useDictationState,
  useDictationTranscript,
  type VoiceAgentState,
} from 'stream-chat-expo/ai-components';

import { useAppContext } from '../../contexts/AppContext';
import { useSendAIMessage } from '../../hooks/useSendAIMessage';

/** A turn the agent is never seen working on ends after this long. */
const TURN_TIMEOUT_MS = 20_000;

// Marks voice mode's dictation as its own, so the composer underneath ignores it.
const DICTATION_SOURCE = { source: 'voice-mode' };

/** Markdown read aloud as plain words. */
const toSpeech = (text: string) =>
  text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/!\[[^\]]*]\([^)]*\)/g, ' ')
    .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
    .replace(/[*_#>`~|]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

/**
 * A hands-free conversation with the AI agent: it listens with dictation, sends what it heard,
 * waits for the agent to answer, reads the answer aloud and listens again.
 *
 * Returns what the agent is doing, for `VoiceGlow`, and whether the user is being listened to,
 * for `VoiceWave`.
 */
export const useVoiceConversation = (active: boolean) => {
  const [phase, setPhase] = useState<VoiceAgentState>('idle');
  const { start, cancel } = useDictation(undefined, DICTATION_SOURCE);
  const { state: dictationState, error } = useDictationState();
  const { transcript } = useDictationTranscript();
  const send = useSendAIMessage();
  const { agentState, chat } = useAppContext();
  // The agent answering, from the session's socket.
  const isWorking = agentState.kind === 'responding' || agentState.kind === 'working';

  // The messages already in the channel when the question was sent, so the reply is found by
  // identity rather than by comparing the server's timestamps with the device's clock.
  const turn = useRef<{ seenIds: Set<string>; sawWorking: boolean } | null>(null);
  const timeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  // An error left over from an earlier turn, ignored until dictation starts again and clears it.
  const staleError = useRef(dictationStore.getLatestValue().error);

  const listen = useStableCallback(() => {
    if (!active) return;
    turn.current = null;
    staleError.current = dictationStore.getLatestValue().error;
    setPhase('connecting');
    void start();
  });

  const latestReply = useStableCallback(() => {
    const seenIds = turn.current?.seenIds ?? new Set<string>();
    const messages = chat?.state.getLatestValue().messages ?? [];
    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      if (seenIds.has(message.id)) break;
      if (message.custom?.ai_generated && message.text) return message.text;
    }
    return null;
  });

  /** Ends the agent's turn: reads its answer aloud if there is one, then listens again. */
  const finishTurn = useStableCallback(() => {
    if (timeout.current) clearTimeout(timeout.current);
    timeout.current = null;
    const reply = latestReply();
    turn.current = null;
    if (!active) return;
    if (!reply) {
      listen();
      return;
    }
    setPhase('speaking');
    Speech.speak(toSpeech(reply), {
      onDone: listen,
      onError: listen,
      // Its own audio session on iOS, so the reply is heard whatever dictation left the app's in.
      useApplicationAudioSession: false,
    });
  });

  // Starts listening when voice mode opens, and stops everything when it closes.
  useEffect(() => {
    if (!active) return;
    listen();
    return () => {
      if (timeout.current) clearTimeout(timeout.current);
      timeout.current = null;
      turn.current = null;
      cancel();
      void Speech.stop();
      setPhase('idle');
    };
  }, [active, cancel, listen]);

  // Follows dictation: listening while it records, and sends what it heard once it stops.
  useEffect(() => {
    if (!active || turn.current) return;
    if (dictationState === 'listening') {
      setPhase('listening');
      return;
    }
    if (dictationState !== 'idle' || (phase !== 'listening' && phase !== 'connecting')) return;
    if (error && error !== staleError.current) {
      setPhase('idle');
      return;
    }
    if (phase === 'connecting') return;
    const text = transcript.trim();
    if (!text) {
      // Nothing was heard; keep listening.
      listen();
      return;
    }
    turn.current = {
      sawWorking: false,
      seenIds: new Set((chat?.state.getLatestValue().messages ?? []).map((message) => message.id)),
    };
    setPhase('thinking');
    void send({ text });
    timeout.current = setTimeout(() => {
      if (!turn.current?.sawWorking) finishTurn();
    }, TURN_TIMEOUT_MS);
  }, [active, chat, dictationState, error, finishTurn, listen, phase, send, transcript]);

  // Follows the agent: its turn ends once it has been seen working and is done.
  useEffect(() => {
    const current = turn.current;
    if (!active || !current) return;
    if (isWorking) {
      current.sawWorking = true;
    } else if (current.sawWorking) {
      finishTurn();
    }
  }, [active, isWorking, finishTurn]);

  return {
    // Only an error from the current attempt, not one left over while dictation restarts.
    error: phase === 'idle' ? error : null,
    listen,
    phase,
    userIsTalking: phase === 'listening',
  };
};
