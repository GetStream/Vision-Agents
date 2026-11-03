import type { ToolApprovalRequest, Turn } from "@stream-io/calypso";
import type { VoiceSnapshot } from "@stream-io/calypso/video";
import type { Call, StreamVideoParticipant } from "@stream-io/video-client";
import { useEffect, useRef, useState } from "react";
import {
  AIToolApprovalCard,
  useAudioLevel,
  VoiceGlow,
  type AIToolApprovalState,
  type VoiceAgentState,
} from "stream-chat-react/ai-components";

import type { AgentChat } from "./agentChat";
import { Captions, type Caption } from "./captions";
import type { ChatScreen } from "./chatScreen";
import { Store, useSelected, useStore } from "./store";
import { VoiceMeter } from "./voiceMeter";
import { Presence, phaseLabel, troubleText, type Phase } from "./voicePresence";

/** The colors the glow takes while the person is talking. The agent's are Stream's. */
const PERSON_COLORS = ["#ff2626"];

const NOT_JOINED = new Store<VoiceSnapshot>({ muted: false, cameraEnabled: false, status: "connecting" });

/**
 * Talking to Juno, over the conversation that was open: in its pane beside the sidebar on a wide
 * window, over everything on a narrow one.
 *
 * What is said on the call is written into the conversation's channel, so the chat shows it once
 * voice mode is closed. The session's socket stays open the whole time and runs this page's tools,
 * and a tool that asks first asks here, on a card over the call.
 *
 * The camera button shows Juno what the camera is pointing at: the session answers the vision
 * skill's requests to look with what the camera showed, and the preview is that view.
 */
export function VoiceMode({ screen, chat }: { screen: ChatScreen; chat: AgentChat }) {
  const { voice, voiceFailure, voiceOpenedAt } = useStore(screen);
  const snapshot = useStore<VoiceSnapshot>(voice ?? NOT_JOINED);
  // Only the parts this shows, so the reply streaming into the transcript redraws none of it.
  const conversation = chat.session.conversation;
  const approvals = useSelected(conversation, (each) => each.approvals);
  const failureAt = useSelected(conversation, (each) => each.failureAt);
  const failureContext = useSelected(conversation, (each) => each.failureContext);
  // What the agent cannot do on the call right now: the call stays up when it fails, so without
  // this voice mode would go on saying it is listening to someone it cannot hear. A failure from
  // before voice mode opened belongs to the chat.
  const trouble = failureAt > 0 && failureAt >= voiceOpenedAt ? troubleText(failureContext) : undefined;
  const call = snapshot.call;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") screen.hangUp();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [screen]);

  const reason = voiceFailure ?? snapshot.failure?.message ?? trouble;
  const request = approvals[0];

  return (
    <div className="voice-mode" role="dialog" aria-label="Talking to Juno">
      <div className="voice-top">
        <button aria-label="Back to chat" className="icon-button" onClick={() => screen.hangUp()} type="button">
          ✕
        </button>
      </div>
      {call && snapshot.cameraEnabled && <CameraPreview call={call} />}
      <AgentPresence call={call} chat={chat} isStarting={!voice && !voiceFailure} />
      {request && <ApprovalCard chat={chat} key={request.id} request={request} />}
      {reason && <p className="voice-failure">{reason}</p>}
      {call && <RemoteAudio call={call} />}
      {voice && (
        <div className="call-controls">
          <button
            aria-label={snapshot.cameraEnabled ? "Stop showing Juno" : "Show Juno the camera"}
            className="control"
            disabled={!call}
            onClick={() => void screen.setCameraEnabled(!snapshot.cameraEnabled)}
            type="button"
          >
            {snapshot.cameraEnabled ? "Camera on" : "Camera off"}
          </button>
          <button
            aria-label={snapshot.muted ? "Unmute" : "Mute"}
            className="control"
            disabled={!call}
            onClick={() => void screen.setMuted(!snapshot.muted)}
            type="button"
          >
            {snapshot.muted ? "Unmute" : "Mute"}
          </button>
          <button aria-label="Hang up" className="control hang-up" onClick={() => screen.hangUp()} type="button">
            Hang up
          </button>
        </div>
      )}
    </div>
  );
}

/**
 * The glow, moving with whoever is talking, a word for what the agent is doing, and the captions.
 *
 * While the agent speaks, the glow moves with its voice in Stream's colors; while the person
 * speaks, with theirs in red. The levels are read thirty times a second, into a ref the glow reads
 * once a frame, so they redraw the glow and nothing else.
 */
function AgentPresence({ call, chat, isStarting }: { call: Call | undefined; chat: AgentChat; isStarting: boolean }) {
  const [phase, setPhase] = useState<Phase>({ kind: "ready" });
  const [caption, setCaption] = useState<{ caption: Caption | undefined; visible: boolean }>({
    caption: undefined,
    visible: false,
  });
  const level = useRef<number | undefined>(undefined);
  const [agentStream, setAgentStream] = useState<MediaStream | undefined>(undefined);
  const [micStream, setMicStream] = useState<MediaStream | undefined>(undefined);
  const agentLevel = useAudioLevel(agentStream);
  const personLevel = useAudioLevel(micStream);

  useEffect(() => {
    if (!call) return;
    // The agent is the call's other participant. Its track can change when the call
    // resubscribes, so the level follows whichever one is current.
    const remote = call.state.remoteParticipants$.subscribe((participants) =>
      setAgentStream(participants.find((each) => each.audioStream)?.audioStream),
    );
    const mic = call.microphone.state.mediaStream$.subscribe((stream) => setMicStream(stream ?? undefined));
    return () => {
      remote.unsubscribe();
      mic.unsubscribe();
    };
  }, [call]);

  useEffect(() => {
    setPhase({ kind: "ready" });
    if (!call) return;
    const agentMeter = new VoiceMeter(0.04);
    const personMeter = new VoiceMeter(0.02);
    const presence = new Presence(performance.now());
    const captions = new Captions();
    // The call carries on the chat's conversation, so what was written before it is already
    // there. Only what is said on the call moves the glow and the captions.
    const before = new Set(chat.session.conversation.get().turns.map((turn) => turn.id));
    let last = performance.now();
    let shown: Phase = presence.phase;

    const timer = setInterval(() => {
      const now = performance.now();
      const agent = agentMeter.read(agentLevel(), now);
      const person = personMeter.read(personLevel(), now);
      const turns = chat.session.conversation.get().turns;
      const latest: Turn | undefined = turns[turns.length - 1];
      const turn = latest && !before.has(latest.id) ? latest : undefined;
      const state = chat.session.state.get();

      const next = presence.step({
        now,
        agentAudible: agent !== undefined,
        personSpeaking: call.state.localParticipant?.isSpeaking === true,
        state: state.kind,
        skills: state.skills,
        turn,
      });
      if (next !== shown) {
        shown = next;
        setPhase(next);
      }
      level.current = next.kind === "speaking" ? (agent ?? 0) : next.kind === "listening" ? (person ?? 0) : undefined;

      captions.update({
        turn,
        agentIsAudible: agent !== undefined,
        agentIsAnswering: state.kind === "responding" || state.kind === "working",
        now,
        elapsed: now - last,
      });
      last = now;
      setCaption((was) =>
        was.caption === captions.caption && was.visible === captions.isVisible
          ? was
          : { caption: captions.caption, visible: captions.isVisible },
      );
    }, 33);
    return () => clearInterval(timer);
  }, [call, chat, agentLevel, personLevel]);

  const state: VoiceAgentState = !call ? (isStarting ? "connecting" : "idle") : glowState(phase);
  const status = !call ? (isStarting ? "Connecting" : "") : phaseLabel(phase);

  return (
    <div className="presence">
      <VoiceGlow
        audioLevel={() => level.current}
        className="glow"
        colors={phase.kind === "listening" ? PERSON_COLORS : undefined}
        state={state}
      />
      {/* The line keeps its height so the glow does not move as it changes. */}
      <p className="voice-status">{status || " "}</p>
      <div className="caption-band">
        <p className={`caption${caption.caption?.isPerson ? " person" : ""}${caption.visible ? " visible" : ""}`}>
          {caption.caption?.text ?? ""}
        </p>
      </div>
    </div>
  );
}

function glowState(phase: Phase): VoiceAgentState {
  switch (phase.kind) {
    case "ready":
      return "idle";
    case "listening":
      return "listening";
    case "thinking":
    case "working":
      return "thinking";
    case "speaking":
      return "speaking";
  }
}

/** What the camera shows, which is what Juno sees when it looks. */
function CameraPreview({ call }: { call: Call }) {
  const video = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const subscription = call.camera.state.mediaStream$.subscribe((stream) => {
      if (video.current) video.current.srcObject = stream ?? null;
    });
    return () => subscription.unsubscribe();
  }, [call]);
  return <video aria-label="What Juno sees" autoPlay className="camera-preview" muted playsInline ref={video} />;
}

/** Plays the agent: an audio element bound to each remote participant on the call. */
function RemoteAudio({ call }: { call: Call }) {
  const [participants, setParticipants] = useState<readonly StreamVideoParticipant[]>([]);
  useEffect(() => {
    const subscription = call.state.remoteParticipants$.subscribe(setParticipants);
    return () => subscription.unsubscribe();
  }, [call]);
  return (
    <>
      {participants.map((participant) => (
        <ParticipantAudio call={call} key={participant.sessionId} sessionId={participant.sessionId} />
      ))}
    </>
  );
}

function ParticipantAudio({ call, sessionId }: { call: Call; sessionId: string }) {
  const audio = useRef<HTMLAudioElement>(null);
  useEffect(() => {
    if (!audio.current) return;
    return call.bindAudioElement(audio.current, sessionId);
  }, [call, sessionId]);
  return <audio autoPlay ref={audio} />;
}

/**
 * A tool's question, asked over the call with Stream's approval card. The tool runs only once it
 * is allowed, and the agent is told when it is not.
 */
function ApprovalCard({ request, chat }: { request: ToolApprovalRequest; chat: AgentChat }) {
  const [state, setState] = useState<AIToolApprovalState>({ isSending: false, failed: false });
  return (
    <AIToolApprovalCard
      approval={{
        title: request.approval.title,
        ...(request.approval.message ? { message: request.approval.message } : {}),
        ...(request.reason ? { reason: request.reason } : {}),
        ...(request.approval.allowTitle ? { allowTitle: request.approval.allowTitle } : {}),
        ...(request.approval.declineTitle ? { declineTitle: request.approval.declineTitle } : {}),
      }}
      className="voice-approval"
      onDecide={(allowed) => {
        setState({ isSending: true, failed: false });
        try {
          chat.session.decide(request.id, { allowed });
        } catch {
          setState({ isSending: false, failed: true });
        }
      }}
      state={state}
    />
  );
}
