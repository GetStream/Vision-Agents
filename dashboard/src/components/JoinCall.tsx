"use client";

import {
  createSoundDetector,
  ParticipantsAudio,
  StreamCall,
  StreamVideo,
  StreamVideoClient,
  useCallStateHooks,
  type Call,
} from "@stream-io/video-react-sdk";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { Button } from "@/components/ui";
import { router, type CallToken } from "@/lib/router";

/**
 * JoinCall puts the person reading a call into it.
 *
 * Nothing happens until they ask for it: the token is minted, the client connected and
 * the microphone opened on the click, not on the render. A dashboard that took the
 * microphone to show a page would be a surprise, and a call is being listened to either
 * way. The level meter is the same bargain - it runs only while it is asked for.
 *
 * The speaker is left on the system default, which is the one the reader already has
 * their audio on.
 */
export function JoinCall({
  callID,
  className = "",
}: {
  callID: string;
  className?: string;
}) {
  const [credentials, setCredentials] = useState<CallToken>();
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState<string>();
  const [chosen, setChosen] = useState("");
  const [testing, setTesting] = useState(false);
  const preview = useMicPreview(chosen, testing);
  const mics = useMics(preview);
  const level = useAudioLevel(preview);

  const join = async () => {
    setJoining(true);
    setError(undefined);
    try {
      setCredentials(await router.callToken(callID));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setJoining(false);
    }
  };

  if (credentials) {
    return (
      <Stage
        credentials={credentials}
        callID={callID}
        className={className}
        mic={chosen}
        onLeave={() => setCredentials(undefined)}
      />
    );
  }

  return (
    <div className={className}>
      <Bar>
        <Row>
          <Button onClick={join} disabled={joining}>
            {joining ? "Joining…" : "Join call"}
          </Button>
        </Row>
        <MicPicker
          devices={mics}
          value={chosen}
          onPick={setChosen}
          level={level}
          testing={testing}
          onTest={() => setTesting((on) => !on)}
        />
      </Bar>
      <p className="mt-2 px-1 text-xs text-muted">
        {error ?? "You join with your microphone on and no camera."}
      </p>
    </div>
  );
}

/** Bar keeps the strip identical before and after joining, so nothing jumps on the click. */
function Bar({ children }: { children: ReactNode }) {
  return (
    <div className="space-y-3 rounded-xl border border-line bg-surface px-4 py-3">
      {children}
    </div>
  );
}

function Row({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center gap-3">{children}</div>;
}

/**
 * MicPicker is the dashboard's own control rather than the SDK's, so the call bar matches
 * the page it sits on instead of bringing its own theme with it.
 *
 * The meter is under the list rather than beside every row: only the picked microphone is
 * open, and a bar next to a device nothing is listening through would read as silence.
 */
function MicPicker({
  devices,
  value,
  onPick,
  level,
  testing = false,
  onTest,
}: {
  devices: MediaDeviceInfo[];
  value: string;
  onPick: (deviceId: string) => void;
  level: number;
  testing?: boolean;
  onTest?: () => void;
}) {
  return (
    <div>
      <div className="px-1 text-xs uppercase tracking-wide text-muted">
        Microphone
      </div>
      <div className="mt-1.5" role="radiogroup" aria-label="Microphone">
        <Choice
          label="System default"
          selected={value === ""}
          onPick={() => onPick("")}
        />
        {devices.map((device, index) => (
          <Choice
            key={device.deviceId}
            label={device.label || `Microphone ${index + 1}`}
            selected={value === device.deviceId}
            onPick={() => onPick(device.deviceId)}
          />
        ))}
      </div>
      <div className="mt-2 flex items-center gap-2.5 border-t border-line pt-2.5">
        <MicIcon />
        <Level level={level} />
        {onTest ? (
          <button
            type="button"
            onClick={onTest}
            className="shrink-0 rounded-lg border border-line px-2 py-0.5 text-xs text-muted transition hover:bg-line/40"
          >
            {testing ? "Stop" : "Test"}
          </button>
        ) : null}
      </div>
    </div>
  );
}

function Choice({
  label,
  selected,
  onPick,
}: {
  label: string;
  selected: boolean;
  onPick: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      onClick={onPick}
      className={`flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left text-xs transition ${
        selected ? "bg-line/50" : "hover:bg-line/30"
      }`}
    >
      <span
        className={`grid size-3.5 shrink-0 place-items-center rounded-full border ${
          selected ? "border-foreground" : "border-line"
        }`}
      >
        {selected ? <span className="size-1.5 rounded-full bg-foreground" /> : null}
      </span>
      <span className="min-w-0 flex-1 truncate">{label}</span>
    </button>
  );
}

/** Level is how loud the open microphone is right now, in segments so it reads at a glance. */
function Level({ level }: { level: number }) {
  const segments = 16;
  const lit = Math.round((Math.min(Math.max(level, 0), 100) / 100) * segments);

  return (
    <div className="flex min-w-0 flex-1 items-center gap-0.5" aria-hidden>
      {Array.from({ length: segments }, (_, index) => (
        <span
          key={index}
          className={`h-2 flex-1 rounded-sm ${
            index < lit ? "bg-emerald-500" : "bg-line"
          }`}
        />
      ))}
    </div>
  );
}

function MicIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      className="size-4 shrink-0 text-muted"
      aria-hidden
    >
      <path d="M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Z" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
    </svg>
  );
}

/**
 * useMics lists what is plugged in before there is a call to ask through.
 *
 * A browser withholds the labels until it has been given permission once, so the list is
 * read again when a stream opens, and an unnamed device is numbered rather than blank.
 */
function useMics(reread?: MediaStream): MediaDeviceInfo[] {
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);

  useEffect(() => {
    const media = navigator.mediaDevices;
    if (!media) {
      return;
    }
    const read = () => {
      media
        .enumerateDevices()
        .then((all) =>
          setMics(all.filter((device) => device.kind === "audioinput")),
        )
        .catch(console.error);
    };
    read();
    media.addEventListener("devicechange", read);
    return () => media.removeEventListener("devicechange", read);
  }, [reread]);

  return mics;
}

/** useMicPreview opens the picked microphone only for as long as the meter is wanted. */
function useMicPreview(
  deviceId: string,
  active: boolean,
): MediaStream | undefined {
  const [stream, setStream] = useState<MediaStream>();

  useEffect(() => {
    if (!active) {
      return;
    }
    let opened: MediaStream | undefined;
    let wanted = true;

    navigator.mediaDevices
      .getUserMedia({
        audio: deviceId ? { deviceId: { exact: deviceId } } : true,
      })
      .then((media) => {
        opened = media;
        if (wanted) {
          setStream(media);
        } else {
          media.getTracks().forEach((track) => track.stop());
        }
      })
      .catch(console.error);

    return () => {
      wanted = false;
      opened?.getTracks().forEach((track) => track.stop());
      setStream(undefined);
    };
  }, [deviceId, active]);

  return stream;
}

/** useAudioLevel reads a stream's loudness as a percentage, without taking the stream over. */
function useAudioLevel(stream: MediaStream | undefined): number {
  const [level, setLevel] = useState(0);

  useEffect(() => {
    if (!stream) {
      return;
    }
    const stop = createSoundDetector(
      stream,
      ({ audioLevel }) => setLevel(audioLevel),
      // The stream is the call's or the preview's; stopping the meter must not close it.
      { detectionFrequencyInMs: 80, destroyStreamOnStop: false },
    );
    return () => void stop();
  }, [stream]);

  // Read as flat with nothing open rather than holding the last reading it took.
  return stream ? level : 0;
}

/**
 * Stage holds the connection for as long as it is mounted, which is what leaving is: the
 * parent drops the credentials and this goes with them.
 */
function Stage({
  credentials,
  callID,
  className,
  mic,
  onLeave,
}: {
  credentials: CallToken;
  callID: string;
  className: string;
  mic: string;
  onLeave: () => void;
}) {
  const [stage, setStage] = useState<{ client: StreamVideoClient; call: Call }>();
  const [error, setError] = useState<string>();

  const { api_key: apiKey, user_id: userID, user_name: userName } = credentials;
  const { call_id: streamCallID, call_type: callType } = credentials;

  // What was picked before joining is only read on the way in. Afterwards the controls
  // drive the call's own device managers, and re-reading this would fight them.
  const picked = useRef(mic);

  useEffect(() => {
    // The client is built here rather than kept between mounts: it is disconnected on the
    // way out, and a disconnected client cannot be joined with again.
    //
    // The token provider is defined in here for the same reason - one built during render
    // changes identity every render, and would rebuild the client each time. Asking the
    // router again is also what lets an expired token recover on its own.
    const tokenProvider = async () =>
      (await router.callToken(callID, { user_id: userID })).token;

    const client = new StreamVideoClient({
      apiKey,
      user: { id: userID, name: userName },
      tokenProvider,
    });
    const call = client.call(callType, streamCallID);
    let active = true;

    const enter = async () => {
      // Video off before joining rather than after, so the camera is never published and
      // never asked for.
      await call.camera.disable();
      await call.join({ create: true });
      if (!active) {
        return;
      }
      setStage({ client, call });

      // The microphone is its own attempt: a refused permission should leave the call
      // standing, with the controls there to retry from.
      try {
        if (picked.current) {
          await call.microphone.select(picked.current);
        }
        await call.microphone.enable();
      } catch (cause) {
        console.error(cause);
      }
    };

    enter().catch((cause: unknown) => {
      if (active) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    });

    return () => {
      active = false;
      setStage(undefined);
      call
        .leave()
        .catch(() => {})
        .finally(() => client.disconnectUser().catch(console.error));
    };
  }, [apiKey, userID, userName, callID, callType, streamCallID]);

  if (error) {
    return (
      <div className={className}>
        <Bar>
          <Row>
            <Button variant="quiet" onClick={onLeave}>
              Close
            </Button>
            <p className="text-sm text-muted">Could not join: {error}</p>
          </Row>
        </Bar>
      </div>
    );
  }

  if (!stage) {
    return (
      <div className={className}>
        <Bar>
          <Row>
            <p className="text-sm text-muted">Connecting…</p>
          </Row>
        </Bar>
      </div>
    );
  }

  return (
    <div className={className}>
      <StreamVideo client={stage.client}>
        <StreamCall call={stage.call}>
          <Controls onLeave={onLeave} />
        </StreamCall>
      </StreamVideo>
    </div>
  );
}

/** Controls is the same strip as before joining, wired to the call instead of to a guess. */
function Controls({ onLeave }: { onLeave: () => void }) {
  const { useMicrophoneState, useRemoteParticipants } = useCallStateHooks();
  const mic = useMicrophoneState();
  const remote = useRemoteParticipants();
  const level = useAudioLevel(mic.mediaStream);

  return (
    <>
      {/* Nothing is heard without this. No video is rendered on a voice call, so there is
          no participant view holding the agent's audio track. */}
      <ParticipantsAudio participants={remote} />
      <Bar>
        <Row>
          <Button variant="danger" onClick={onLeave}>
            Leave
          </Button>
          <Button variant="quiet" onClick={() => void mic.microphone.toggle()}>
            {mic.isMute ? "Unmute" : "Mute"}
          </Button>
        </Row>
        <MicPicker
          devices={mic.devices}
          value={mic.selectedDevice ?? ""}
          onPick={(deviceId) => void mic.microphone.select(deviceId || undefined)}
          level={level}
        />
      </Bar>
    </>
  );
}
