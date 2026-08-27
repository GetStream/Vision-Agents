"use client";

import { useQuery } from "@tanstack/react-query";
import { use } from "react";

import { DecisionLog } from "@/components/DecisionLog";
import { LatencyPanel } from "@/components/LatencyPanel";
import { SpeakingTimeline } from "@/components/SpeakingTimeline";
import { Transcript } from "@/components/Transcript";
import { VoicePanel } from "@/components/VoicePanel";
import { duration, Failure, PageHeading, Panel, Tile } from "@/components/ui";
import { router } from "@/lib/router";
import { useSession } from "@/lib/useSession";

export default function CallPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);

  const call = useQuery({
    queryKey: ["call", id],
    queryFn: () => router.call(id),
    refetchInterval: (query) => (query.state.data?.ended_at ? false : 15_000),
  });

  const running = Boolean(call.data && !call.data.ended_at);
  const session = useSession(id, running);

  // A finished call is read back; a running one arrives on the socket. The stored history
  // is fetched either way, because a page opened mid-call has missed everything so far.
  const events = useQuery({
    queryKey: ["call-events", id],
    queryFn: () => router.callEvents(id),
    refetchInterval: running ? 20_000 : false,
  });

  const timeline = useQuery({
    queryKey: ["call-timeline", id],
    queryFn: () => router.callTimeline(id),
    enabled: !running,
  });

  const transcript = useQuery({
    queryKey: ["call-transcript", id],
    queryFn: () => router.callTranscript(id),
    enabled: !running,
    retry: false,
  });

  if (call.isError) {
    return <Failure error={call.error} />;
  }
  if (!call.data) {
    return <p className="text-sm text-muted">Loading the call…</p>;
  }

  const who =
    call.data.direction === "inbound"
      ? (call.data.from_number ?? "unknown caller")
      : (call.data.to_number ?? "unknown number");

  return (
    <>
      <PageHeading
        title={who}
        description={`${call.data.direction} · started ${new Date(
          call.data.started_at,
        ).toLocaleString()}`}
        action={
          running ? (
            <span className="inline-flex items-center gap-2 rounded-full border border-line px-3 py-1 text-xs">
              <span
                className={`size-1.5 rounded-full ${
                  session.connected ? "bg-emerald-500" : "bg-amber-500"
                }`}
              />
              {session.connected ? "watching live" : "connecting"}
            </span>
          ) : null
        }
      />

      {running ? <VoicePanel className="mb-6" voice={session.voice} /> : null}

      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Tile
          label="Length"
          value={duration(call.data.started_at, call.data.ended_at)}
        />
        <Tile label="Agent" value={call.data.agent_id} />
        <Tile
          label="Score"
          value={call.data.review_score ? `${call.data.review_score}/5` : "—"}
          hint={call.data.review_notes ?? undefined}
        />
        <Tile
          label="Decisions"
          value={String(
            Math.max(events.data?.length ?? 0, session.decisions.length),
          )}
          hint="Judgements the conversation made"
        />
      </div>

      {call.data.summary ? (
        <Panel title="Summary" className="mb-6">
          <p className="px-4 py-3 text-sm leading-relaxed">
            {call.data.summary}
          </p>
        </Panel>
      ) : null}

      <SpeakingTimeline
        className="mb-6"
        startedAt={call.data.started_at}
        endedAt={call.data.ended_at}
        stored={events.data ?? []}
        live={session.decisions}
        fallback={timeline.data ?? []}
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <Transcript
          running={running}
          hearing={session.hearing}
          lines={session.lines}
          stored={transcript.data ?? []}
        />
        <DecisionLog stored={events.data ?? []} live={session.decisions} />
      </div>

      <LatencyPanel
        className="mt-6"
        live={session.timings}
        stored={timeline.data ?? []}
      />
    </>
  );
}
