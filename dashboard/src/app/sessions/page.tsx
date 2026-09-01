"use client";

import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { SessionTable } from "@/components/SessionTable";
import { Button, Empty, Failure, PageHeading, Panel } from "@/components/ui";
import { router } from "@/lib/router";

export default function Sessions() {
  const [live, setLive] = useState(false);

  const calls = useQuery({
    queryKey: ["calls", 50, live],
    queryFn: () => router.calls({ limit: 50, running: live || undefined }),
    refetchInterval: 10_000,
  });

  const configs = useQuery({ queryKey: ["configs"], queryFn: router.configs });

  const agents = useMemo(
    () =>
      Object.fromEntries(
        (configs.data ?? []).map((config) => [config.id, config.name]),
      ),
    [configs.data],
  );

  return (
    <>
      <PageHeading
        title="Sessions"
        description="Voice calls for now. SMS, Slack and other channels will show up here too."
        action={
          <Button variant="quiet" onClick={() => setLive((on) => !on)}>
            {live ? "Showing live" : "Live only"}
          </Button>
        }
      />

      {calls.isError ? <Failure error={calls.error} /> : null}

      <Panel title={live ? "Live sessions" : "Latest sessions"}>
        {calls.data?.length === 0 ? (
          <Empty>
            {live
              ? "Nothing is running. A session will appear here the moment it starts."
              : "No sessions yet. A call will appear here the moment it starts."}
          </Empty>
        ) : null}
        {calls.data?.length ? (
          <SessionTable calls={calls.data} agents={agents} />
        ) : null}
      </Panel>
    </>
  );
}
