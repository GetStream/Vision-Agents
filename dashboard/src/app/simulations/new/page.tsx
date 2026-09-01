"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { blank, SimulationForm } from "@/components/SimulationForm";
import { Button, Failure, PageHeading, Panel } from "@/components/ui";
import { router, type SimulationRequest } from "@/lib/router";

export default function NewSimulation() {
  const client = useQueryClient();
  const navigate = useRouter();
  const [simulation, setSimulation] = useState<SimulationRequest>(blank);

  const configs = useQuery({ queryKey: ["configs"], queryFn: router.configs });

  const create = useMutation({
    mutationFn: () => router.createSimulation(simulation),
    onSuccess: (created) => {
      client.invalidateQueries({ queryKey: ["simulations"] });
      navigate.push(`/simulations/${created.id}`);
    },
  });

  const ready =
    Boolean(simulation.name) &&
    Boolean(simulation.config_id) &&
    Boolean(simulation.scenario) &&
    Boolean(simulation.assertion);

  return (
    <>
      <PageHeading
        title="New simulation"
        description="Say what to ask, who to ask it of, and what has to be true at the end."
        action={
          <Link href="/simulations" className="text-sm text-muted hover:underline">
            Back to simulations
          </Link>
        }
      />

      <Panel>
        <form
          className="space-y-4 px-4 py-4"
          onSubmit={(event) => {
            event.preventDefault();
            create.mutate();
          }}
        >
          <SimulationForm
            simulation={simulation}
            configs={configs.data ?? []}
            onChange={setSimulation}
          />

          {create.error ? <Failure error={create.error} /> : null}

          <div className="flex gap-2">
            <Button type="submit" disabled={create.isPending || !ready}>
              {create.isPending ? "Creating…" : "Create simulation"}
            </Button>
          </div>
        </form>
      </Panel>
    </>
  );
}
