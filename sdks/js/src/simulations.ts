import type { Client, Schemas } from "./client.js";

/**
 * Conversations to put an agent through, and something that has to be true at the end of each.
 *
 * Server side only: what an agent is tested on is the app's business, and a page asking gets
 * a 403.
 *
 * ```ts
 * const simulation = await api.simulations.create({ name, config_id, scenario, assertion });
 * let run = await api.simulations.run(simulation.id);
 * run = await api.simulations.runs.get(run.id);
 * ```
 */
export class Simulations {
  /** What the simulations have come to. */
  readonly runs: SimulationRuns;
  private readonly client: Client;

  constructor(client: Client) {
    this.client = client;
    this.runs = new SimulationRuns(client);
  }

  /** Writes a simulation down. Nothing is run until `run` is called. */
  create(simulation: Schemas["SimulationRequest"]): Promise<Schemas["Simulation"]> {
    return this.client.post("/v1/agents/simulations", { body: simulation });
  }

  get(id: string): Promise<Schemas["Simulation"]> {
    return this.client.get("/v1/agents/simulations/{id}", { path: { id } });
  }

  list(): Promise<readonly Schemas["Simulation"][]> {
    return this.client.get("/v1/agents/simulations");
  }

  /**
   * Replaces a simulation: every field is written, so pass what it now asks rather than what
   * changed. The runs it already has keep their own copy of what they tested.
   */
  update(id: string, simulation: Schemas["SimulationRequest"]): Promise<Schemas["Simulation"]> {
    return this.client.put("/v1/agents/simulations/{id}", { path: { id }, body: simulation });
  }

  /** Deletes a simulation. The runs that named it are kept. */
  delete(id: string): Promise<void> {
    return this.client.delete("/v1/agents/simulations/{id}", { path: { id } });
  }

  /**
   * Starts a run. It answers once the run is written rather than once it is over, so read it
   * again with `runs.get` until its state is no longer `running`.
   */
  run(id: string): Promise<Schemas["SimulationRun"]> {
    return this.client.post("/v1/agents/simulations/{id}/run", { path: { id } });
  }
}

/** Which runs to list. */
export interface SimulationRunQuery {
  /** Only the runs of this simulation; left out, every simulation's. */
  readonly simulationId?: string;
  readonly limit?: number;
}

/** The runs of every simulation, newest first. */
export class SimulationRuns {
  private readonly client: Client;

  constructor(client: Client) {
    this.client = client;
  }

  /** One run, with the conversations it had. */
  get(id: string): Promise<Schemas["SimulationRun"]> {
    return this.client.get("/v1/agents/simulation-runs/{id}", { path: { id } });
  }

  list(query: SimulationRunQuery = {}): Promise<readonly Schemas["SimulationRun"][]> {
    return this.client.get("/v1/agents/simulation-runs", {
      query: { simulation_id: query.simulationId, limit: query.limit },
    });
  }

  /** Stops a run in progress, ending the conversations still in flight. */
  cancel(id: string): Promise<Schemas["SimulationRun"]> {
    return this.client.post("/v1/agents/simulation-runs/{id}/cancel", { path: { id } });
  }
}
