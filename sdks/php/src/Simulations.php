<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\Simulation;
use GetStream\VisionAgents\Generated\SimulationRequest;
use GetStream\VisionAgents\Generated\SimulationRun;

/**
 * Conversations to put an agent through, and something that has to be true at the end of each.
 *
 *     $simulation = $client->simulations->create(new SimulationRequest(
 *         name: 'lunch order', configId: $config->id, scenario: $scenario, assertion: $assertion,
 *     ));
 *     $run = $client->simulations->run($simulation->id);
 *     $run = $client->simulations->runs->get($run->id);
 *
 * Server side only: what an agent is tested on is the app's business.
 */
final readonly class Simulations
{
    public SimulationRuns $runs;

    public function __construct(private Client $client)
    {
        $this->runs = new SimulationRuns($client);
    }

    /**
     * Writes a simulation down. Nothing is run until `run` is called.
     */
    public function create(SimulationRequest $simulation): Simulation
    {
        return Simulation::fromArray(Json::asObject($this->client->post('/v1/agents/simulations', body: $simulation->toArray())));
    }

    public function get(string $id): Simulation
    {
        return Simulation::fromArray(Json::asObject($this->client->get('/v1/agents/simulations/{id}', ['id' => $id])));
    }

    /**
     * @return list<Simulation>
     */
    public function list(): array
    {
        return array_map(Simulation::fromArray(...), Json::objects(['rows' => $this->client->get('/v1/agents/simulations')], 'rows'));
    }

    /**
     * Replaces a simulation: every field is written, so pass what it now asks rather than what
     * changed. The runs it already has keep their own copy of what they tested.
     */
    public function update(string $id, SimulationRequest $simulation): Simulation
    {
        return Simulation::fromArray(Json::asObject($this->client->put('/v1/agents/simulations/{id}', ['id' => $id], $simulation->toArray())));
    }

    /**
     * Deletes a simulation. The runs that named it are kept.
     */
    public function delete(string $id): void
    {
        $this->client->delete('/v1/agents/simulations/{id}', ['id' => $id]);
    }

    /**
     * Starts a run. It returns once the run is written rather than once it is over, so read it
     * again with `runs->get` until its state is no longer `running`.
     */
    public function run(string $id): SimulationRun
    {
        return SimulationRun::fromArray(Json::asObject($this->client->post('/v1/agents/simulations/{id}/run', ['id' => $id])));
    }
}
