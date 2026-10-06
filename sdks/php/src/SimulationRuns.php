<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\SimulationRun;

/**
 * The runs of every simulation, newest first.
 */
final readonly class SimulationRuns
{
    public function __construct(private Client $client)
    {
    }

    /**
     * One run, with the conversations it had.
     */
    public function get(string $id): SimulationRun
    {
        return SimulationRun::fromArray(Json::asObject($this->client->get('/v1/agents/simulation-runs/{id}', ['id' => $id])));
    }

    /**
     * @param ?string $simulationId only this simulation's runs; null is every simulation's
     * @return list<SimulationRun>
     */
    public function list(?string $simulationId = null, ?string $state = null, ?int $limit = null): array
    {
        $listed = $this->client->get('/v1/agents/simulation-runs', query: ['simulation_id' => $simulationId, 'state' => $state, 'limit' => $limit]);
        return array_map(SimulationRun::fromArray(...), Json::objects(['rows' => $listed], 'rows'));
    }

    /**
     * Stops a run in progress, ending the conversations still in flight.
     */
    public function cancel(string $id): SimulationRun
    {
        return SimulationRun::fromArray(Json::asObject($this->client->post('/v1/agents/simulation-runs/{id}/cancel', ['id' => $id])));
    }
}
