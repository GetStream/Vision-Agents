<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\AgentConfig;

/**
 * An agent configured in the backend, addressed by name, and its conversations.
 *
 * The other way round from `Agent`, which spells an agent out in code: this one was decided
 * once, in a config or a synced folder, and is only named here.
 */
final readonly class AgentHandle
{
    public Sessions $sessions;

    public function __construct(private Client $client, public string $name)
    {
        $this->sessions = new Sessions($client, $name);
    }

    /**
     * How the agent is configured, or null for a name nothing is stored under. Server side only.
     */
    public function config(): ?AgentConfig
    {
        return Agent::storedConfig($this->client, $this->name);
    }
}
