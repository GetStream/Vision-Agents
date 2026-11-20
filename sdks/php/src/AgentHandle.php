<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\AgentConfig;
use GetStream\VisionAgents\Generated\AgentConfigPatch;

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

    /**
     * Changes some of how the agent is configured and returns the config as it now is. A
     * field left out of the patch keeps what is stored, so setting a guardrail leaves the
     * instructions, skills and models alone. Server side only.
     */
    public function updateConfig(AgentConfigPatch $patch): AgentConfig
    {
        $config = $this->config();
        if ($config === null) {
            throw new ConfigurationException("there is no agent called {$this->name} to update");
        }
        return AgentConfig::fromArray(Json::asObject($this->client->patch('/v1/agents/configs/{id}', ['id' => $config->id], $patch->toArray())));
    }
}
