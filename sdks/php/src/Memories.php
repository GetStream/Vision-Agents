<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;

/**
 * What agents remember about the app's users between conversations.
 */
final readonly class Memories
{
    public function __construct(private Client $client)
    {
    }

    /**
     * Deletes everything remembered about one user: every session's and every agent's,
     * whatever memory filter it was written under. Server side only.
     *
     * @param string $userId the `user_id` of the memory filter the sessions were opened with
     */
    public function truncate(string $userId): void
    {
        if ($userId === '') {
            throw new ConfigurationException('truncating memories needs a user id');
        }
        $this->client->delete('/v1/agents/users/{user_id}/memories', ['user_id' => $userId]);
    }
}
