<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\AgentResponse as ResponseRow;

/**
 * One turn, and a way to read what it was made of.
 *
 * A handle on an answer in progress: `items` reads what has been written down so far, and a
 * session's watch is what sees it arrive.
 */
final readonly class AgentResponse
{
    public Items $items;

    public function __construct(Client $client, public ResponseRow $created)
    {
        $this->items = new Items($client, $created->sessionId, $created->id);
    }

    /**
     * The backend's id for this turn, empty for a session that records nothing. It is not the
     * `turn_id` socket events carry.
     */
    public function id(): string
    {
        return $this->created->id;
    }

    public function status(): string
    {
        return $this->created->status;
    }
}
