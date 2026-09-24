<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

/**
 * One Stream call, named the way the backend needs it named.
 */
final readonly class Call
{
    /**
     * @param string $id empty names a new call after a random id
     */
    public function __construct(public string $id = '', public string $type = Edge::DEFAULT_CALL_TYPE)
    {
    }
}
