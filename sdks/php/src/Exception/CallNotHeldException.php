<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

use RuntimeException;

/**
 * A tool call asked to keep going through interruptions too late: the person had already
 * talked over the agent, so the router cancels the call instead.
 */
final class CallNotHeldException extends RuntimeException implements VisionAgentsException
{
    public function __construct(public readonly string $toolCallId)
    {
        parent::__construct("tool call {$toolCallId} was not held: it was already cancelled");
    }
}
