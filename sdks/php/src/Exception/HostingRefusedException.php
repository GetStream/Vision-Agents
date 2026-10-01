<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

use RuntimeException;

/**
 * The router refused the tools a worker offered to host. Reconnecting would only be told again.
 */
final class HostingRefusedException extends RuntimeException implements VisionAgentsException
{
    public function __construct(public readonly string $agentId, public readonly string $reason)
    {
        parent::__construct("the router refused to host tools for agent {$agentId}: {$reason}");
    }
}
