<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

use RuntimeException;

/**
 * A socket closed while something was still being sent or waited for on it.
 */
final class SocketClosedException extends RuntimeException implements VisionAgentsException
{
    public function __construct(string $message, public readonly int $closeCode = 0)
    {
        parent::__construct($message);
    }
}
