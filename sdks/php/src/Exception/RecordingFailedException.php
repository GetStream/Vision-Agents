<?php

declare(strict_types=1);

namespace GetStream\StreamAgents\Exception;

use RuntimeException;

/**
 * A transcription or speech job the router accepted and then could not finish.
 */
final class RecordingFailedException extends RuntimeException implements StreamAgentsException
{
    public function __construct(public readonly string $id, public readonly string $said)
    {
        parent::__construct($said === '' ? "recording {$id} failed" : "recording {$id} failed: {$said}");
    }
}
