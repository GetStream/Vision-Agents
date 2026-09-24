<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

use RuntimeException;

/**
 * A routed stream sent an `error` frame: the model was refused, every provider failed, or the
 * start frame asked for something the modality cannot do.
 */
final class RealtimeException extends RuntimeException implements VisionAgentsException
{
    /**
     * @param array<string, mixed> $frame the error frame as sent
     */
    public function __construct(public readonly string $modality, string $said, public readonly array $frame = [])
    {
        parent::__construct("the {$modality} stream failed: {$said}");
    }
}
