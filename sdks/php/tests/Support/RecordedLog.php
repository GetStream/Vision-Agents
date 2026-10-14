<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Support;

use Psr\Log\AbstractLogger;
use Stringable;

/**
 * A logger that keeps what it was told, level and message.
 */
final class RecordedLog extends AbstractLogger
{
    /** @var list<array{string, string}> */
    public array $lines = [];

    /**
     * @param array<mixed> $context
     */
    public function log($level, string|Stringable $message, array $context = []): void
    {
        $this->lines[] = [is_string($level) ? $level : '', (string) $message];
    }
}
