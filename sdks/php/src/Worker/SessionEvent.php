<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use GetStream\VisionAgents\Json;

/**
 * One thing the conversation did: `heard`, `responded`, `spoke`, `tool_ran`, `left`, and the
 * rest the socket reports. The fields are the frame's own, since each kind carries different
 * ones and the router adds kinds faster than a class per kind would keep up.
 */
final readonly class SessionEvent
{
    /**
     * @param array<string, mixed> $data the whole frame, `type` included
     */
    public function __construct(public string $kind, public array $data)
    {
    }

    /**
     * @param array<string, mixed> $frame
     */
    public static function fromFrame(array $frame): self
    {
        return new self(Json::string($frame, 'type'), $frame);
    }

    /**
     * What was said, for the kinds that carry words.
     */
    public function text(): string
    {
        return Json::string($this->data, 'text');
    }
}
