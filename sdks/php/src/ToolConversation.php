<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

/**
 * The conversation a tool call happens in.
 */
final readonly class ToolConversation
{
    /**
     * @param string $cid the channel, as `type:id`
     * @param array<string, mixed> $custom the custom data on the channel
     */
    public function __construct(public string $cid = '', public array $custom = [])
    {
    }
}
