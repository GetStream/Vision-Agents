<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

/**
 * Who a tool call answers, as the router reports them.
 */
final readonly class ToolUser
{
    /**
     * @param array<string, mixed> $custom the custom data on their user
     */
    public function __construct(public string $id = '', public string $name = '', public array $custom = [])
    {
    }
}
