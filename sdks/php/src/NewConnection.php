<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

/**
 * A connection to create with `Connections::create`, pending until an account is connected.
 */
final readonly class NewConnection
{
    /**
     * @param string $connectorId a built-in, such as linear, or one of the app's own
     * @param ?string $label tells the owner's connections to the same connector apart
     * @param ?string $userId the user the connection is for; none is the app's own
     */
    public function __construct(
        public string $connectorId,
        public ?string $label = null,
        public ?string $userId = null,
    ) {
    }
}
