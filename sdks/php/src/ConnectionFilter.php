<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

/**
 * Which connections `Connections::list` reads.
 */
final readonly class ConnectionFilter
{
    /**
     * @param ?string $userId lists that user's connections; none lists the app's own
     * @param ?string $connectorId keeps one connector's
     */
    public function __construct(
        public ?string $userId = null,
        public ?string $connectorId = null,
    ) {
    }
}
