<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

/**
 * What `Connections::authorize` asks when it starts a consent.
 */
final readonly class AuthorizeOptions
{
    /**
     * @param ?string $returnUrl the app's page the browser goes back to when the consent ends,
     *     with connection_id and status added to its query; none goes back to the dashboard
     * @param ?string $userId the user the connection belongs to; none is the app's own
     */
    public function __construct(
        public ?string $returnUrl = null,
        public ?string $userId = null,
    ) {
    }
}
