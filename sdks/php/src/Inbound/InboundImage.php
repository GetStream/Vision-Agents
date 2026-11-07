<?php

declare(strict_types=1);

namespace GetStream\StreamAgents\Inbound;

/**
 * A photo sent with an inbound message, as a URL to fetch.
 */
final readonly class InboundImage
{
    /**
     * @param string $mimeType empty when the router does not know it
     */
    public function __construct(public string $url, public string $mimeType = '')
    {
    }
}
