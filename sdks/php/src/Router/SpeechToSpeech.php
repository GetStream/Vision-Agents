<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Router;

use GetStream\VisionAgents\Generated\StsOptions;
use GetStream\VisionAgents\Router;
use GetStream\VisionAgents\Worker\Realtime;

/**
 * A conversation with one native audio model. There is no recording: a conversation is live or
 * it is not one.
 */
final readonly class SpeechToSpeech
{
    public function __construct(private Router $router)
    {
    }

    /**
     * Needs amphp/websocket-client.
     */
    public function realtime(?StsOptions $options = null): Realtime
    {
        return Realtime::open($this->router->client, 'sts', Jobs::start($this->router, 'sts', $options?->toArray()));
    }
}
