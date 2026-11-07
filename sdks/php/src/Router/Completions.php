<?php

declare(strict_types=1);

namespace GetStream\StreamAgents\Router;

use GetStream\StreamAgents\Generated\LlmOptions;
use GetStream\StreamAgents\Router;
use GetStream\StreamAgents\Worker\Realtime;

/**
 * The model that answers. There is no recording here: what the socket buys is the answer
 * arriving as it is written.
 */
final readonly class Completions
{
    public function __construct(private Router $router)
    {
    }

    /**
     * Needs amphp/websocket-client.
     */
    public function realtime(?LlmOptions $options = null): Realtime
    {
        return Realtime::open($this->router->client, 'llm', Jobs::start($this->router, 'llm', $options?->toArray()));
    }
}
