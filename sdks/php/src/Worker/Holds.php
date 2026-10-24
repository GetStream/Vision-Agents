<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use Amp\DeferredFuture;
use Amp\Future;

/**
 * The calls that asked to be held, waiting on the router's `tool_held` or `tool_cancel`, and
 * the ones it held. Keyed by whatever the socket's `tool_cancel` names a call by.
 *
 * @internal
 */
final class Holds
{
    /** @var array<string, DeferredFuture<bool>> */
    private array $asking = [];

    /** @var array<string, true> */
    private array $held = [];

    /**
     * @return Future<bool>
     */
    public function ask(string $key): Future
    {
        /** @var DeferredFuture<bool> $answer */
        $answer = new DeferredFuture();
        $this->asking[$key] = $answer;
        return $answer->getFuture();
    }

    /**
     * Hands the router's answer to the call waiting on it, if one is.
     */
    public function answer(string $key, bool $held): void
    {
        $asking = $this->asking[$key] ?? null;
        if ($asking === null) {
            return;
        }
        unset($this->asking[$key]);
        if ($held) {
            $this->held[$key] = true;
        }
        $asking->complete($held);
    }

    /**
     * Whether the router keeps this call going when the person talks over the agent.
     */
    public function held(string $key): bool
    {
        return isset($this->held[$key]);
    }

    public function forget(string $key): void
    {
        unset($this->asking[$key], $this->held[$key]);
    }

    /**
     * Refuses every hold still waiting, for a socket that ended: nobody is left to answer.
     */
    public function release(): void
    {
        foreach (array_keys($this->asking) as $key) {
            $this->answer($key, false);
        }
        $this->held = [];
    }
}
