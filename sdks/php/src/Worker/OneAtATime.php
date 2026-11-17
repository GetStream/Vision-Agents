<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use Amp\Sync\LocalMutex;
use Closure;

/**
 * Runs what it is handed one at a time, across the requests each makes. Only Amp's client
 * suspends a fiber mid-request, and Amp brings amphp/sync with it; without it each already runs
 * alone.
 *
 * @internal
 */
final class OneAtATime
{
    private ?LocalMutex $mutex = null;

    /**
     * @template T
     * @param Closure(): T $run
     * @return T
     */
    public function run(Closure $run): mixed
    {
        $lock = class_exists(LocalMutex::class) ? ($this->mutex ??= new LocalMutex())->acquire() : null;
        try {
            return $run();
        } finally {
            $lock?->release();
        }
    }
}
