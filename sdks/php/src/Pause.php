<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use Revolt\EventLoop;

/**
 * Waiting between polls.
 *
 * Inside a worker a blocking sleep would stall every other call the process is holding, so
 * where Revolt is installed this suspends on its loop instead, which from plain code is the
 * same as sleeping.
 *
 * @internal
 */
final class Pause
{
    public static function for(float $seconds): void
    {
        if (class_exists(EventLoop::class)) {
            $suspension = EventLoop::getSuspension();
            EventLoop::delay($seconds, static fn () => $suspension->resume());
            $suspension->suspend();
            return;
        }
        usleep((int) ($seconds * 1_000_000));
    }
}
