<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Router;

use GetStream\VisionAgents\Exception\RecordingFailedException;
use GetStream\VisionAgents\Generated\RecordingStatus;
use GetStream\VisionAgents\Pause;
use GetStream\VisionAgents\Router;

/**
 * @internal
 */
final class Jobs
{
    private const float POLL_EVERY = 0.5;

    /**
     * A stream's `start` frame. The options go under the modality's own key: the router reads
     * only the block belonging to the socket's modality.
     *
     * @param 'stt'|'tts'|'llm'|'sts' $modality
     * @param array<string, mixed>|null $options
     * @return array<string, mixed>
     */
    public static function start(Router $router, string $modality, ?array $options): array
    {
        return array_filter(
            ['config_id' => $router->configId(), 'tags' => $router->labels(), $modality => $options],
            static fn (mixed $value): bool => $value !== null && $value !== [],
        );
    }

    public static function pause(): void
    {
        Pause::for(self::POLL_EVERY);
    }

    public static function settled(string $job, RecordingStatus|string $status, ?string $error): void
    {
        if ($status === RecordingStatus::Failed) {
            throw new RecordingFailedException($job, $error ?? '');
        }
    }
}
