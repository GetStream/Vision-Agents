<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Support;

/**
 * Bodies the router answers with, as little of each as the spec requires.
 */
final class Rows
{
    public const string AT = '2026-09-24T10:00:00.123456789Z';

    /**
     * @param array<string, mixed> $extra
     * @return array<string, mixed>
     */
    public static function session(string $id = 'ses_1', array $extra = []): array
    {
        return ['id' => $id, 'call_id' => '', 'call_type' => '', 'user_id' => 'jean', 'agent_id' => 'jean', 'state' => 'live', 'created_at' => self::AT, ...$extra];
    }

    /**
     * @return array<string, mixed>
     */
    public static function config(string $id, string $name): array
    {
        return ['id' => $id, 'name' => $name, 'mode' => 'voice', 'created_at' => self::AT, 'updated_at' => self::AT];
    }

    /**
     * @return array<string, mixed>
     */
    public static function response(string $id, string $status = 'completed'): array
    {
        return ['id' => $id, 'session_id' => 'ses_1', 'status' => $status, 'created_at' => self::AT];
    }

    /**
     * @return array<string, mixed>
     */
    public static function item(string $responseId, int $ordinal, string $kind = 'message'): array
    {
        return ['response_id' => $responseId, 'ordinal' => $ordinal, 'kind' => $kind, 'at' => self::AT];
    }

    /**
     * @return array<string, mixed>
     */
    public static function page(string $state, int $passages = 0): array
    {
        return ['id' => 'kurl_1', 'namespace' => 'jean', 'url' => 'https://example.com/plans', 'state' => $state, 'passages' => $passages, 'created_at' => self::AT, 'updated_at' => self::AT];
    }

    /**
     * @param array<string, mixed> $extra
     * @return array<string, mixed>
     */
    public static function job(string $status, array $extra = []): array
    {
        return ['id' => 'job_1', 'status' => $status, 'created_at' => self::AT, 'updated_at' => self::AT, ...$extra];
    }
}
