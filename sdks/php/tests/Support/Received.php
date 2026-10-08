<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Support;

use GetStream\VisionAgents\Json;

/**
 * One request the local router received.
 */
final readonly class Received
{
    /**
     * @param array<string, string> $headers lower-cased names
     */
    public function __construct(
        public string $method,
        public string $path,
        public string $query,
        public array $headers,
        public string $body,
    ) {
    }

    public static function fromLine(string $line): self
    {
        $data = Json::asObject(Json::decode($line));
        return new self(
            Json::string($data, 'method'),
            Json::string($data, 'path'),
            Json::string($data, 'query'),
            Json::stringMap($data, 'headers'),
            Json::string($data, 'body'),
        );
    }

    /**
     * @return array<string, mixed>
     */
    public function json(): array
    {
        return Json::asObject(Json::decode($this->body));
    }

    /**
     * @return array<string, string>
     */
    public function params(): array
    {
        parse_str($this->query, $params);
        /** @var array<string, string> $params */
        return $params;
    }
}
