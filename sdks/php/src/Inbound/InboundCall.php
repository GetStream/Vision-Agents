<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Inbound;

use DateTimeImmutable;
use GetStream\VisionAgents\Json;

/**
 * A call the router handed to this worker. The caller is already in the Stream call.
 */
final readonly class InboundCall
{
    /**
     * @param string $calledNumber the number they rang, which the agent acts from and can transfer on
     * @param string $callerNumber the number they rang from, where the vendor passed it on
     * @param array<string, string> $custom whatever was put on the Stream call
     */
    public function __construct(
        public string $callId,
        public string $callType = 'default',
        public string $calledNumber = '',
        public string $callerNumber = '',
        public array $custom = [],
        public ?DateTimeImmutable $at = null,
    ) {
    }

    /**
     * @param array<string, mixed> $frame
     */
    public static function fromFrame(array $frame): self
    {
        $type = Json::string($frame, 'call_type');
        return new self(
            callId: Json::string($frame, 'call_id'),
            callType: $type === '' ? 'default' : $type,
            calledNumber: Json::string($frame, 'called_number'),
            callerNumber: Json::string($frame, 'caller_number'),
            custom: Json::stringMap($frame, 'custom'),
            at: Json::parseDate(Json::string($frame, 'at')),
        );
    }
}
