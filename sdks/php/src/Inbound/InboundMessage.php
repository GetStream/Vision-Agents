<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Inbound;

use DateTimeImmutable;
use GetStream\VisionAgents\Json;

/**
 * A message written to an agent that is not running, or to a running session whose agent
 * leaves text to dispatch.
 *
 * Otherwise one written to an agent that is already running never arrives here: the router
 * answers it from that session, because that agent is the one that knows what has been said.
 */
final readonly class InboundMessage
{
    /**
     * @param string $channelId the channel it was written in, which is the conversation to answer in
     * @param string $agentId who to answer as, which names the channel replies are written into
     * @param string $configId the stored agent config the router matched, if it matched one
     * @param string $sessionId the running session it was written to; nothing has answered it yet
     * @param string $requestId the request it was sent as, passed back when answering
     * @param array<string, string> $custom
     */
    public function __construct(
        public string $channelId,
        public string $text,
        public string $channelType = 'agent',
        public string $agentId = '',
        public string $configId = '',
        public string $messageId = '',
        public string $userId = '',
        public string $userName = '',
        public array $custom = [],
        public ?DateTimeImmutable $at = null,
        public string $sessionId = '',
        public string $requestId = '',
    ) {
    }

    /**
     * @param array<string, mixed> $frame
     */
    public static function fromFrame(array $frame): self
    {
        $type = Json::string($frame, 'channel_type');
        return new self(
            channelId: Json::string($frame, 'channel_id'),
            text: Json::string($frame, 'text'),
            channelType: $type === '' ? 'agent' : $type,
            agentId: Json::string($frame, 'agent_id'),
            configId: Json::string($frame, 'config_id'),
            messageId: Json::string($frame, 'message_id'),
            userId: Json::string($frame, 'user_id'),
            userName: Json::string($frame, 'user_name'),
            custom: Json::stringMap($frame, 'custom'),
            at: Json::parseDate(Json::string($frame, 'at')),
            sessionId: Json::string($frame, 'session_id'),
            requestId: Json::string($frame, 'request_id'),
        );
    }

    /**
     * The conversation as the backend names it, `type:id`.
     */
    public function conversationId(): string
    {
        return "{$this->channelType}:{$this->channelId}";
    }
}
