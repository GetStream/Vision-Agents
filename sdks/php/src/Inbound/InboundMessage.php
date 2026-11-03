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
     * @param list<InboundImage> $images the photos sent with it; empty when none was sent
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
        public array $images = [],
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
            images: self::imagesOf($frame),
        );
    }

    /**
     * The photos a message frame carries. An entry with no string URL is dropped, since there
     * is nothing to fetch, and so is one that is not an object.
     *
     * @param array<string, mixed> $frame
     * @return list<InboundImage>
     */
    private static function imagesOf(array $frame): array
    {
        $sent = $frame['images'] ?? null;
        if (!is_array($sent) || !array_is_list($sent)) {
            return [];
        }
        $images = [];
        foreach ($sent as $described) {
            if (!is_array($described)) {
                continue;
            }
            $url = $described['url'] ?? null;
            $mimeType = $described['mime_type'] ?? null;
            if (is_string($url) && $url !== '') {
                $images[] = new InboundImage($url, is_string($mimeType) ? $mimeType : '');
            }
        }
        return $images;
    }

    /**
     * The conversation as the backend names it, `type:id`.
     */
    public function conversationId(): string
    {
        return "{$this->channelType}:{$this->channelId}";
    }
}
