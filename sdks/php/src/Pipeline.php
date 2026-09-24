<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\SessionVideo;

/**
 * Which models hold the conversation.
 *
 * Every target is a provider/model name or a capability shortcut such as `llm-fast`. A field
 * left null is left out of the request, so the stored config or the router decides it; a
 * schema default is never copied in, since that is how a caller silently loses the model
 * their config named.
 */
final readonly class Pipeline
{
    /**
     * @param ?string $sts a speech-to-speech target; naming one means no transcriber or voice is opened
     * @param ?string $greeting said on joining without going through the model
     * @param ?bool $backchannel murmur while a caller is still talking, the way a person does
     * @param list<string>|null $keyterms words the transcriber would otherwise get wrong
     */
    public function __construct(
        public ?string $llm = null,
        public ?string $stt = null,
        public ?string $tts = null,
        public ?string $sts = null,
        public ?string $voice = null,
        public ?string $language = null,
        public ?string $greeting = null,
        public ?bool $backchannel = null,
        public ?int $maxTokens = null,
        public ?int $toolTimeoutMs = null,
        public ?array $keyterms = null,
        public ?SessionVideo $video = null,
    ) {
    }
}
