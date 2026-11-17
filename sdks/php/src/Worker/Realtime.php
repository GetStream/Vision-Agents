<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\RealtimeException;
use GetStream\VisionAgents\Json;
use Generator;

/**
 * One modality routed over a socket, for a pipeline running in this process: PCM in and
 * transcripts out for speech-to-text, `speak` in and audio out for a voice, `respond` in and
 * deltas out for a model. Failover and billing are the same as inside a session.
 *
 *     $stt = $router->stt->realtime();
 *     $stt->sendAudio($pcm16);
 *     foreach ($stt->frames() as $frame) { ... }
 *
 * Text frames arrive as decoded objects and audio as the raw bytes, header included; the
 * router ends every stream with a `closed` frame, after which the iterator stops.
 */
final class Realtime
{
    private function __construct(private readonly Socket $socket, public readonly string $modality)
    {
    }

    /**
     * @param 'stt'|'tts'|'llm'|'sts' $modality
     * @param array<string, mixed> $start the `start` frame's fields: `config_id`, the modality's
     *     own options, `agent_id`, `call_id`, `tags`
     */
    public static function open(Client $client, string $modality, array $start): self
    {
        $socket = Socket::open($client, "/v1/{$modality}/stream");
        $socket->send(['type' => 'start', ...$start]);
        return new self($socket, $modality);
    }

    /**
     * PCM16 at the sample rate the start frame named, 16 kHz mono by default.
     */
    public function sendAudio(string $pcm): void
    {
        $this->socket->sendBinary($pcm);
    }

    /**
     * Text for a voice to speak.
     */
    public function speak(string $text): void
    {
        $this->socket->send(['type' => 'speak', 'text' => $text]);
    }

    /**
     * Asks a model for one response.
     *
     * @param array<string, mixed> $request `messages`, `tools`, and anything from LlmOptions for
     *     this response alone
     */
    public function respond(string $id, array $request): void
    {
        $this->socket->send(['type' => 'respond', 'id' => $id, ...$request]);
    }

    /**
     * Any other frame the modality takes, such as `interrupt` or `tool_result`.
     *
     * @param array<string, mixed> $frame
     */
    public function send(array $frame): void
    {
        $this->socket->send($frame);
    }

    /**
     * What the router sends until the stream is closed. An `error` frame is raised, since the
     * stream has failed and whatever follows it is only the close.
     *
     * @return Generator<int, array<string, mixed>|string>
     */
    public function frames(): Generator
    {
        while (($frame = $this->socket->receive()) !== null) {
            if (is_array($frame)) {
                $type = Json::string($frame, 'type');
                if ($type === 'closed') {
                    return;
                }
                if ($type === 'error') {
                    $said = Json::string($frame, 'error');
                    $said = $said === '' ? Json::string($frame, 'message') : $said;
                    throw new RealtimeException($this->modality, $said, $frame);
                }
            }
            yield $frame;
        }
    }

    public function close(): void
    {
        $this->socket->close();
    }
}
