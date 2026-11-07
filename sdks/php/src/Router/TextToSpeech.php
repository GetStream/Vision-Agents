<?php

declare(strict_types=1);

namespace GetStream\StreamAgents\Router;

use GetStream\StreamAgents\Generated\RecordingStatus;
use GetStream\StreamAgents\Generated\Speech;
use GetStream\StreamAgents\Generated\SpeechRequest;
use GetStream\StreamAgents\Generated\TtsOptions;
use GetStream\StreamAgents\Json;
use GetStream\StreamAgents\Router;
use GetStream\StreamAgents\Worker\Realtime;

/**
 * A voice, live or recorded.
 */
final readonly class TextToSpeech
{
    public function __construct(private Router $router)
    {
    }

    /**
     * Speaks text sent over a socket as it arrives. Needs amphp/websocket-client.
     */
    public function realtime(?TtsOptions $options = null): Realtime
    {
        return Realtime::open($this->router->client, 'tts', Jobs::start($this->router, 'tts', $options?->toArray()));
    }

    /**
     * Speaks a whole text into one file. Nothing is listening while it is made, so this asks
     * for the file rather than the stream. It waits for the job unless a callback is given.
     *
     * @throws \GetStream\StreamAgents\Exception\RecordingFailedException when the job failed
     */
    public function recording(string $text, ?TtsOptions $options = null, string $callback = ''): Speech
    {
        $body = new SpeechRequest(
            text: $text,
            configId: $this->router->configId(),
            options: $options,
            callback: $callback === '' ? null : $callback,
            tags: $this->router->labels(),
        );
        $client = $this->router->client;
        $job = Speech::fromArray(Json::asObject($client->post('/v1/tts/recordings', body: $body->toArray())));
        if ($callback !== '') {
            return $job;
        }
        while ($job->status === RecordingStatus::Queued || $job->status === RecordingStatus::Running) {
            Jobs::pause();
            $job = Speech::fromArray(Json::asObject($client->get('/v1/tts/recordings/{id}', ['id' => $job->id])));
        }
        Jobs::settled($job->id, $job->status, $job->error);
        return $job;
    }
}
