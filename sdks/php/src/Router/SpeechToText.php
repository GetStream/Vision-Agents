<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Router;

use GetStream\VisionAgents\Generated\RecordingSource;
use GetStream\VisionAgents\Generated\RecordingStatus;
use GetStream\VisionAgents\Generated\SttOptions;
use GetStream\VisionAgents\Generated\Transcription;
use GetStream\VisionAgents\Generated\TranscriptionRequest;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Router;
use GetStream\VisionAgents\Worker\Realtime;

/**
 * Transcription, live or from a recording.
 */
final readonly class SpeechToText
{
    public function __construct(private Router $router)
    {
    }

    /**
     * Transcribes live audio sent over a socket. Needs amphp/websocket-client.
     */
    public function realtime(?SttOptions $options = null): Realtime
    {
        return Realtime::open($this->router->client, 'stt', Jobs::start($this->router, 'stt', $options?->toArray()));
    }

    /**
     * Transcribes a whole recording, by the batch half of a vendor, which is cheaper and more
     * accurate than its streaming model. It waits for the job unless a callback is given.
     *
     * @param RecordingSource|string $source a URL for the provider to fetch, or a source with
     *     the audio inline
     *
     * @throws \GetStream\VisionAgents\Exception\RecordingFailedException when the job failed
     */
    public function recording(RecordingSource|string $source, ?SttOptions $options = null, string $callback = ''): Transcription
    {
        $body = new TranscriptionRequest(
            source: is_string($source) ? new RecordingSource(url: $source) : $source,
            configId: $this->router->configId(),
            options: $options,
            callback: $callback === '' ? null : $callback,
            tags: $this->router->labels(),
        );
        $client = $this->router->client;
        $job = Transcription::fromArray(Json::asObject($client->post('/v1/stt/recordings', body: $body->toArray())));
        if ($callback !== '') {
            return $job;
        }
        while ($job->status === RecordingStatus::Queued || $job->status === RecordingStatus::Running) {
            Jobs::pause();
            $job = Transcription::fromArray(Json::asObject($client->get('/v1/stt/recordings/{id}', ['id' => $job->id])));
        }
        Jobs::settled($job->id, $job->status, $job->error);
        return $job;
    }
}
