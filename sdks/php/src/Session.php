<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\ForkSessionRequest;
use GetStream\VisionAgents\Generated\SayRequest;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Worker\Watch;

/**
 * One conversation, held in the acceleration backend.
 *
 * Nothing here does inference or touches media. The backend joins the call, hears the caller,
 * answers and speaks. Everything a caller does to the conversation goes over HTTP, so a session
 * is usable from an ordinary request handler; watching it and answering tool calls needs a
 * socket held open, which is `watch()` and a long-running process.
 */
final class Session
{
    public readonly Responses $responses;
    private bool $closed = false;
    private bool $voice;

    /**
     * @param ?Agent $agent the agent that opened it, or null for one found by id
     */
    public function __construct(
        public readonly Client $client,
        public readonly SessionRow $created,
        public readonly ?Agent $agent = null,
    ) {
        $this->responses = new Responses($client, $created->id);
        $this->voice = $created->callId !== '';
    }

    /**
     * The backend's id for the session.
     */
    public function id(): string
    {
        return $this->created->id;
    }

    /**
     * The Stream Chat channel replies are written into, as `type:id`, for a session that keeps
     * one.
     */
    public function conversationId(): string
    {
        return $this->created->conversationId ?? '';
    }

    /**
     * The Stream call the conversation is on, empty for one held in writing.
     */
    public function callId(): string
    {
        return $this->created->callId;
    }

    /**
     * The Stream call the conversation is on while voice is started, named after the session.
     */
    public function call(): Call
    {
        return new Call($this->id());
    }

    /**
     * Has the agent join the session's call and carry the conversation on there. Starting
     * voice that is already on does nothing.
     */
    public function startVoice(): SessionRow
    {
        $started = SessionRow::fromArray(Json::asObject($this->client->post('/v1/agents/sessions/{id}/voice', ['id' => $this->id()])));
        $this->voice = true;
        return $started;
    }

    /**
     * Takes the agent off the call and carries the conversation on in writing. Stopping voice
     * that is off does nothing.
     */
    public function stopVoice(): SessionRow
    {
        $stopped = SessionRow::fromArray(Json::asObject($this->client->delete('/v1/agents/sessions/{id}/voice', ['id' => $this->id()])));
        $this->voice = false;
        return $stopped;
    }

    /**
     * Whether the agent is on the call, as this process last saw it.
     */
    public function voiceStarted(): bool
    {
        return $this->voice;
    }

    /**
     * Speaks text without going through the model, for when you already know what to say.
     */
    public function say(string $text): void
    {
        $this->client->post('/v1/agents/sessions/{id}/say', ['id' => $this->id()], body: (new SayRequest($text))->toArray());
    }

    /**
     * Abandons the reply being spoken.
     */
    public function interrupt(): void
    {
        $this->client->post('/v1/agents/sessions/{id}/interrupt', ['id' => $this->id()]);
    }

    /**
     * Continues this conversation as a new one.
     *
     * What a fork is for is asking the same question differently: from here on with a harder
     * model, of a different agent, or from an earlier answer (`responseId`). The parent is
     * untouched and keeps its own transcript. An incognito parent cannot be forked.
     */
    public function fork(?ForkSessionRequest $options = null): self
    {
        $forked = $this->client->post('/v1/agents/sessions/{id}/fork', ['id' => $this->id()], body: ($options ?? new ForkSessionRequest())->toArray());
        return new self($this->client, SessionRow::fromArray(Json::asObject($forked)), $this->agent);
    }

    /**
     * Changes this session only: its title, description, custom labels, models or voice. The config it started from is untouched, and a target that does not route is
     * refused before anything changes. See `Sessions::update`.
     *
     * @param array<string, mixed>|null $custom
     */
    public function update(
        ?string $title = null,
        ?string $description = null,
        ?array $custom = null,
        ?string $llm = null,
        ?string $stt = null,
        ?string $tts = null,
        ?string $sts = null,
        ?string $voice = null,
        ?string $thinking = null,
        ?float $temperature = null,
        ?int $maxOutputTokens = null,
        ?string $verbosity = null,
    ): SessionRow {
        return $this->sessions()->update($this->id(), $title, $description, $custom, $llm, $stt, $tts, $sts, $voice, $thinking, $temperature, $maxOutputTokens, $verbosity);
    }

    /**
     * Deletes this conversation, its turns and what it remembered. See `Sessions::delete`.
     */
    public function delete(): void
    {
        $this->sessions()->delete($this->id());
        $this->closed = true;
    }

    /**
     * Deletes what this conversation remembered. See `Sessions::deleteMemories`.
     */
    public function deleteMemories(): void
    {
        $this->sessions()->deleteMemories($this->id());
    }

    /**
     * The session as the router has it now.
     */
    public function refresh(): SessionRow
    {
        return SessionRow::fromArray(Json::asObject($this->client->get('/v1/agents/sessions/{id}', ['id' => $this->id()])));
    }

    /**
     * Watches the conversation over its events socket and answers the model's tool calls.
     *
     * Needs amphp/websocket-client, and a process that stays up for the conversation: a web
     * request that returns closes the socket with it.
     */
    public function watch(bool $interim = false, bool $decisions = true): Watch
    {
        if (!interface_exists(\Amp\Websocket\Client\WebsocketConnection::class)) {
            throw new ConfigurationException('watching a session needs a socket; composer require amphp/websocket-client');
        }
        return Watch::open($this, $this->agent?->tools, $interim, $decisions);
    }

    /**
     * Stops the conversation: the agent leaves, and what it recorded and remembered is kept;
     * `delete` takes it away. Safe to call after it has already ended.
     */
    public function close(): void
    {
        if ($this->closed) {
            return;
        }
        $this->closed = true;
        try {
            $this->client->post('/v1/agents/sessions/{id}/stop', ['id' => $this->id()]);
        } catch (RouterException $failed) {
            if ($failed->status !== 404) {
                throw $failed;
            }
        }
    }

    public function closed(): bool
    {
        return $this->closed;
    }

    private function sessions(): Sessions
    {
        return new Sessions($this->client, $this->agent->name ?? '');
    }
}
