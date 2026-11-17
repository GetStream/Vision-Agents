<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use Amp\Sync\LocalMutex;
use Closure;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\AddSessionToolsRequest;
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
     * The functions `addTools` gave this session, by name, each in the `Tools` it came in; they
     * run before the agent's own of the same name. `withdrawn` are the names `removeTool` took,
     * which the agent's own no longer run, and `unsettled` the ones whose adding got no answer,
     * which the router may or may not offer.
     *
     * @var array<string, Tools>
     */
    private array $added = [];
    /** @var array<string, true> */
    private array $withdrawn = [];
    /** @var array<string, true> */
    private array $unsettled = [];
    /** Runs one change of the session's tools at a time, across its requests. */
    private ?LocalMutex $changing = null;

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
     * Changes this session only: its title, description, custom labels, pin, models or voice. The config it started from is untouched, and a target that does not route is
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
        ?bool $pinned = null,
    ): SessionRow {
        return $this->sessions()->update($this->id(), $title, $description, $custom, $llm, $stt, $tts, $sts, $voice, $thinking, $temperature, $maxOutputTokens, $verbosity, $pinned);
    }

    /**
     * Pins this conversation, and returns it as it now is, `pinnedAt` set. See `Sessions::pin`.
     */
    public function pin(): SessionRow
    {
        return $this->sessions()->pin($this->id());
    }

    /**
     * Unpins this conversation, and returns it as it now is. See `Sessions::unpin`.
     */
    public function unpin(): SessionRow
    {
        return $this->sessions()->unpin($this->id());
    }

    /**
     * Deletes this conversation, its turns and what it remembered. See `Sessions::delete`.
     */
    public function delete(): void
    {
        $this->sessions()->delete($this->id());
        $this->closed = true;
        $this->added = $this->withdrawn = $this->unsettled = [];
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
        return Watch::open($this, $interim, $decisions);
    }

    /**
     * Gives the running session more of the caller's tools, for this session alone, and runs
     * their calls here while it is watched.
     *
     * A name the session already has is refused: to change a tool, `removeTool` it first, so no
     * call runs half under each. The router waits for the turn in progress and every tool call
     * in flight, and offers the model the tools from the next turn, so do not call this from
     * inside one of this session's tools: it would wait for itself. Only the kind of caller that
     * opened the session may change its tools.
     *
     * @throws ConfigurationException when `$tools` holds none
     * @throws RouterException when the router refuses them, or when its answer was lost
     *     (`status` 0); their calls are then still run here until the name is changed again
     */
    public function addTools(Tools $tools): void
    {
        $declared = $tools->declared();
        if ($declared === []) {
            throw new ConfigurationException('there are no tools to add');
        }
        $this->oneAtATime(function () use ($tools, $declared): void {
            // A name whose adding got no answer may or may not be the router's: it is withdrawn
            // first, so the router and this process agree on which function it is.
            foreach ($declared as $tool) {
                if (isset($this->unsettled[$tool->name])) {
                    $this->withdraw($tool->name, true);
                }
            }
            // Run from before the router answers: the next turn can call one before the answer
            // is here. A name already run here keeps its function, since the router refuses to
            // add it again.
            $fresh = [];
            foreach ($declared as $tool) {
                if (!isset($this->added[$tool->name]) && (isset($this->withdrawn[$tool->name]) || $this->agent?->tools->has($tool->name) !== true)) {
                    $this->added[$tool->name] = $tools;
                    $fresh[] = $tool->name;
                }
            }
            try {
                $this->client->post('/v1/agents/sessions/{id}/tools', ['id' => $this->id()], body: (new AddSessionToolsRequest($declared))->toArray());
            } catch (RouterException $failed) {
                foreach ($fresh as $name) {
                    if ($failed->status === 0) {
                        // The router may have added it, so its calls are still run here until
                        // a later change of the name settles it.
                        $this->unsettled[$name] = true;
                    } else {
                        unset($this->added[$name]);
                    }
                }
                throw $failed;
            }
        });
    }

    /**
     * Takes one of the session's tools back, whether it was declared when the session opened or
     * added since. The router waits as it does for `addTools`, so once this returns no call of
     * the tool is in flight and none can arrive; do not call it from inside one of this
     * session's tools.
     *
     * @throws RouterException 404 when the session has no tool by that name or is not running
     */
    public function removeTool(string $name): void
    {
        $this->oneAtATime(function () use ($name): void {
            $this->withdraw($name, isset($this->unsettled[$name]));
        });
    }

    /**
     * The functions that run the tool called `$name` in this process: the ones `addTools` gave
     * the session, none once `removeTool` took it, else the agent's; null when none here has it.
     *
     * @internal for the session's watcher
     */
    public function toolsFor(string $name): ?Tools
    {
        $tools = $this->added[$name] ?? (isset($this->withdrawn[$name]) ? null : $this->agent?->tools);
        return $tools?->has($name) === true ? $tools : null;
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
        // What addTools and removeTool changed was this session's alone.
        $this->added = $this->withdrawn = $this->unsettled = [];
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

    /**
     * Takes the tool called `$name` from the session and from what runs here. `$absentIsFine`
     * settles a name whose adding got no answer: the router having no tool by it is as good as
     * withdrawing it.
     */
    private function withdraw(string $name, bool $absentIsFine): void
    {
        try {
            $this->client->delete('/v1/agents/sessions/{id}/tools/{name}', ['id' => $this->id(), 'name' => $name]);
        } catch (RouterException $failed) {
            if (!$absentIsFine || $failed->status !== 404) {
                throw $failed;
            }
        }
        unset($this->added[$name], $this->unsettled[$name]);
        $this->withdrawn[$name] = true;
    }

    /**
     * Runs one change of the session's tools at a time. A handler is put in place before the
     * router is asked and taken back when it refuses, so two changes interleaved across their
     * requests could take back what the other put in place. Only Amp's client suspends a fiber
     * mid-request, and Amp brings amphp/sync with it.
     *
     * @param Closure(): void $change
     */
    private function oneAtATime(Closure $change): void
    {
        $lock = class_exists(LocalMutex::class) ? ($this->changing ??= new LocalMutex())->acquire() : null;
        try {
            $change();
        } finally {
            $lock?->release();
        }
    }
}
