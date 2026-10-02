<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use Amp\DeferredFuture;
use Amp\Future;
use Amp\Sync\LocalKeyedMutex;
use Amp\TimeoutCancellation;
use Amp\CancelledException;
use Closure;
use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\AgentResponse;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\HostingRefusedException;
use GetStream\VisionAgents\Generated\SessionTool;
use GetStream\VisionAgents\Inbound\InboundCall;
use GetStream\VisionAgents\Inbound\InboundMessage;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Responses;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Tools;
use Revolt\EventLoop;
use Throwable;

use function Amp\async;

/**
 * A worker the router hands inbound calls and messages to.
 *
 * The router cannot reach into this process, so the worker connects out and waits. Run it as a
 * long-lived CLI process, not inside a web request: it holds a socket for as long as it is
 * willing to take calls.
 *
 *     $dispatch = new Dispatch(capacity: 4);
 *     $dispatch->waitForCall(function (InboundCall $call): void {
 *         (new Agent(config: 'support'))->answer($call)->watch()->wait();
 *     });
 *     $dispatch->run();
 *
 * Server side only: a worker is offered other people's callers.
 */
final class Dispatch
{
    public const int DEFAULT_CAPACITY = 4;
    private const float PING_TIMEOUT = 5.0;

    public readonly Client $client;
    /** What the router calls this connection, for matching a log line here against one there. */
    public string $workerId = '';
    /** @var list<string> the agent ids the router has said it sends this worker's tools for */
    public array $hosting = [];

    /** @var list<array{agentId: string, tools: Tools, timeoutMs: int}> */
    private array $hosted = [];
    /** @var array<int, Future<mixed>> */
    private array $running = [];
    private int $nextJob = 0;
    /** The calls and messages being handled, which the router counts against capacity; hosted tool calls are not. */
    private int $handling = 0;
    /** @var array<string, Session> which session is answering which channel */
    private array $answering = [];
    private readonly LocalKeyedMutex $channels;
    /** @var ?Closure(InboundCall): void */
    private ?Closure $onCall = null;
    /** @var ?Closure(InboundMessage): void */
    private ?Closure $onMessage = null;
    private ?Socket $socket = null;
    /** @var ?DeferredFuture<float> */
    private ?DeferredFuture $pong = null;
    /** The last round trip measured, from this side, because this is the side audio crosses. */
    private float $latencyMs = 0.0;

    /**
     * @param int $capacity how many calls this worker takes at once
     * @param float $reportEvery seconds between load reports
     */
    public function __construct(
        public readonly int $capacity = self::DEFAULT_CAPACITY,
        private readonly float $reportEvery = 15.0,
        ?Client $client = null,
    ) {
        if ($capacity < 1) {
            throw new ConfigurationException('a worker that can hold no calls cannot answer any');
        }
        $this->client = $client ?? new Client();
        $this->channels = new LocalKeyedMutex();
    }

    /**
     * Registers what to do with an arriving call. It runs in its own fiber, so one long call
     * does not stop the next from being answered; what it throws is reported to the router with
     * the `done` that frees the room it took.
     *
     * @param callable(InboundCall): mixed $handler
     */
    public function waitForCall(callable $handler): self
    {
        $this->onCall = static function (InboundCall $call) use ($handler): void {
            $handler($call);
        };
        return $this;
    }

    /**
     * Registers what to do with a message written to an agent that is not running, or to a
     * running session whose agent leaves text to dispatch (see `answer`).
     *
     * @param callable(InboundMessage): mixed $handler
     */
    public function waitForMessage(callable $handler): self
    {
        $this->onMessage = static function (InboundMessage $message) use ($handler): void {
            $handler($message);
        };
        return $this;
    }

    /**
     * Runs these tools for every session opened under an agent id, whoever opened it.
     *
     * A session's own tools run in the process that opened it, which is no use to a conversation
     * opened from a browser. The router offers these to each session naming the agent and sends
     * every call here. Call before `run()`.
     *
     * @param int $timeoutMs how long the router gives one call; 0 takes its default
     */
    public function host(string $agentId, Tools $tools, int $timeoutMs = 0): self
    {
        $this->hosted[] = ['agentId' => $agentId, 'tools' => $tools, 'timeoutMs' => $timeoutMs];
        return $this;
    }

    /**
     * The session answering on this message's channel, started if none is.
     *
     * A channel is one conversation, so the second message on it goes to the session that
     * answered the first, which knows what has been said. Sessions are kept until this worker
     * stops waiting, so a conversation is not restarted between messages.
     *
     * @param callable(): Agent $createAgent builds the agent for a channel nothing is answering
     */
    public function getOrCreateAgent(InboundMessage $message, callable $createAgent): Session
    {
        if ($message->sessionId !== '') {
            throw new ConfigurationException('a session is already holding this conversation; answer it there with answer()');
        }
        // Two messages on one channel arriving together would otherwise start two agents.
        $lock = $this->channels->acquire($message->channelId);
        try {
            $open = $this->answering[$message->channelId] ?? null;
            if ($open !== null && !$open->closed()) {
                return $open;
            }
            $session = $createAgent()->reply($message);
            $this->answering[$message->channelId] = $session;
            return $session;
        } finally {
            $lock->release();
        }
    }

    /**
     * Has the model answer a message written to a running session whose agent leaves text to
     * dispatch.
     *
     * The response is created with this worker's own credential, acting for whoever wrote the
     * message, so it goes to the model rather than back to a worker. It carries the message's
     * command, so the answer lands on it.
     */
    public function answer(InboundMessage $message): AgentResponse
    {
        if ($message->sessionId === '') {
            throw new ConfigurationException('no session is holding this message; open one with getOrCreateAgent');
        }
        $client = $this->client->withBackend($this->client->backend->onBehalfOf($message->userId));
        return (new Responses($client, $message->sessionId))->create($message->text, commandId: $message->commandId);
    }

    /**
     * Waits for calls, messages and hosted tool calls until the router closes the socket or
     * `stop()` is called.
     *
     * Work still being handled is waited for on the way out, because dropping a call would hang
     * up on whoever is talking. Where pcntl is loaded, SIGINT and SIGTERM stop it the same way.
     *
     * @throws HostingRefusedException when the router refuses the hosted tools
     */
    public function run(): void
    {
        if ($this->onCall === null && $this->onMessage === null && $this->hosted === []) {
            throw new ConfigurationException('register a handler with waitForCall or waitForMessage, or host tools, before running');
        }
        $socket = Socket::open($this->client, '/v1/dispatch', $this->waiting());
        $this->socket = $socket;
        $this->hosting = [];

        $watchers = [EventLoop::repeat($this->reportEvery, fn () => $this->report())];
        if (\extension_loaded('pcntl')) {
            foreach ([\SIGINT, \SIGTERM] as $signal) {
                $watchers[] = EventLoop::onSignal($signal, fn () => $this->stop());
            }
        }
        try {
            while (($frame = $socket->receive()) !== null) {
                if (is_string($frame)) {
                    continue;
                }
                match (Json::string($frame, 'type')) {
                    'call' => $this->pickUp(Json::string($frame, 'work_id'), InboundCall::fromFrame($frame)),
                    'message' => $this->write(Json::string($frame, 'work_id'), InboundMessage::fromFrame($frame)),
                    'ready' => $this->ready(Json::string($frame, 'worker_id')),
                    'tool_call' => $this->runHosted($frame),
                    'hosting' => $this->hosting[] = Json::string($frame, 'agent_id'),
                    // A worker whose tools were refused is one nobody will call.
                    'hosting_refused' => throw new HostingRefusedException(Json::string($frame, 'agent_id'), Json::string($frame, 'reason')),
                    'pong' => $this->pong?->isComplete() === false ? $this->pong->complete(Json::float($frame, 'at')) : null,
                    default => null,
                };
            }
        } finally {
            array_map(EventLoop::cancel(...), $watchers);
            Future\awaitAll($this->running);
            $this->answering = [];
            $socket->close();
            $this->socket = null;
        }
    }

    /**
     * Stops waiting. Work already being handled is still waited for by `run()`.
     */
    public function stop(): void
    {
        $this->socket?->close();
    }

    /**
     * How much work is being handled right now.
     */
    public function active(): int
    {
        return count($this->running);
    }

    /**
     * What this worker says about itself on the handshake: how much it can hold, how much it is
     * already holding, and which kinds of work it answers (said even when it is none).
     *
     * @return array<string, scalar>
     */
    private function waiting(): array
    {
        $handles = [];
        if ($this->onCall !== null) {
            $handles[] = 'call';
        }
        if ($this->onMessage !== null) {
            $handles[] = 'message';
        }
        return ['capacity' => $this->capacity, 'active' => $this->handling, 'handles' => implode(',', $handles)];
    }

    /**
     * Tells the router what this worker hosts, once it is listening.
     */
    private function ready(string $workerId): void
    {
        $this->workerId = $workerId;
        foreach ($this->hosted as $offer) {
            $this->tell([
                'type' => 'host_tools',
                'agent_id' => $offer['agentId'],
                'tools' => array_map(static fn (SessionTool $tool): array => $tool->toArray(), $offer['tools']->declared()),
                'timeout_ms' => $offer['timeoutMs'],
            ]);
        }
    }

    /**
     * Answers one hosted tool call in its own fiber, since the socket it arrived on also
     * delivers the next.
     *
     * @param array<string, mixed> $frame
     */
    private function runHosted(array $frame): void
    {
        $id = Json::string($frame, 'id');
        $name = Json::string($frame, 'name');
        $tools = null;
        foreach ($this->hosted as $offer) {
            foreach ($offer['tools']->declared() as $tool) {
                if ($tool->name === $name) {
                    $tools = $offer['tools'];
                }
            }
        }
        if ($tools === null) {
            $this->tell(['type' => 'tool_result', 'id' => $id, 'error' => "this worker does not run {$name}"]);
            return;
        }
        $arguments = $frame['arguments'] ?? '';
        $this->track(function () use ($tools, $id, $name, $arguments): void {
            $result = ['type' => 'tool_result', 'id' => $id];
            try {
                $result['output'] = $tools->call($name, is_string($arguments) ? $arguments : Json::encode($arguments));
            } catch (Throwable $failed) {
                $result['error'] = $failed->getMessage();
            }
            $this->tell($result);
        });
    }

    private function pickUp(string $workId, InboundCall $call): void
    {
        $handler = $this->onCall;
        if ($handler === null) {
            $this->done($workId, 'this worker answers no calls');
            return;
        }
        $this->handle($workId, static fn () => $handler($call));
    }

    private function write(string $workId, InboundMessage $message): void
    {
        $handler = $this->onMessage;
        if ($handler === null) {
            $this->done($workId, 'this worker answers no messages');
            return;
        }
        $this->handle($workId, static fn () => $handler($message));
    }

    /**
     * Runs one call or message and says `done` when it ends, which is what gives the router this
     * worker's room back. What it threw goes with it, so a failure shows up there rather than
     * only in this process's log.
     *
     * @param Closure(): void $work
     */
    private function handle(string $workId, Closure $work): void
    {
        $this->handling++;
        $this->track(function () use ($workId, $work): void {
            try {
                $work();
                $this->done($workId);
            } catch (Throwable $failed) {
                $this->done($workId, $failed->getMessage());
            } finally {
                $this->handling--;
            }
        });
    }

    /**
     * Said even for work there was no handler for, because the router holds its room until then.
     */
    private function done(string $workId, ?string $error = null): void
    {
        $this->tell(['type' => 'done', 'work_id' => $workId, ...($error === null ? [] : ['error' => $error])]);
    }

    /**
     * @param Closure(): void $work
     */
    private function track(Closure $work): void
    {
        $job = $this->nextJob++;
        $this->running[$job] = async(function () use ($work, $job): void {
            try {
                $work();
            } catch (Throwable) {
                // What it threw was already reported by the work itself; one failed
                // conversation must not take the worker down with it.
            } finally {
                unset($this->running[$job]);
            }
        });
    }

    /**
     * Tells the router how this process is doing. Only what PHP can honestly measure is sent:
     * host CPU and memory are not portable, and an invented figure would be read as real.
     */
    private function report(): void
    {
        $this->measure();
        $this->tell(['type' => 'load', 'active_agents' => $this->active(), 'latency_ms' => $this->latencyMs]);
    }

    private function measure(): void
    {
        /** @var DeferredFuture<float> $pong */
        $pong = new DeferredFuture();
        $this->pong = $pong;
        $this->tell(['type' => 'ping', 'at' => microtime(true)]);
        try {
            $at = $pong->getFuture()->await(new TimeoutCancellation(self::PING_TIMEOUT));
            $this->latencyMs = (microtime(true) - $at) * 1000;
        } catch (CancelledException) {
        } finally {
            $this->pong = null;
        }
    }

    /**
     * A closed socket is not an error here: each of these is something the router would like
     * to know rather than something a call depends on.
     *
     * @param array<string, mixed> $frame
     */
    private function tell(array $frame): void
    {
        if ($this->socket !== null && !$this->socket->closed()) {
            $this->socket->send($frame);
        }
    }
}
