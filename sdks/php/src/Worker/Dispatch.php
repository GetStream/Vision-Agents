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
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Inbound\InboundCall;
use GetStream\VisionAgents\Inbound\InboundMessage;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
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

    /** @var array<int, Future<mixed>> */
    private array $running = [];
    private int $nextJob = 0;
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
     * does not stop the next from being answered; what it throws is reported to the router as a
     * call nobody took.
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
     * Registers what to do with a message written to an agent that is not running.
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
     * Waits for calls and messages until the router closes the socket or `stop()` is called.
     *
     * Work still being handled is waited for on the way out, because dropping a call would hang
     * up on whoever is talking. Where pcntl is loaded, SIGINT and SIGTERM stop it the same way.
     */
    public function run(): void
    {
        if ($this->onCall === null && $this->onMessage === null) {
            throw new ConfigurationException('register a handler with waitForCall or waitForMessage before running');
        }
        $socket = Socket::open($this->client, '/v1/dispatch', ['capacity' => $this->capacity]);
        $this->socket = $socket;

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
                    'call' => $this->answer(InboundCall::fromFrame($frame)),
                    'message' => $this->write(InboundMessage::fromFrame($frame)),
                    'ready' => $this->workerId = Json::string($frame, 'worker_id'),
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

    private function answer(InboundCall $call): void
    {
        $handler = $this->onCall;
        if ($handler === null) {
            return;
        }
        $this->track(function () use ($handler, $call): void {
            try {
                $handler($call);
            } catch (Throwable $failed) {
                // The router is told, so a call nobody answered shows up there rather than only
                // in this process's log.
                $this->tell(['type' => 'rejected', 'call_id' => $call->callId, 'reason' => $failed->getMessage()]);
                return;
            }
            $this->tell(['type' => 'accepted', 'call_id' => $call->callId]);
        });
    }

    /**
     * Nothing is reported back for a message: accepting and rejecting are about a caller
     * waiting on a line, and there is no line here.
     */
    private function write(InboundMessage $message): void
    {
        $handler = $this->onMessage;
        if ($handler === null) {
            return;
        }
        $this->track(static fn () => $handler($message));
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
                // Whether it threw was already decided by the handler's wrapper; one failed
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
