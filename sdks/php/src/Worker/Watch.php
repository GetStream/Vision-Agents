<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use Amp\Future;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\ToolCall;
use GetStream\VisionAgents\Tools;
use Generator;
use IteratorAggregate;
use Throwable;

use function Amp\async;

/**
 * A session's socket: what the conversation does as it does it, and the model's calls to the
 * caller's own tools, answered as they arrive.
 *
 *     foreach ($session->watch() as $event) {
 *         if ($event->kind === 'responded') { echo $event->text(), "\n"; }
 *     }
 *
 * A tool call is answered only while this is being read, since reading is what runs it.
 *
 * @implements IteratorAggregate<int, SessionEvent>
 */
final class Watch implements IteratorAggregate
{
    /** @var array<string, Future<mixed>> tool calls still running, by id */
    private array $running = [];

    /** @var array<string, ToolCall> the calls those are running for, by id */
    private array $calls = [];

    private readonly Holds $holds;

    private function __construct(private readonly Socket $socket, private readonly ?Tools $tools, private readonly string $sessionId = '')
    {
        $this->holds = new Holds();
    }

    public static function open(Session $session, ?Tools $tools, bool $interim = false, bool $decisions = true): self
    {
        $socket = Socket::open($session->client, "/v1/agents/sessions/{$session->id()}/events", [
            'interim' => $interim,
            'decisions' => $decisions,
        ]);
        return new self($socket, $tools, $session->id());
    }

    /**
     * @return Generator<int, SessionEvent>
     */
    public function getIterator(): Generator
    {
        try {
            while (($frame = $this->socket->receive()) !== null) {
                if (is_string($frame)) {
                    continue;
                }
                $kind = Json::string($frame, 'type');
                if ($kind === 'tool_call') {
                    // Every watcher is sent every call and the first answer wins: answering a
                    // tool held elsewhere would beat whoever runs it, with an error. One the
                    // router addresses to another watcher says so with `run` false.
                    if ($this->tools === null || !$this->tools->has(Json::string($frame, 'name')) || ($frame['run'] ?? true) === false) {
                        continue;
                    }
                    $id = Json::string($frame, 'id');
                    $toolCallContext = ToolCall::fromFrame($frame, $this->sessionId, $this->socket->send(...), $this->holds, $id);
                    $this->calls[$id] = $toolCallContext;
                    // Each call in its own fiber, so a slow tool does not hold up the events
                    // behind it or the other calls the model made in the same turn.
                    $this->running[$id] = async(function () use ($frame, $id, $toolCallContext): void {
                        try {
                            $this->runTool($frame, $toolCallContext);
                        } finally {
                            unset($this->running[$id], $this->calls[$id]);
                            $this->holds->forget($id);
                        }
                    });
                    continue;
                }
                if ($kind === 'tool_cancel') {
                    // A held call is not cancelled by an interruption: only its timeout and the
                    // end of the socket stop it. A PHP callable cannot be stopped from outside;
                    // its answer is dropped instead.
                    $id = Json::string($frame, 'id');
                    if (!$this->holds->held($id)) {
                        ($this->calls[$id] ?? null)?->stop();
                        unset($this->running[$id], $this->calls[$id]);
                    }
                    $this->holds->answer($id, false);
                    continue;
                }
                if ($kind === 'tool_held') {
                    $this->holds->answer(Json::string($frame, 'id'), true);
                }
                yield SessionEvent::fromFrame($frame);
            }
        } finally {
            foreach ($this->calls as $toolCallContext) {
                $toolCallContext->stop();
            }
            $this->holds->release();
            Future\awaitAll($this->running);
        }
    }

    /**
     * Reads until the conversation ends, answering tool calls, for a caller that wants the tools
     * run and nothing else.
     */
    public function wait(): void
    {
        foreach ($this as $event) {
            unset($event);
        }
    }

    public function say(string $text): void
    {
        $this->socket->send(['type' => 'say', 'text' => $text]);
    }

    public function interrupt(): void
    {
        $this->socket->send(['type' => 'interrupt']);
    }

    /**
     * Stops watching. The session goes on; `Session::close()` ends it.
     */
    public function close(): void
    {
        $this->socket->close();
    }

    /**
     * Runs one tool and answers the model. A failure is reported rather than dropped: the model
     * is waiting on this, and can only say something useful about a tool that did not work if
     * it is told so.
     *
     * @param array<string, mixed> $frame
     */
    private function runTool(array $frame, ToolCall $toolCallContext): void
    {
        $id = Json::string($frame, 'id');
        $name = Json::string($frame, 'name');
        $result = ['type' => 'tool_result', 'tool_call_id' => $id];
        // A durable command's result has to name the request and turn it answers, so it cannot
        // be adopted by another.
        foreach (['request_id', 'turn_id'] as $key) {
            if (Json::string($frame, $key) !== '') {
                $result[$key] = $frame[$key];
            }
        }
        try {
            if ($this->tools === null) {
                throw new \RuntimeException("{$name} was asked for, and no tools are registered");
            }
            $arguments = $frame['arguments'] ?? '';
            $result['output'] = $this->tools->call($name, is_string($arguments) ? $arguments : Json::encode($arguments), $toolCallContext);
        } catch (Throwable $failed) {
            $result['error'] = $failed->getMessage();
        }
        if (isset($this->running[$id]) && !$this->socket->closed()) {
            $this->socket->send($result);
        }
    }
}
