<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use Amp\Future;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
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

    private function __construct(private readonly Socket $socket, private readonly ?Tools $tools)
    {
    }

    public static function open(Session $session, ?Tools $tools, bool $interim = false, bool $decisions = true): self
    {
        $socket = Socket::open($session->client, "/v1/agents/sessions/{$session->id()}/events", [
            'interim' => $interim,
            'decisions' => $decisions,
        ]);
        return new self($socket, $tools);
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
                    $id = Json::string($frame, 'id');
                    // Each call in its own fiber, so a slow tool does not hold up the events
                    // behind it or the other calls the model made in the same turn.
                    $this->running[$id] = async(function () use ($frame, $id): void {
                        try {
                            $this->runTool($frame);
                        } finally {
                            unset($this->running[$id]);
                        }
                    });
                    continue;
                }
                if ($kind === 'tool_cancel') {
                    // A PHP callable cannot be stopped from outside; its answer is dropped instead.
                    unset($this->running[Json::string($frame, 'id')]);
                    continue;
                }
                yield SessionEvent::fromFrame($frame);
            }
        } finally {
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

    public function respond(string $text): void
    {
        $this->socket->send(['type' => 'respond', 'text' => $text]);
    }

    public function interrupt(): void
    {
        $this->socket->send(['type' => 'interrupt']);
    }

    public function setInstructions(string $instructions): void
    {
        $this->socket->send(['type' => 'instructions', 'instructions' => $instructions]);
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
    private function runTool(array $frame): void
    {
        $id = Json::string($frame, 'id');
        $name = Json::string($frame, 'name');
        $result = ['type' => 'tool_result', 'tool_call_id' => $id];
        // A durable command's result has to name the command and turn it answers, so it cannot
        // be adopted by another.
        foreach (['command_id', 'turn_id'] as $key) {
            if (Json::string($frame, $key) !== '') {
                $result[$key] = $frame[$key];
            }
        }
        try {
            if ($this->tools === null) {
                throw new \RuntimeException("{$name} was asked for, and no tools are registered");
            }
            $arguments = $frame['arguments'] ?? '';
            $result['output'] = $this->tools->call($name, is_string($arguments) ? $arguments : Json::encode($arguments));
        } catch (Throwable $failed) {
            $result['error'] = $failed->getMessage();
        }
        if (isset($this->running[$id]) && !$this->socket->closed()) {
            $this->socket->send($result);
        }
    }
}
