<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use Amp\Cancellation;
use Amp\DeferredCancellation;
use Closure;
use GetStream\VisionAgents\Exception\CallNotHeldException;
use GetStream\VisionAgents\Worker\Holds;

/**
 * One call of one of the caller's tools: who it is for, and its line back to them.
 *
 * A tool whose callable takes a second argument is handed one. A call made with no socket
 * behind it, as a test makes, has no line, so its updates go nowhere.
 */
final class ToolCall
{
    private readonly DeferredCancellation $stopping;

    /**
     * @param string $id the id the model gave the call, which a call sent again keeps
     * @param bool $approved whether a person allowed this call before it ran
     * @param ?Closure(array<string, mixed>): void $send sends a frame on the socket the call came on
     * @param string $key what that socket's `tool_cancel` names the call by
     */
    public function __construct(
        public readonly string $id = '',
        public readonly string $sessionId = '',
        public readonly string $turnId = '',
        public readonly ToolUser $user = new ToolUser(),
        public readonly ToolConversation $conversation = new ToolConversation(),
        public readonly bool $approved = false,
        private readonly ?Closure $send = null,
        private readonly ?Holds $holds = null,
        private readonly string $key = '',
    ) {
        $this->stopping = new DeferredCancellation();
    }

    /**
     * The call a `tool_call` frame asks for. `$sessionId` stands in for a frame that does not
     * name its session.
     *
     * @internal
     * @param array<string, mixed> $frame
     * @param Closure(array<string, mixed>): void $send
     */
    public static function fromFrame(array $frame, string $sessionId, Closure $send, Holds $holds, string $key): self
    {
        $user = Json::object($frame, 'user');
        $conversation = Json::object($frame, 'conversation');
        return new self(
            id: Json::string($frame, 'id'),
            sessionId: Json::string($frame, 'session_id') !== '' ? Json::string($frame, 'session_id') : $sessionId,
            turnId: Json::string($frame, 'turn_id'),
            user: new ToolUser(Json::string($user, 'id'), Json::string($user, 'name'), Json::object($user, 'custom')),
            conversation: new ToolConversation(Json::string($conversation, 'cid'), Json::object($conversation, 'custom')),
            approved: Json::bool($frame, 'approved'),
            send: $send,
            holds: $holds,
            key: $key,
        );
    }

    /**
     * Tells the people in the conversation how the call is going, such as "Found three
     * flights, comparing prices". The agent puts it in its own words.
     */
    public function update(string $text): void
    {
        $this->sendUpdate($text, false);
    }

    /**
     * `update()`, said word for word.
     */
    public function say(string $text): void
    {
        $this->sendUpdate($text, true);
    }

    /**
     * Keeps the call going if the person talks over the agent from here on, for a call that is
     * past the point of taking back, such as one that has started a payment.
     *
     * The router has the last word. A held call still ends at its timeout, or with the socket.
     *
     * @throws CallNotHeldException when the call had already been cancelled
     */
    public function disallowInterruptions(): void
    {
        if ($this->send === null || $this->holds === null || $this->holds->held($this->key)) {
            return;
        }
        if ($this->cancelled()) {
            throw new CallNotHeldException($this->id);
        }
        $answer = $this->holds->ask($this->key);
        ($this->send)(['type' => 'tool_hold', 'tool_call_id' => $this->id]);
        if (!$answer->await()) {
            throw new CallNotHeldException($this->id);
        }
    }

    /**
     * Whether the router gave up on the call, or it ran out of time. A PHP fiber cannot be
     * stopped from outside, so a long tool checks this, and its answer is dropped either way.
     */
    public function cancelled(): bool
    {
        return $this->stopping->isCancelled();
    }

    /**
     * Cancelled with the call, for handing to Amp's own I/O, such as `delay()` or an HTTP
     * request, so it stops with it.
     */
    public function cancellation(): Cancellation
    {
        return $this->stopping->getCancellation();
    }

    /**
     * @internal
     */
    public function stop(): void
    {
        $this->stopping->cancel();
    }

    private function sendUpdate(string $text, bool $verbatim): void
    {
        if ($this->send !== null) {
            ($this->send)(['type' => 'tool_update', 'tool_call_id' => $this->id, 'text' => $text, 'verbatim' => $verbatim]);
        }
    }
}
