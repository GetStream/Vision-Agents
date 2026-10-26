<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use Amp\Websocket\WebsocketClient;
use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SessionTool;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Tests\Support\LocalSocketServer;
use GetStream\VisionAgents\Tests\Support\Rows;
use GetStream\VisionAgents\ToolCall;
use GetStream\VisionAgents\Tools;
use GetStream\VisionAgents\Worker\Dispatch;
use PHPUnit\Framework\TestCase;

use function Amp\delay;

final class ToolCallTest extends TestCase
{
    private ?LocalSocketServer $server = null;

    protected function tearDown(): void
    {
        $this->server?->stop();
    }

    public function testAToolIsToldWhoItsCallAnswersAndCanSayHowItIsGoing(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, [
                'type' => 'tool_call', 'id' => 't1', 'name' => 'book', 'arguments' => '{}', 'turn_id' => 'turn-1', 'approved' => true,
                'user' => ['id' => 'u1', 'name' => 'Ada', 'custom' => ['tier' => 'gold']],
                'conversation' => ['cid' => 'messaging:c1', 'custom' => []],
            ]);
            self::until($socket, $server, 'tool_result');
        });
        $agent = new Agent(name: 'jean', client: $server->client());
        $agent->tools->register('book', 'Books a flight', [], static function (array $args, ToolCall $toolCallContext): string {
            $toolCallContext->update('Looking at flights');
            $toolCallContext->say('One moment.');
            return implode(' ', [$toolCallContext->id, $toolCallContext->sessionId, $toolCallContext->turnId, $toolCallContext->user->id, $toolCallContext->user->name,
                Json::string($toolCallContext->user->custom, 'tier'), $toolCallContext->conversation->cid, $toolCallContext->approved ? 'approved' : 'asked']);
        });

        self::watch($server, $agent);

        self::assertSame([
            ['type' => 'tool_update', 'tool_call_id' => 't1', 'text' => 'Looking at flights', 'verbatim' => false],
            ['type' => 'tool_update', 'tool_call_id' => 't1', 'text' => 'One moment.', 'verbatim' => true],
        ], $server->sent('tool_update'));
        self::assertSame('t1 ses_1 turn-1 u1 Ada gold messaging:c1 approved', $server->sent('tool_result')[0]['output']);
    }

    public function testACallTheRouterRunsElsewhereIsLeftAlone(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'f1', 'name' => 'fast', 'arguments' => '{}', 'run' => false]);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'f2', 'name' => 'fast', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_result');
            delay(0.1);
        });
        $agent = new Agent(name: 'jean', client: $server->client());
        self::payments($agent->tools);

        self::watch($server, $agent);

        self::assertSame(['f2'], array_map(static fn (array $result): string => Json::string($result, 'tool_call_id'), $server->sent('tool_result')));
    }

    public function testAHeldCallIsNotCancelledByThePersonTalkingOverTheAgent(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'p1', 'name' => 'pay', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_hold');
            LocalSocketServer::send($socket, ['type' => 'tool_held', 'id' => 'p1']);
            LocalSocketServer::send($socket, ['type' => 'tool_cancel', 'id' => 'p1']);
            self::until($socket, $server, 'tool_result');
        });
        $agent = new Agent(name: 'jean', client: $server->client());
        self::payments($agent->tools);

        $kinds = self::watch($server, $agent);

        self::assertSame([['type' => 'tool_hold', 'tool_call_id' => 'p1']], $server->sent('tool_hold'));
        self::assertSame([['type' => 'tool_result', 'tool_call_id' => 'p1', 'output' => 'paid for ses_1']], $server->sent('tool_result'));
        self::assertSame(['tool_held', 'left'], $kinds);
    }

    public function testAHoldAskedForTooLateIsCancelledWithTheCall(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'p1', 'name' => 'pay', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_hold');
            LocalSocketServer::send($socket, ['type' => 'tool_cancel', 'id' => 'p1']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'f1', 'name' => 'fast', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_result');
            delay(0.4);
        });
        $agent = new Agent(name: 'jean', client: $server->client());
        self::payments($agent->tools);

        self::watch($server, $agent);

        self::assertSame(['f1'], array_map(static fn (array $result): string => Json::string($result, 'tool_call_id'), $server->sent('tool_result')));
    }

    public function testACallThatOutlastsItsTimeoutIsStoppedAndFails(): void
    {
        $stopped = false;
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 's1', 'name' => 'slow', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_result');
        });
        $agent = new Agent(name: 'jean', client: $server->client());
        $agent->tools->register('slow', 'Takes a while', [], static function (array $args, ToolCall $toolCallContext) use (&$stopped): string {
            try {
                delay(5, cancellation: $toolCallContext->cancellation());
            } finally {
                $stopped = $toolCallContext->cancelled();
            }
            return 'done';
        }, timeout: 0.05);

        self::watch($server, $agent);

        self::assertSame([['type' => 'tool_result', 'tool_call_id' => 's1', 'error' => 'slow took longer than 0.05s']], $server->sent('tool_result'));
        delay(0.05);
        self::assertTrue($stopped, 'a call past its timeout was left running');
    }

    public function testAToolIsDeclaredWithOnlyThePoliciesItSets(): void
    {
        $tools = new Tools();
        self::payments($tools);
        $tools->register('refund', 'Refunds', self::amount(), static fn (array $args): string => 'refunded', interruptible: true, timeout: 5, approval: ' Refund {amount}? ');

        $declared = [];
        foreach ($tools->declared() as $tool) {
            $declared[$tool->name] = $tool->toArray();
        }

        self::assertSame(['title' => 'Refund {amount}?'], $declared['refund']['approval']);
        self::assertSame([true, 5000], [$declared['refund']['interruptible'], $declared['refund']['timeout_ms']]);
        self::assertSame(['description' => 'Does not', 'name' => 'fast'], $declared['fast']);
    }

    public function testAPolicyThatCannotWorkIsRefusedWhenItIsRegistered(): void
    {
        $tools = new Tools();
        $refused = static function (?float $timeout = null, ?string $approval = null) use ($tools): ?string {
            try {
                $tools->register('refund', 'Refunds', self::amount(), static fn (array $args): string => '', timeout: $timeout, approval: $approval);
                return null;
            } catch (ConfigurationException $refused) {
                return $refused->getMessage();
            }
        };

        self::assertSame("refund's approval names {card}, which is not one of its parameters", $refused(approval: 'Refund {amount} to {card}?'));
        self::assertSame("refund's approval question cannot be blank", $refused(approval: '  '));
        self::assertSame('refund has a timeout that is not positive', $refused(timeout: 0));
        self::assertFalse($tools->has('refund'));
        self::assertNull($refused(approval: 'Refund {amount}? {not a placeholder}'));
        self::assertTrue($tools->has('refund'));
    }

    public function testAHostedToolIsDeclaredWithItsPoliciesAndAnsweredByCallId(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-1']);
            self::until($socket, $server, 'host_tools');
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'c1', 'session_id' => 's1', 'name' => 'fast', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_result');
            $socket->close();
        });
        $agent = $server->client()->agent('payments');
        self::payments($agent->tools);

        (new Dispatch(client: $server->client()))->host($agent)->run();

        $declared = [];
        foreach (Json::objects($server->sent('host_tools')[0], 'tools') as $tool) {
            $declared[Json::string($tool, 'name')] = $tool;
        }
        self::assertSame(['title' => 'Pay {amount}?'], $declared['pay']['approval']);
        self::assertTrue($declared['slow']['interruptible']);
        self::assertArrayNotHasKey('interruptible', $declared['fast']);
        self::assertSame([['type' => 'tool_result', 'tool_call_id' => 'c1', 'id' => 'c1', 'output' => 'done']], $server->sent('tool_result'));
    }

    public function testAHostedCallCancelledByAnInterruptionIsNotAnswered(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-1']);
            self::until($socket, $server, 'host_tools');
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'c1', 'session_id' => 's1', 'name' => 'slow', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'c1', 'session_id' => 's2', 'name' => 'fast', 'arguments' => '{}', 'run' => false]);
            // Ids are only unique within a session: this names a call that is not running.
            LocalSocketServer::send($socket, ['type' => 'tool_cancel', 'id' => 'c1', 'session_id' => 's3']);
            LocalSocketServer::send($socket, ['type' => 'tool_cancel', 'id' => 'c1', 'session_id' => 's1']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'c2', 'session_id' => 's1', 'name' => 'fast', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_result');
            delay(0.4);
            $socket->close();
        });
        $agent = $server->client()->agent('payments');
        self::payments($agent->tools);

        (new Dispatch(client: $server->client()))->host($agent)->run();

        self::assertSame([['type' => 'tool_result', 'tool_call_id' => 'c2', 'id' => 'c2', 'output' => 'done']], $server->sent('tool_result'));
    }

    public function testAHostedCallHeldThroughInterruptionsIsAnswered(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-1']);
            self::until($socket, $server, 'host_tools');
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'c1', 'session_id' => 's1', 'name' => 'pay', 'arguments' => '{}']);
            self::until($socket, $server, 'tool_hold');
            LocalSocketServer::send($socket, ['type' => 'tool_held', 'id' => 'c1', 'session_id' => 's1']);
            LocalSocketServer::send($socket, ['type' => 'tool_cancel', 'id' => 'c1', 'session_id' => 's1']);
            self::until($socket, $server, 'tool_result');
            $socket->close();
        });
        $agent = $server->client()->agent('payments');
        self::payments($agent->tools);

        (new Dispatch(client: $server->client()))->host($agent)->run();

        self::assertSame([['type' => 'tool_hold', 'tool_call_id' => 'c1']], $server->sent('tool_hold'));
        self::assertSame([['type' => 'tool_result', 'tool_call_id' => 'c1', 'id' => 'c1', 'output' => 'paid for s1']], $server->sent('tool_result'));
    }

    /**
     * `pay`, which holds itself through interruptions and asks first, `slow`, which may be
     * interrupted, and `fast`.
     */
    private static function payments(Tools $tools): void
    {
        $tools
            ->register('pay', 'Pays', self::amount(), static function (array $args, ToolCall $toolCallContext): string {
                $toolCallContext->disallowInterruptions();
                delay(0.2);
                return 'paid for ' . $toolCallContext->sessionId;
            }, approval: 'Pay {amount}?')
            ->register('slow', 'Takes a while', [], static function (array $args, ToolCall $toolCallContext): string {
                delay(5, cancellation: $toolCallContext->cancellation());
                return 'done';
            }, interruptible: true)
            ->register('fast', 'Does not', [], static fn (array $args): string => 'done');
    }

    /**
     * @return array<string, mixed>
     */
    private static function amount(): array
    {
        return ['type' => 'object', 'properties' => ['amount' => ['type' => 'number']]];
    }

    /**
     * Reads what the SDK sends until a frame of `$type` arrives.
     */
    private static function until(WebsocketClient $socket, LocalSocketServer $server, string $type): void
    {
        while ($server->sent($type) === [] && $server->next($socket) !== null) {
        }
    }

    /**
     * Watches a session until the script is done with it, as the kinds of event it yielded.
     *
     * @return list<string>
     */
    private static function watch(LocalSocketServer $server, Agent $agent): array
    {
        $session = new Session($server->client(), SessionRow::fromArray(Rows::session()), $agent);
        $kinds = [];
        foreach ($session->watch() as $event) {
            $kinds[] = $event->kind;
        }
        return $kinds;
    }

    private function serve(\Closure $script): LocalSocketServer
    {
        return $this->server = new LocalSocketServer(static function (WebsocketClient $socket, LocalSocketServer $server) use ($script): void {
            $script($socket, $server);
            if (!$socket->isClosed()) {
                LocalSocketServer::send($socket, ['type' => 'left']);
                $socket->close();
            }
        });
    }
}
