<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use Amp\Websocket\WebsocketClient;
use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SessionTool;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\LocalSocketServer;
use GetStream\VisionAgents\Tests\Support\Rows;
use GetStream\VisionAgents\ToolCallContext;
use GetStream\VisionAgents\Tools;
use GetStream\VisionAgents\Worker\Dispatch;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

use function Amp\async;
use function Amp\delay;

final class ToolCallTest extends TestCase
{
    private const string ADD = 'POST /v1/agents/sessions/ses_1/tools';
    private const string REMOVE = 'DELETE /v1/agents/sessions/ses_1/tools/reserve_table';

    private ?LocalSocketServer $server = null;
    /**
     * @var array<string, string> who each `reserving` function that ran is, by the call it ran
     *     for, kept as it starts: a watch is over only once every call it started has ended, so
     *     this holds every one that ran, however slow
     */
    private array $ran = [];

    protected function tearDown(): void
    {
        $this->server?->stop();
    }

    public function testAToolIsToldWhoItsCallAnswers(): void
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
        $agent->tools->register('book', 'Books a flight', [], static function (ToolCallContext $toolCallContext, array $args): string {
            return implode(' ', [$toolCallContext->id, $toolCallContext->sessionId, $toolCallContext->turnId, $toolCallContext->user->id, $toolCallContext->user->name,
                Json::string($toolCallContext->user->custom, 'tier'), $toolCallContext->conversation->cid, $toolCallContext->approved ? 'approved' : 'asked']);
        });

        self::watch($server, $agent);

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
        $agent->tools->register('slow', 'Takes a while', [], static function (ToolCallContext $toolCallContext, array $args) use (&$stopped): string {
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

    public function testAToolIsHandedTheCallFirstOnlyWhenItTakesOne(): void
    {
        $tools = (new Tools())
            ->register('simple', 'Only the arguments', [], static fn (array $args): string => Json::encode(func_get_args()))
            ->register('context', 'The call, then the arguments', [], static fn (ToolCallContext $toolCallContext, array $args): string => $toolCallContext->id . ' ' . Json::encode($args))
            ->register('only_call', 'Only the call', [], static fn (ToolCallContext $toolCallContext): string => $toolCallContext->id);
        $call = new ToolCallContext(id: 'c1');

        self::assertSame('[{"city":"Rome"}]', $tools->call('simple', '{"city":"Rome"}', $call));
        self::assertSame('c1 {"city":"Rome"}', $tools->call('context', '{"city":"Rome"}', $call));
        self::assertSame('c1', $tools->call('only_call', '{}', $call));
    }

    public function testStringArgumentsAreAllRequiredUnlessTheCallerSaysWhich(): void
    {
        $properties = ['location' => 'a city or region', 'unit' => 'celsius or fahrenheit'];
        $described = [
            'location' => ['type' => 'string', 'description' => 'a city or region'],
            'unit' => ['type' => 'string', 'description' => 'celsius or fahrenheit'],
        ];

        self::assertSame(['type' => 'object', 'properties' => $described, 'required' => ['location', 'unit']], Tools::strings($properties));
        self::assertSame(['type' => 'object', 'properties' => $described, 'required' => ['location']], Tools::strings($properties, ['location']));
        self::assertSame([], Tools::strings($properties, [])['required']);
    }

    public function testAToolWithNoStringArgumentsIsSentAnEmptyPropertiesObject(): void
    {
        $router = new LocalRouter();
        try {
            $router->answer('POST', '/v1/agents/sessions', 201, Rows::session());
            $agent = new Agent(name: 'jean', client: $router->client());
            $agent->tools->register('ping', 'Checks the line', Tools::strings([]), static fn (array $args): string => 'pong');

            $agent->chat();

            self::assertStringContainsString(
                '"tools":[{"description":"Checks the line","name":"ping","parameters":{"type":"object","properties":{},"required":[]}}]',
                $router->to('POST', '/v1/agents/sessions')[0]->body,
            );
        } finally {
            $router->stop();
        }
    }

    public function testAToolRegisteredWithNamedArgumentsRuns(): void
    {
        $tools = (new Tools())->register(
            name: 'weather_lookup',
            description: 'Look up the current weather for a location',
            parameters: Tools::strings(['location' => 'a city or region']),
            run: static fn (array $args): string => 'sunny in ' . Json::string($args, 'location'),
        );

        self::assertSame('sunny in Lyon', $tools->call('weather_lookup', '{"location":"Lyon"}'));
        self::assertSame(['location'], Json::object($tools->declared()[0]->toArray(), 'parameters')['required']);
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
        $client = $server->client();
        self::payments($client->tools);

        (new Dispatch(client: $client))->host($client)->run();

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
        $client = $server->client();
        self::payments($client->tools);

        (new Dispatch(client: $client))->host($client)->run();

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
        $client = $server->client();
        self::payments($client->tools);

        (new Dispatch(client: $client))->host($client)->run();

        self::assertSame([['type' => 'tool_hold', 'tool_call_id' => 'c1']], $server->sent('tool_hold'));
        self::assertSame([['type' => 'tool_result', 'tool_call_id' => 'c1', 'id' => 'c1', 'output' => 'paid for s1']], $server->sent('tool_result'));
    }

    public function testAToolAddedWhileTheSessionIsWatchedIsRunHere(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{"table":"T4"}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $session = self::session($server);

        $refused = self::watchChanging($session, ['add' => fn () => $session->addTools($this->reserving('first'))]);

        self::assertSame([], $refused);
        self::assertSame([self::ADD . ' {"tools":[{"description":"Reserve a table","name":"reserve_table","parameters":{"type":"object","properties":{"table":{"type":"string"}}}}]}'], $server->requests);
        self::assertSame(['r1' => 'first T4', 'w1' => 'sunny'], self::answers($server));
    }

    public function testAToolTheRouterRefusedIsNotRunHere(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [409];
        $session = self::session($server);

        $refused = self::watchChanging($session, ['add' => fn () => $session->addTools($this->reserving('first'))]);

        self::assertSame(['add' => 409], $refused);
        self::assertSame(['w1' => 'sunny'], self::answers($server));
        self::assertSame([], $this->ran);
    }

    public function testARemovedToolIsNoLongerRunHere(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'remove']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r2', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $server->answers['DELETE /v1/agents/sessions/ses_1/tools/reserve_table'] = [204];
        $session = self::session($server);

        $refused = self::watchChanging($session, [
            'add' => fn () => $session->addTools($this->reserving('first')),
            'remove' => static fn () => $session->removeTool('reserve_table'),
        ]);

        self::assertSame([], $refused);
        self::assertSame(['r1' => 'first ', 'w1' => 'sunny'], self::answers($server));
        self::assertSame(['r1' => 'first'], $this->ran);
        self::assertSame('DELETE /v1/agents/sessions/ses_1/tools/reserve_table', $server->requests[1]);
    }

    public function testANameTheSessionRunsKeepsItsFunctionWhenAddedAgain(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        // The second answer is lost, which leaves its fate unknown here; the router never
        // replaces a tool the session has, its own or one added.
        $server->answers[self::ADD] = [204, -1];
        $session = self::session($server);
        $session->addTools($this->reserving('first'));
        $again = $this->reserving('second')->register('get_weather', 'The weather', [], static fn (array $args): string => 'stormy');

        try {
            $session->addTools($again);
            self::fail('an add whose answer was lost reported success');
        } catch (RouterException $lost) {
            self::assertSame(0, $lost->status);
        }
        self::watchChanging($session, []);

        self::assertSame(['r1' => 'first ', 'w1' => 'sunny'], self::answers($server));
    }

    public function testAToolWithdrawnFromTheAgentsOwnRunsNothingUntilAnotherIsAddedInItsPlace(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'remove']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r2', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $server->answers['DELETE /v1/agents/sessions/ses_1/tools/reserve_table'] = [204];
        $session = self::session($server, $this->reserving('agent'));

        $refused = self::watchChanging($session, [
            'remove' => static fn () => $session->removeTool('reserve_table'),
            'add' => fn () => $session->addTools($this->reserving('replacement')),
        ]);

        self::assertSame([], $refused);
        self::assertSame(['r2' => 'replacement ', 'w1' => 'sunny'], self::answers($server));
        self::assertSame(['r2' => 'replacement'], $this->ran);
    }

    /**
     * @return iterable<string, array{int, string}> an answer after which the router may or may not
     *     have made the change, as a status or a `LocalSocketServer::$answers` marker, and its body
     */
    public static function answersThatAreNotARefusal(): iterable
    {
        yield 'none' => [-1, ''];
        yield "a gateway's 502" => [502, ''];
        yield 'a 200 that is not JSON' => [200, '<html>done</html>'];
        yield 'a 200 cut off' => [-2, ''];
    }

    #[DataProvider('answersThatAreNotARefusal')]
    public function testAToolWhoseAddingWasNotRefusedRunsHereAndIsWithdrawnBeforeItsNameIsAddedAgain(int $answer, string $body): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'again']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r2', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w2', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w2');
        });
        // The router added it, then the answer went wrong on its way here.
        $server->answers[self::ADD] = [$answer, 204];
        $server->bodies[self::ADD] = $body;
        $server->answers[self::REMOVE] = [204];
        $session = self::session($server);
        try {
            $session->addTools($this->reserving('first'));
            $failed = null;
        } catch (\RuntimeException $failed) {
        }

        $refused = self::watchChanging($session, ['again' => fn () => $session->addTools($this->reserving('second'))]);

        self::assertNotNull($failed, 'an add the router did not answer with a success reported one');
        self::assertSame([], $refused);
        self::assertSame(['r1' => 'first ', 'w1' => 'sunny', 'r2' => 'second ', 'w2' => 'sunny'], self::answers($server));
        self::assertSame(['r1' => 'first', 'r2' => 'second'], $this->ran);
        self::assertSame([self::ADD, self::REMOVE, self::ADD], array_map(static fn (string $request): string => explode(' {', $request)[0], $server->requests));
    }

    public function testRemovingAToolWhoseAddingGotNoAnswerTakesNotFoundAsDone(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'remove unknown']);
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'remove']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [-1];
        $server->answers['DELETE /v1/agents/sessions/ses_1/tools/reserve_table'] = [404];
        $server->answers['DELETE /v1/agents/sessions/ses_1/tools/book_flight'] = [404];
        $session = self::session($server);

        $refused = self::watchChanging($session, [
            'remove unknown' => static fn () => $session->removeTool('book_flight'),
            'add' => fn () => $session->addTools($this->reserving('first')),
            'remove' => static fn () => $session->removeTool('reserve_table'),
        ]);

        self::assertSame(['remove unknown' => 404, 'add' => 0], $refused);
        self::assertSame(['w1' => 'sunny'], self::answers($server));
        self::assertSame([], $this->ran);
    }

    #[DataProvider('answersThatAreNotARefusal')]
    public function testAToolWhoseRemovalWasNotRefusedRunsHereUntilItIsRemovedAgain(int $answer, string $body): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'remove again']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r2', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w2', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w2');
        });
        // The router withdrew it and the answer went wrong on its way here: asked again, it has
        // no tool by the name. Sent again by the transport on its own, that 404 would be read as
        // the answer.
        $server->answers[self::ADD] = [204];
        $server->answers[self::REMOVE] = [$answer, 404];
        $server->bodies[self::REMOVE] = $body;
        $session = self::session($server);
        $session->addTools($this->reserving('first'));
        try {
            $session->removeTool('reserve_table');
            $failed = null;
        } catch (\RuntimeException $failed) {
        }

        $refused = self::watchChanging($session, ['remove again' => static fn () => $session->removeTool('reserve_table')]);

        self::assertNotNull($failed, 'a removal the router did not answer with a success reported one');
        self::assertSame([], $refused);
        self::assertSame(['r1' => 'first ', 'w1' => 'sunny', 'w2' => 'sunny'], self::answers($server));
        self::assertSame(['r1' => 'first'], $this->ran);
        self::assertSame([self::ADD, self::REMOVE, self::REMOVE], array_map(static fn (string $request): string => explode(' {', $request)[0], $server->requests));
    }

    public function testAToolWhoseRemovalGotNoAnswerIsWithdrawnBeforeItsNameIsAddedAgain(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'remove']);
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        // The router withdrew it and the answer was lost, so it takes the name again.
        $server->answers[self::ADD] = [204];
        $server->answers[self::REMOVE] = [-1, 404];
        $session = self::session($server);
        $session->addTools($this->reserving('first'));

        $refused = self::watchChanging($session, [
            'remove' => static fn () => $session->removeTool('reserve_table'),
            'add' => fn () => $session->addTools($this->reserving('second')),
        ]);

        self::assertSame(['remove' => 0], $refused);
        self::assertSame(['r1' => 'second ', 'w1' => 'sunny'], self::answers($server));
        self::assertSame([self::ADD, self::REMOVE, self::REMOVE, self::ADD], array_map(static fn (string $request): string => explode(' {', $request)[0], $server->requests));
    }

    public function testAnAddedToolKeepsItsFunctionWhenAnotherIsRegisteredUnderItsName(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{"table":"T4"}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $session = self::session($server);
        $tools = $this->reserving('first');
        $session->addTools($tools);

        $tools->register('reserve_table', 'Reserve a table', [], static fn (array $args): string => 'replaced');
        self::watchChanging($session, []);

        self::assertSame(['r1' => 'first T4', 'w1' => 'sunny'], self::answers($server));
    }

    public function testAForkRunsTheToolsTheSessionHadWhenItWasForked(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'b1', 'name' => 'book_flight', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $server->answers[self::REMOVE] = [204];
        $server->answers['POST /v1/agents/sessions/ses_1/fork'] = [201];
        $server->bodies['POST /v1/agents/sessions/ses_1/fork'] = Rows::session('ses_2');
        $session = self::session($server, $this->reserving('agent'));
        $session->addTools((new Tools())->register('book_flight', 'Book a flight', [], static fn (array $args): string => 'booked'));
        $session->removeTool('reserve_table');

        $fork = $session->fork();
        self::watchChanging($fork, []);

        self::assertStringStartsWith('/v1/agents/sessions/ses_2/events', $server->handshakes[0]);
        self::assertSame(['b1' => 'booked', 'w1' => 'sunny'], self::answers($server));
        self::assertSame([], $this->ran);
    }

    public function testAForkWithdrawsANameWhoseAddingGotNoAnswerBeforeItAddsItAgain(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'add']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        // The parent's add never reached the router, which so gives the fork no tool by the name
        // and takes the fork's add of it.
        $server->answers[self::ADD] = [-1];
        $server->answers['POST /v1/agents/sessions/ses_1/fork'] = [201];
        $server->bodies['POST /v1/agents/sessions/ses_1/fork'] = Rows::session('ses_2');
        $server->answers['DELETE /v1/agents/sessions/ses_2/tools/reserve_table'] = [404];
        $server->answers['POST /v1/agents/sessions/ses_2/tools'] = [204];
        $session = self::session($server);
        try {
            $session->addTools($this->reserving('first'));
            self::fail('an add whose answer was lost reported success');
        } catch (RouterException $lost) {
            self::assertSame(0, $lost->status);
        }

        $fork = $session->fork();
        $refused = self::watchChanging($fork, ['add' => fn () => $fork->addTools($this->reserving('second'))]);

        self::assertSame([], $refused);
        self::assertSame(['r1' => 'second ', 'w1' => 'sunny'], self::answers($server));
        self::assertSame(['r1' => 'second'], $this->ran);
        self::assertSame(
            [self::ADD, 'POST /v1/agents/sessions/ses_1/fork', 'DELETE /v1/agents/sessions/ses_2/tools/reserve_table', 'POST /v1/agents/sessions/ses_2/tools'],
            array_map(static fn (string $request): string => explode(' {', $request)[0], $server->requests),
        );
    }

    public function testChangesToASessionsToolsAreMadeOneAtATime(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        // Interleaved, the refusal of the first would take back the function the second put in
        // place, and the router would offer a tool nothing here runs.
        $server->answers[self::ADD] = [409, 204];
        $server->answerAfter = 0.05;
        $session = self::session($server);

        $first = async(fn () => $session->addTools($this->reserving('first')));
        $second = async(fn () => $session->addTools($this->reserving('second')));
        try {
            $first->await();
            self::fail('a refused add reported success');
        } catch (RouterException $refused) {
            self::assertSame(409, $refused->status);
        }
        $second->await();
        self::watchChanging($session, []);

        self::assertSame(['r1' => 'second ', 'w1' => 'sunny'], self::answers($server));
    }

    public function testClosingForgetsTheToolsTheSessionWasGiven(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $server->answers['POST /v1/agents/sessions/ses_1/stop'] = [204];
        $session = self::session($server);
        $session->addTools($this->reserving('first'));

        $session->close();
        self::watchChanging($session, []);

        self::assertSame(['w1' => 'sunny'], self::answers($server));
        self::assertSame([], $this->ran);
    }

    public function testAToolIsKeptWhenStoppingTheSessionFailsAndClosingIsAskedAgain(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'close']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'r1', 'name' => 'reserve_table', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 'w1', 'name' => 'get_weather', 'arguments' => '{}']);
            self::untilAnswered($socket, $server, 'w1');
        });
        $server->answers[self::ADD] = [204];
        $server->answers['POST /v1/agents/sessions/ses_1/stop'] = [503, 204];
        $session = self::session($server);
        $session->addTools($this->reserving('first'));

        $refused = self::watchChanging($session, ['close' => static fn () => $session->close()]);
        $session->close();

        self::assertSame(['close' => 503], $refused);
        self::assertSame(['r1' => 'first ', 'w1' => 'sunny'], self::answers($server));
        self::assertTrue($session->closed());
        self::assertSame(['POST /v1/agents/sessions/ses_1/stop', 'POST /v1/agents/sessions/ses_1/stop'], array_slice($server->requests, 1));
    }

    public function testAddingNoToolsIsRefusedBeforeAsking(): void
    {
        $server = $this->serve(static fn (WebsocketClient $socket, LocalSocketServer $server) => null);

        try {
            self::session($server)->addTools(new Tools());
            self::fail('an empty add was sent');
        } catch (ConfigurationException $refused) {
            self::assertSame('there are no tools to add', $refused->getMessage());
        }
        self::assertSame([], $server->requests);
    }

    /**
     * `pay`, which holds itself through interruptions and asks first, `slow`, which may be
     * interrupted, and `fast`.
     */
    private static function payments(Tools $tools): void
    {
        $tools
            ->register('pay', 'Pays', self::amount(), static function (ToolCallContext $toolCallContext, array $args): string {
                $toolCallContext->disallowInterruptions();
                delay(0.2);
                return 'paid for ' . $toolCallContext->sessionId;
            }, approval: 'Pay {amount}?')
            ->register('slow', 'Takes a while', [], static function (ToolCallContext $toolCallContext, array $args): string {
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

    /**
     * `reserve_table`, which answers with `$who` and the table it was asked for, and is kept in
     * `ran`.
     */
    private function reserving(string $who): Tools
    {
        return (new Tools())->register('reserve_table', 'Reserve a table', ['type' => 'object', 'properties' => ['table' => ['type' => 'string']]],
            function (ToolCallContext $toolCallContext, array $args) use ($who): string {
                $this->ran[$toolCallContext->id] = $who;
                return $who . ' ' . Json::string($args, 'table');
            });
    }

    /**
     * A session whose agent runs `get_weather`, and `$more` of its own.
     */
    private static function session(LocalSocketServer $server, ?Tools $more = null): Session
    {
        $tools = $more ?? new Tools();
        $tools->register('get_weather', 'The weather', [], static fn (array $args): string => 'sunny');
        return new Session($server->client(), SessionRow::fromArray(Rows::session()), new Agent(name: 'jean', client: $server->client(), tools: $tools));
    }

    /**
     * Watches a session until the script is done with it, making the change each `heard` names.
     *
     * @param array<string, \Closure(): void> $changes
     * @return array<string, int> the status each change that threw was refused with, by name
     */
    private static function watchChanging(Session $session, array $changes): array
    {
        $refused = [];
        foreach ($session->watch() as $event) {
            if ($event->kind !== 'heard') {
                continue;
            }
            try {
                $changes[$event->text()]();
            } catch (RouterException $failed) {
                $refused[$event->text()] = $failed->status;
            }
        }
        return $refused;
    }

    /**
     * Reads what the SDK sends until the call `$id` is answered. A call sent before it may still
     * be running then, so whether one ran is read from `ran` once the watch is over.
     */
    private static function untilAnswered(WebsocketClient $socket, LocalSocketServer $server, string $id): void
    {
        while (!array_key_exists($id, self::answers($server)) && $server->next($socket) !== null) {
        }
    }

    /**
     * @return array<string, string> each call's output, by its id, in the order answered
     */
    private static function answers(LocalSocketServer $server): array
    {
        $answers = [];
        foreach ($server->sent('tool_result') as $result) {
            $answers[Json::string($result, 'tool_call_id')] = Json::string($result, 'output');
        }
        return $answers;
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
