<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use Amp\Websocket\WebsocketClient;
use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\HostingRefusedException;
use GetStream\VisionAgents\Exception\RealtimeException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SttOptions;
use GetStream\VisionAgents\Inbound\InboundCall;
use GetStream\VisionAgents\Inbound\InboundMessage;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\LocalSocketServer;
use GetStream\VisionAgents\Tests\Support\Rows;
use GetStream\VisionAgents\Worker\Dispatch;
use PHPUnit\Framework\TestCase;
use RuntimeException;

use function Amp\delay;

final class WorkerTest extends TestCase
{
    private ?LocalSocketServer $server = null;
    private ?LocalRouter $router = null;

    protected function tearDown(): void
    {
        $this->server?->stop();
        $this->router?->stop();
    }

    public function testDispatchSaysDoneForFinishedAndFailedCalls(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-1']);
            LocalSocketServer::send($socket, ['type' => 'call', 'work_id' => 'wk1', 'call_id' => 'c1', 'caller_number' => '+15550001111', 'custom' => ['lang' => 'fr']]);
            LocalSocketServer::send($socket, ['type' => 'call', 'work_id' => 'wk2', 'call_id' => 'c2', 'call_type' => 'phone']);
            while (count($server->sent('done')) < 2 && $server->next($socket) !== null) {
            }
            $socket->close();
        });
        $answered = [];
        $dispatch = new Dispatch(capacity: 2, client: $server->client());
        $dispatch->waitForCall(static function (InboundCall $call) use (&$answered): void {
            $answered[] = $call;
            if ($call->callId === 'c2') {
                throw new RuntimeException('no agent for phone calls');
            }
        });

        $dispatch->run();

        self::assertSame('w-1', $dispatch->workerId);
        self::assertSame('/v1/dispatch?capacity=2&active=0&handles=call', $server->handshakes[0]);
        self::assertSame('examples', $server->headers[0]['x-customer-id']);
        self::assertSame([
            ['type' => 'done', 'work_id' => 'wk1'],
            ['type' => 'done', 'work_id' => 'wk2', 'error' => 'no agent for phone calls'],
        ], $server->sent('done'));
        self::assertSame([], $server->sent('accepted'));
        self::assertSame([], $server->sent('rejected'));
        self::assertSame('+15550001111', $answered[0]->callerNumber);
        self::assertSame(['lang' => 'fr'], $answered[0]->custom);
        self::assertSame('default', $answered[0]->callType);
        self::assertSame('phone', $answered[1]->callType);
        self::assertSame(0, $dispatch->active());
    }

    public function testDispatchKeepsOneSessionPerChannel(): void
    {
        $this->router = new LocalRouter();
        $this->router->answer('POST', '/v1/agents/sessions', 201, Rows::session('ses_1'), Rows::session('ses_2'));
        $http = $this->router->client();
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            foreach (['ch1', 'ch1', 'ch2'] as $i => $channel) {
                LocalSocketServer::send($socket, ['type' => 'message', 'work_id' => "wk{$i}", 'channel_id' => $channel, 'text' => 'hi', 'agent_id' => 'jean']);
            }
            while (count($server->sent('done')) < 3 && $server->next($socket) !== null) {
            }
            $socket->close();
        });
        $sessions = [];
        $dispatch = new Dispatch(client: $server->client());
        $dispatch->waitForMessage(static function (InboundMessage $message) use ($dispatch, $http, &$sessions): void {
            $session = $dispatch->getOrCreateAgent($message, static fn (): Agent => new Agent(name: 'jean', client: $http));
            $sessions[$message->channelId][] = $session->id();
        });

        $dispatch->run();

        self::assertCount(2, $this->router->to('POST', '/v1/agents/sessions'));
        self::assertCount(2, $sessions['ch1']);
        self::assertSame($sessions['ch1'][0], $sessions['ch1'][1]);
        self::assertNotSame($sessions['ch1'][0], $sessions['ch2'][0]);
        self::assertSame('/v1/dispatch?capacity=4&active=0&handles=message', $server->handshakes[0]);
        self::assertCount(3, $server->sent('done'));
    }

    public function testDispatchSaysDoneWithAnErrorForAMessageItHasNoHandlerFor(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'message', 'work_id' => 'wk1', 'channel_id' => 'ch1', 'text' => 'hi']);
            while ($server->sent('done') === [] && $server->next($socket) !== null) {
            }
            $socket->close();
        });
        $agent = $server->client()->agent('my-agent');
        $agent->tools->register('weather_lookup', 'The weather somewhere', [], static fn (array $args): string => 'sunny');

        (new Dispatch(client: $server->client()))->host($agent)->run();

        self::assertSame('/v1/dispatch?capacity=4&active=0&handles=', $server->handshakes[0]);
        self::assertSame([['type' => 'done', 'work_id' => 'wk1', 'error' => 'this worker answers no messages']], $server->sent('done'));
    }

    public function testDispatchHandsTheMessageItsSessionAndCommand(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'message', 'work_id' => 'wk1', 'session_id' => 'ses_1', 'command_id' => 'cmd_1', 'agent_id' => 'jean', 'text' => 'hi', 'user_id' => 'ada']);
            while ($server->sent('done') === [] && $server->next($socket) !== null) {
            }
            $socket->close();
        });
        $received = [];
        $dispatch = new Dispatch(client: $server->client());
        $dispatch->waitForMessage(static function (InboundMessage $message) use (&$received): void {
            $received[] = $message;
        });

        $dispatch->run();

        self::assertSame('ses_1', $received[0]->sessionId);
        self::assertSame('cmd_1', $received[0]->commandId);
        self::assertSame('', $received[0]->channelId);
    }

    public function testAnswerCreatesTheResponseActingForTheWriterWithTheServerCredential(): void
    {
        $this->router = new LocalRouter();
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1', 'running'));
        $dispatch = new Dispatch(client: new Client(new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret')));

        $dispatch->answer(new InboundMessage(channelId: '', text: 'what does it cost?', agentId: 'jean', userId: 'ada', sessionId: 'ses_1', commandId: 'cmd_1'));

        $sent = $this->router->to('POST', '/v1/agents/sessions/ses_1/responses')[0];
        self::assertSame(['text' => 'what does it cost?', 'command_id' => 'cmd_1'], $sent->json());
        self::assertSame('ada', $sent->headers['x-stream-user-id']);
        self::assertSame('server', $sent->headers['stream-auth-type']);
        self::assertStringStartsWith('Bearer ', $sent->headers['authorization']);
    }

    public function testAnswerBehindTheProxySendsTheServerTokenNotTheWritersOwn(): void
    {
        $this->router = new LocalRouter();
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1', 'running'));
        $dispatch = new Dispatch(client: new Client(new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret', authenticate: true)));

        $dispatch->answer(new InboundMessage(channelId: '', text: 'hi', userId: 'ada', sessionId: 'ses_1'));

        $sent = $this->router->to('POST', '/v1/agents/sessions/ses_1/responses')[0];
        self::assertSame('ada', $sent->headers['x-stream-user-id']);
        self::assertSame('jwt', $sent->headers['stream-auth-type']);
        $payload = explode('.', substr($sent->headers['authorization'], strlen('Bearer ')))[1];
        $claims = Json::asObject(Json::decode((string) base64_decode(strtr($payload, '-_', '+/'), true)));
        self::assertTrue($claims['server']);
        self::assertArrayNotHasKey('user_id', $claims);
    }

    public function testAnswerNeedsASession(): void
    {
        $dispatch = new Dispatch(client: new Client(new Backend(url: 'http://127.0.0.1:1', customerId: 'examples')));

        $this->expectException(ConfigurationException::class);
        $dispatch->answer(new InboundMessage(channelId: 'ch1', text: 'hi'));
    }

    public function testGetOrCreateAgentRefusesAMessageASessionIsHolding(): void
    {
        $dispatch = new Dispatch(client: new Client(new Backend(url: 'http://127.0.0.1:1', customerId: 'examples')));

        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('a session is already holding this conversation');
        $dispatch->getOrCreateAgent(
            new InboundMessage(channelId: 'ch1', text: 'hi', sessionId: 'ses_1'),
            static fn (): Agent => throw new RuntimeException('no agent should be built'),
        );
    }

    public function testDispatchReportsLoadAfterTimingARoundTrip(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            while (($frame = $server->next($socket)) !== null) {
                if ($frame['type'] === 'ping') {
                    LocalSocketServer::send($socket, ['type' => 'pong', 'at' => $frame['at']]);
                }
                if ($frame['type'] === 'load') {
                    break;
                }
            }
            $socket->close();
        });
        $dispatch = new Dispatch(reportEvery: 0.1, client: $server->client());
        $dispatch->waitForCall(static fn (InboundCall $call) => null);

        $dispatch->run();

        $load = $server->sent('load')[0];
        self::assertSame(0, $load['active_agents']);
        self::assertGreaterThan(0, $load['latency_ms']);
    }

    public function testDispatchDeclaresAndAnswersHostedTools(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-1']);
            $server->next($socket);
            LocalSocketServer::send($socket, ['type' => 'hosting', 'agent_id' => 'my-agent']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 't1', 'session_id' => 'ses_1', 'name' => 'slow', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 't2', 'session_id' => 'ses_1', 'name' => 'weather_lookup', 'arguments' => '{"location":"Boulder, Colorado"}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 't3', 'session_id' => 'ses_1', 'name' => 'broken', 'arguments' => '{}']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 't4', 'session_id' => 'ses_1', 'name' => 'read_source', 'arguments' => '{}']);
            while (count($server->sent('tool_result')) < 4 && $server->next($socket) !== null) {
            }
            $socket->close();
        });
        $agent = $server->client()->agent('my-agent');
        $agent->tools
            ->register('weather_lookup', 'The weather somewhere', ['type' => 'object', 'properties' => ['location' => ['type' => 'string']]], static fn (array $args): array => ['location' => $args['location'], 'sky' => 'sunny'])
            ->register('slow', 'Takes a while', [], static function (array $args): string {
                delay(0.2);
                return 'done';
            })
            ->register('broken', 'Always fails', [], static fn (array $args): string => throw new RuntimeException('the weather service is down'));
        $dispatch = (new Dispatch(client: $server->client()))->host($agent, toolTimeoutMs: 30000);

        $dispatch->run();

        self::assertSame([[
            'type' => 'host_tools',
            'agent_id' => 'my-agent',
            'tools' => [
                ['description' => 'The weather somewhere', 'name' => 'weather_lookup', 'parameters' => ['type' => 'object', 'properties' => ['location' => ['type' => 'string']]]],
                ['description' => 'Takes a while', 'name' => 'slow'],
                ['description' => 'Always fails', 'name' => 'broken'],
            ],
            'timeout_ms' => 30000,
        ]], $server->sent('host_tools'));
        self::assertSame(['my-agent'], $dispatch->hosting);
        $results = [];
        foreach ($server->sent('tool_result') as $result) {
            $results[Json::string($result, 'id')] = $result;
        }
        self::assertSame(['type' => 'tool_result', 'id' => 't2', 'output' => '{"location":"Boulder, Colorado","sky":"sunny"}'], $results['t2']);
        self::assertSame(['type' => 'tool_result', 'id' => 't3', 'error' => 'the weather service is down'], $results['t3']);
        self::assertSame(['type' => 'tool_result', 'id' => 't4', 'error' => 'this worker does not run read_source'], $results['t4']);
        self::assertSame(['type' => 'tool_result', 'id' => 't1', 'output' => 'done'], $results['t1']);
        self::assertSame('t1', array_key_last($results), 'a slow tool held up the calls behind it');
        self::assertSame(0, $dispatch->active());
    }

    public function testDispatchHostsAgainOnEveryConnection(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-' . count($server->handshakes)]);
            $server->next($socket);
            $socket->close();
        });
        $agent = $server->client()->agent('my-agent');
        $agent->tools->register('weather_lookup', 'The weather somewhere', [], static fn (array $args): string => 'sunny');
        $dispatch = (new Dispatch(client: $server->client()))->host($agent);

        $dispatch->run();
        $dispatch->run();

        self::assertSame('w-2', $dispatch->workerId);
        self::assertCount(2, $server->handshakes);
        self::assertSame([0, 0], array_map(static fn (array $frame): mixed => $frame['timeout_ms'], $server->sent('host_tools')));
    }

    public function testDispatchStopsWhenHostingIsRefused(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'hosting_refused', 'agent_id' => 'my-agent', 'reason' => 'hosting no tools is not hosting']);
            while ($server->next($socket) !== null) {
            }
        });
        $dispatch = (new Dispatch(client: $server->client()))->host($server->client()->agent('my-agent'));

        try {
            $dispatch->run();
            self::fail('a worker nobody will call kept waiting');
        } catch (HostingRefusedException $refused) {
            self::assertSame('my-agent', $refused->agentId);
            self::assertSame('hosting no tools is not hosting', $refused->reason);
            self::assertSame('the router refused to host tools for agent my-agent: hosting no tools is not hosting', $refused->getMessage());
        }
    }

    public function testDispatchNeedsAHandlerOrHostedTools(): void
    {
        $this->expectException(ConfigurationException::class);
        (new Dispatch(client: new Client(new Backend(url: 'http://127.0.0.1:1', customerId: 'examples'))))->run();
    }

    public function testWatchAnswersToolCallsAndYieldsEvents(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'heard', 'text' => 'weather in Paris?']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 't1', 'name' => 'get_weather', 'arguments' => '{"city":"Paris"}', 'command_id' => 'cmd1', 'turn_id' => 'turn1']);
            LocalSocketServer::send($socket, ['type' => 'tool_call', 'id' => 't2', 'name' => 'broken', 'arguments' => '{}']);
            while (count($server->sent('tool_result')) < 2 && $server->next($socket) !== null) {
            }
            LocalSocketServer::send($socket, ['type' => 'left']);
            $socket->close();
        });
        $agent = new Agent(name: 'jean', client: $server->client());
        $agent->tools
            ->register('get_weather', 'The weather', ['type' => 'object'], static fn (array $args): array => ['city' => $args['city'], 'sky' => 'clear'])
            ->register('broken', 'Always fails', [], static fn (array $args): string => throw new RuntimeException('the weather service is down'));
        $session = new Session($server->client(), SessionRow::fromArray(Rows::session()), $agent);

        $kinds = [];
        foreach ($session->watch() as $event) {
            $kinds[] = $event->kind;
        }

        self::assertSame(['heard', 'left'], $kinds);
        self::assertSame('/v1/agents/sessions/ses_1/events?interim=false&decisions=true', $server->handshakes[0]);
        $results = [];
        foreach ($server->sent('tool_result') as $result) {
            $results[Json::string($result, 'tool_call_id')] = $result;
        }
        self::assertSame(['type' => 'tool_result', 'tool_call_id' => 't1', 'command_id' => 'cmd1', 'turn_id' => 'turn1', 'output' => '{"city":"Paris","sky":"clear"}'], $results['t1']);
        self::assertSame(['type' => 'tool_result', 'tool_call_id' => 't2', 'error' => 'the weather service is down'], $results['t2']);
    }

    public function testRealtimeSendsTheStartFrameAndStopsAtClosed(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            $server->next($socket);
            LocalSocketServer::send($socket, ['type' => 'started']);
            $server->next($socket);
            LocalSocketServer::send($socket, ['type' => 'transcript', 'text' => 'hello', 'final' => true]);
            LocalSocketServer::send($socket, ['type' => 'closed']);
            delay(0.1);
            $socket->close();
        });
        $router = $server->client()->router('healthcare', ['team' => 'clinical']);

        $stt = $router->stt->realtime(new SttOptions(interim: true));
        $stt->sendAudio(str_repeat("\0", 320));
        $frames = iterator_to_array($stt->frames(), false);
        $stt->close();

        self::assertSame('/v1/stt/stream', $server->handshakes[0]);
        self::assertSame(['type' => 'start', 'config_id' => 'healthcare', 'tags' => ['team' => 'clinical'], 'stt' => ['interim' => true]], $server->frames[0]);
        self::assertSame(['binary' => 320], $server->frames[1]);
        self::assertSame(['started', 'transcript'], array_map(static fn (array|string $frame): string => is_array($frame) ? Json::string($frame, 'type') : 'audio', $frames));
    }

    public function testRealtimeRaisesAnErrorFrame(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            $server->next($socket);
            LocalSocketServer::send($socket, ['type' => 'error', 'error' => 'no model here takes images']);
            delay(0.1);
            $socket->close();
        });

        $llm = $server->client()->router()->llm->realtime();

        $this->expectException(RealtimeException::class);
        $this->expectExceptionMessage('no model here takes images');
        iterator_to_array($llm->frames());
    }

    public function testARefusedHandshakeSaysWhy(): void
    {
        $server = $this->serve(static fn (WebsocketClient $socket) => null);
        $server->refuse = 403;

        try {
            (new Dispatch(client: $server->client()))->waitForCall(static fn (InboundCall $call) => null)->run();
            self::fail('a refused handshake was taken as open');
        } catch (RouterException $refused) {
            self::assertSame(403, $refused->status);
            self::assertSame('GET /v1/dispatch', $refused->operation);
            self::assertSame('not for you', $refused->said);
            self::assertSame('permission', $refused->type);
            self::assertSame('permission', $refused->errorCode);
            self::assertSame('https://getstream.io/agents/docs/api/errors/#permission', $refused->docUrl);
            self::assertSame('req_socket', $refused->requestId);
        }
    }

    public function testServerSocketsSendCredentialsAsHeaders(): void
    {
        $server = $this->serve(static fn (WebsocketClient $socket) => $socket->close());
        $client = new Client(new Backend(url: $server->url, apiKey: 'key', apiSecret: 'secret'));

        (new Dispatch(client: $client))->waitForCall(static fn (InboundCall $call) => null)->run();

        self::assertSame('/v1/dispatch?capacity=4&active=0&handles=call', $server->handshakes[0]);
        self::assertSame('key', $server->headers[0]['x-api-key']);
        self::assertSame('server', $server->headers[0]['stream-auth-type']);
        self::assertStringStartsWith('Bearer ', $server->headers[0]['authorization']);
    }

    /**
     * @param \Closure(WebsocketClient, LocalSocketServer): void $script
     */
    private function serve(\Closure $script): LocalSocketServer
    {
        return $this->server = new LocalSocketServer($script);
    }
}
