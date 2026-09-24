<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use Amp\Websocket\WebsocketClient;
use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\RealtimeException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SttOptions;
use GetStream\VisionAgents\Inbound\InboundCall;
use GetStream\VisionAgents\Inbound\InboundMessage;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Router;
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

    public function testDispatchAcceptsAndRejectsCalls(): void
    {
        $server = $this->serve(static function (WebsocketClient $socket, LocalSocketServer $server): void {
            LocalSocketServer::send($socket, ['type' => 'ready', 'worker_id' => 'w-1']);
            LocalSocketServer::send($socket, ['type' => 'call', 'call_id' => 'c1', 'caller_number' => '+15550001111', 'custom' => ['lang' => 'fr']]);
            LocalSocketServer::send($socket, ['type' => 'call', 'call_id' => 'c2', 'call_type' => 'phone']);
            while (count($server->sent('accepted')) + count($server->sent('rejected')) < 2 && $server->next($socket) !== null) {
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
        self::assertSame('/v1/dispatch?capacity=2', $server->handshakes[0]);
        self::assertSame('examples', $server->headers[0]['x-customer-id']);
        self::assertSame([['type' => 'accepted', 'call_id' => 'c1']], $server->sent('accepted'));
        self::assertSame([['type' => 'rejected', 'call_id' => 'c2', 'reason' => 'no agent for phone calls']], $server->sent('rejected'));
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
        $server = $this->serve(static function (WebsocketClient $socket): void {
            foreach (['ch1', 'ch1', 'ch2'] as $channel) {
                LocalSocketServer::send($socket, ['type' => 'message', 'channel_id' => $channel, 'text' => 'hi', 'agent_id' => 'jean']);
            }
            delay(0.5);
            $socket->close();
        });
        $sessions = [];
        $dispatch = new Dispatch(client: $server->client());
        $dispatch->waitForMessage(static function (InboundMessage $message) use ($dispatch, $http, &$sessions): void {
            $session = $dispatch->getOrCreateAgent($message, static fn (): Agent => new Agent(name: 'jean', client: $http));
            $sessions[] = $message->channelId . '=' . $session->id();
        });

        $dispatch->run();

        sort($sessions);
        self::assertCount(2, $this->router->to('POST', '/v1/agents/sessions'));
        self::assertSame(['ch1=ses_1', 'ch1=ses_1', 'ch2=ses_2'], $sessions);
        self::assertSame([], $server->sent('accepted'), 'a message is not a call to accept');
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
        $router = new Router('healthcare', ['team' => 'clinical'], $server->client());

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

        $llm = (new Router(client: $server->client()))->llm->realtime();

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
            self::assertSame('not for you', $refused->said);
        }
    }

    public function testServerSocketsSendCredentialsAsHeaders(): void
    {
        $server = $this->serve(static fn (WebsocketClient $socket) => $socket->close());
        $client = new Client(new Backend(url: $server->url, apiKey: 'key', apiSecret: 'secret'));

        (new Dispatch(client: $client))->waitForCall(static fn (InboundCall $call) => null)->run();

        self::assertSame('/v1/dispatch?capacity=4', $server->handshakes[0]);
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
