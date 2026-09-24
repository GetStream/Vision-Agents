<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Support;

use Amp\Http\Server\DefaultErrorHandler;
use Amp\Http\Server\Request;
use Amp\Http\Server\Response;
use Amp\Http\Server\SocketHttpServer;
use Amp\Socket\InternetAddress;
use Amp\Websocket\Server\Rfc6455Acceptor;
use Amp\Websocket\Server\Websocket;
use Amp\Websocket\Server\WebsocketAcceptor;
use Amp\Websocket\Server\WebsocketClientHandler;
use Amp\Websocket\WebsocketClient;
use Closure;
use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Json;
use Psr\Log\NullLogger;

/**
 * A real websocket server on this process's event loop, playing the router's side of a socket.
 *
 * The script is handed each connection and talks to it the way the router would; everything the
 * client sent is kept, so a test asserts on the frames rather than on what was called.
 */
final class LocalSocketServer implements WebsocketClientHandler, WebsocketAcceptor
{
    public readonly string $url;
    /** @var list<array<string, mixed>> text frames the client sent, decoded */
    public array $frames = [];
    /** @var list<string> the path and query each handshake asked for */
    public array $handshakes = [];
    /** @var list<array<string, string>> the headers each handshake carried, lower-cased */
    public array $headers = [];
    /** A status to refuse the handshake with, or 0 to accept it. */
    public int $refuse = 0;
    private readonly SocketHttpServer $server;
    private readonly Rfc6455Acceptor $acceptor;

    /**
     * @param Closure(WebsocketClient, self): void $script
     */
    public function __construct(private readonly Closure $script)
    {
        $logger = new NullLogger();
        $this->acceptor = new Rfc6455Acceptor();
        $this->server = SocketHttpServer::createForDirectAccess($logger);
        $this->server->expose(new InternetAddress('127.0.0.1', 0));
        $this->server->start(new Websocket($this->server, $logger, $this, $this), new DefaultErrorHandler());
        $address = $this->server->getServers()[0]->getAddress();
        $this->url = 'http://' . $address->toString();
    }

    public function client(): Client
    {
        return new Client(new Backend(url: $this->url, customerId: 'examples'));
    }

    public function handleHandshake(Request $request): Response
    {
        $uri = $request->getUri();
        $this->handshakes[] = $uri->getPath() . ($uri->getQuery() === '' ? '' : '?' . $uri->getQuery());
        $headers = [];
        foreach ($request->getHeaders() as $name => $values) {
            $headers[strtolower($name)] = implode(', ', $values);
        }
        $this->headers[] = $headers;
        if ($this->refuse !== 0) {
            return new Response($this->refuse, ['content-type' => 'application/json'], Json::encode(['error' => 'not for you']));
        }
        return $this->acceptor->handleHandshake($request);
    }

    public function handleClient(WebsocketClient $client, Request $request, Response $response): void
    {
        ($this->script)($client, $this);
    }

    /**
     * The next frame from the client, kept: text decoded, binary as its length.
     *
     * @return array<string, mixed>|null null once the client has closed
     */
    public function next(WebsocketClient $client): ?array
    {
        while (($message = $client->receive()) !== null) {
            $payload = $message->buffer();
            $frame = $message->isBinary() ? ['binary' => strlen($payload)] : Json::asObject(Json::decode($payload));
            $this->frames[] = $frame;
            return $frame;
        }
        return null;
    }

    /**
     * @param array<string, mixed> $frame
     */
    public static function send(WebsocketClient $client, array $frame): void
    {
        $client->sendText(Json::encode($frame));
    }

    /**
     * The frames the client sent of one type.
     *
     * @return list<array<string, mixed>>
     */
    public function sent(string $type): array
    {
        return array_values(array_filter($this->frames, static fn (array $frame): bool => ($frame['type'] ?? null) === $type));
    }

    public function stop(): void
    {
        $this->server->stop();
    }
}
