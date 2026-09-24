<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Worker;

use Amp\ByteStream\StreamException;
use Amp\Cancellation;
use Amp\CancelledException;
use Amp\Http\Client\Connection\DefaultConnectionFactory;
use Amp\Http\Client\Connection\UnlimitedConnectionPool;
use Amp\Http\Client\HttpClientBuilder;
use Amp\Http\Client\HttpException;
use Amp\Socket\ConnectContext;
use Amp\Socket\DnsSocketConnector;
use Amp\TimeoutCancellation;
use Amp\Websocket\Client\Rfc6455Connector;
use Amp\Websocket\Client\WebsocketConnectException;
use Amp\Websocket\Client\WebsocketConnection;
use Amp\Websocket\Client\WebsocketHandshake;
use Amp\Websocket\WebsocketClosedException;
use Amp\Websocket\WebsocketCloseCode;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Json;
use JsonException;

/**
 * One of the router's websockets, as JSON frames in and out.
 *
 * It never reconnects. What a dropped socket means differs per socket: a session that lost its
 * events has lost them, and a dispatch worker that lost its socket has been taken out of the
 * pool. Each owner decides.
 *
 * @internal
 */
final class Socket
{
    private static ?Rfc6455Connector $connector = null;
    private bool $closed = false;

    private function __construct(private readonly WebsocketConnection $connection)
    {
    }

    /**
     * @param array<string, scalar|null> $query
     */
    public static function open(Client $client, string $path, array $query = [], float $timeout = 10.0): self
    {
        if (!interface_exists(WebsocketConnection::class)) {
            throw new ConfigurationException('a socket needs amphp/websocket-client; composer require amphp/websocket-client');
        }
        $backend = $client->backend;
        $url = $backend->socketUrl($path, array_map(self::queryValue(...), $query));
        // Credentials go in headers. A server-side token in a query string ends up in access logs.
        $handshake = new WebsocketHandshake($url, self::named($backend->headers()));
        $operation = "GET {$path}";
        try {
            return new self(self::connector()->connect($handshake, new TimeoutCancellation($timeout)));
        } catch (WebsocketConnectException $refused) {
            $response = $refused->getResponse();
            $body = '';
            try {
                $body = $response->getBody()->buffer();
            } catch (HttpException) {
            }
            $said = $body;
            try {
                $error = Json::string(Json::asObject(Json::decode($body)), 'error');
                $said = $error === '' ? $body : $error;
            } catch (JsonException) {
            }
            throw new RouterException($response->getStatus(), $operation, $said === '' ? $response->getReason() : $said, $body, 0, $refused);
        } catch (HttpException | CancelledException $failed) {
            throw new RouterException(0, $operation, $failed->getMessage(), '', 0, $failed);
        }
    }

    /**
     * @param array<string, mixed> $frame
     */
    public function send(array $frame): void
    {
        $this->connection->sendText(Json::encode($frame));
    }

    public function sendBinary(string $bytes): void
    {
        $this->connection->sendBinary($bytes);
    }

    /**
     * The next frame: a decoded object for text, the bytes for binary, null once the socket
     * has closed. Text that is not a JSON object is skipped rather than failing the socket,
     * since a frame this version does not understand is not a reason to hang up.
     *
     * @return array<string, mixed>|string|null
     */
    public function receive(?Cancellation $cancellation = null): array|string|null
    {
        while (true) {
            try {
                $message = $this->connection->receive($cancellation);
                if ($message === null) {
                    return null;
                }
                $payload = $message->buffer($cancellation);
            } catch (WebsocketClosedException | StreamException) {
                return null;
            }
            if ($message->isBinary()) {
                return $payload;
            }
            try {
                $frame = Json::decode($payload);
            } catch (JsonException) {
                continue;
            }
            if (is_array($frame) && !array_is_list($frame)) {
                return Json::asObject($frame);
            }
        }
    }

    /**
     * Safe to call more than once, and on a socket the router already closed.
     */
    public function close(int $code = WebsocketCloseCode::NORMAL_CLOSE, string $reason = ''): void
    {
        if ($this->closed) {
            return;
        }
        $this->closed = true;
        $this->connection->close($code, $reason);
    }

    public function closed(): bool
    {
        return $this->closed || $this->connection->isClosed();
    }

    /**
     * Why the router closed it, once it has.
     */
    public function closeCode(): int
    {
        return $this->connection->isClosed() ? $this->connection->getCloseInfo()->getCode() : 0;
    }

    /**
     * Amp's default connector retries a refused connection with a backoff, which made a router
     * that is down take eighteen seconds to say so.
     */
    private static function connector(): Rfc6455Connector
    {
        return self::$connector ??= new Rfc6455Connector(httpClient: (new HttpClientBuilder())
            ->usingPool(new UnlimitedConnectionPool(new DefaultConnectionFactory(new DnsSocketConnector(), (new ConnectContext())->withTcpNoDelay())))
            ->build());
    }

    private static function queryValue(bool|float|int|string|null $value): string
    {
        return match (true) {
            $value === true => 'true',
            $value === false => 'false',
            default => (string) $value,
        };
    }

    /**
     * @param array<string, string> $headers
     * @return array<non-empty-string, string>
     */
    private static function named(array $headers): array
    {
        $named = [];
        foreach ($headers as $name => $value) {
            if ($name !== '') {
                $named[$name] = $value;
            }
        }
        return $named;
    }
}
