<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use Amp\Http\Client\Connection\DefaultConnectionFactory;
use Amp\Http\Client\Connection\UnlimitedConnectionPool;
use Amp\Http\Client\HttpClientBuilder;
use Amp\Http\Client\Interceptor\SetRequestTimeout;
use Amp\Http\Client\Psr7\PsrAdapter;
use Amp\Http\Client\Psr7\PsrHttpClient;
use Amp\Socket\DnsSocketConnector;
use GetStream\VisionAgents\Exception\ConfigurationException;
use Http\Discovery\Exception\NotFoundException;
use Http\Discovery\Psr17FactoryDiscovery;
use Http\Discovery\Psr18ClientDiscovery;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestFactoryInterface;
use Psr\Http\Message\StreamFactoryInterface;
use WeakMap;

/**
 * Finding a PSR-18 client and the PSR-17 factories that go with it.
 *
 * Amp's client is preferred when it is installed: it works from plain code, and inside a worker
 * a request suspends its fiber rather than stalling every other call the process holds.
 *
 * @internal
 */
final class Http
{
    /** @var ?WeakMap<ClientInterface, ClientInterface> each Amp client `client()` built, to the same one without its retries */
    private static ?WeakMap $sentOnce = null;

    public static function client(): ClientInterface
    {
        return self::found(static function (): ClientInterface {
            if (class_exists(PsrHttpClient::class)) {
                // Amp gives a request ten seconds. Opening a session returns once the agent is in
                // the call, which takes longer than that when a provider is slow to connect.
                // Its default connector also retries a refused connection with a six second
                // backoff, so a router that is down took eighteen seconds to say so.
                $amp = (new HttpClientBuilder())
                    ->usingPool(new UnlimitedConnectionPool(new DefaultConnectionFactory(new DnsSocketConnector())))
                    ->intercept(new SetRequestTimeout(10, 10, 120, 120));
                $adapter = new PsrAdapter(Psr17FactoryDiscovery::findRequestFactory(), Psr17FactoryDiscovery::findResponseFactory());
                $client = new PsrHttpClient($amp->build(), $adapter);
                self::$sentOnce ??= new WeakMap();
                self::$sentOnce[$client] = new PsrHttpClient($amp->retry(0)->build(), $adapter);
                return $client;
            }
            return Psr18ClientDiscovery::find();
        });
    }

    /**
     * The client to send a request with that must not be sent twice. Amp's sends a GET, PUT or
     * DELETE again when the connection drops before the answer, and the second answer is read
     * as the first's; this is the same client, on the same connections, without that. A client
     * the app passed in is used as it is.
     */
    public static function sentOnce(ClientInterface $http): ClientInterface
    {
        return self::$sentOnce[$http] ?? $http;
    }

    public static function requests(): RequestFactoryInterface
    {
        return self::found(Psr17FactoryDiscovery::findRequestFactory(...));
    }

    public static function streams(): StreamFactoryInterface
    {
        return self::found(Psr17FactoryDiscovery::findStreamFactory(...));
    }

    /**
     * @template T
     * @param callable(): T $find
     * @return T
     */
    private static function found(callable $find): mixed
    {
        try {
            return $find();
        } catch (NotFoundException $missing) {
            throw new ConfigurationException('there is no PSR-18 client or PSR-17 factory installed; composer require guzzlehttp/guzzle, or pass them in', 0, $missing);
        }
    }
}
