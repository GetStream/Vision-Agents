<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use Nyholm\Psr7\Response;
use PHPUnit\Framework\TestCase;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestInterface;
use Psr\Http\Message\ResponseInterface;

final class RegionTest extends TestCase
{
    private const string REGIONAL_HOST = 'accelerate-us-east4.gcp.stream-io-api.com';

    /** @var array<string, array{string|false, mixed, mixed}> */
    private array $saved = [];

    /** @var list<RequestInterface> */
    private array $sent = [];

    private ClientInterface $http;

    protected function setUp(): void
    {
        foreach ([Backend::URL_ENV, Backend::REGION_ENV, Backend::CUSTOMER_ENV, Backend::AUTHENTICATE_ENV] as $name) {
            $this->setEnv($name, null);
        }
        $this->setEnv(Backend::API_KEY_ENV, 'test-key');
        $this->setEnv(Backend::API_SECRET_ENV, 'test-secret');

        $this->http = new class (function (RequestInterface $request): void {
            $this->sent[] = $request;
        }) implements ClientInterface {
            /** @param \Closure(RequestInterface): void $record */
            public function __construct(private \Closure $record)
            {
            }

            public function sendRequest(RequestInterface $request): ResponseInterface
            {
                ($this->record)($request);
                if ($request->getMethod() !== 'GET' || $request->getUri()->getPath() !== '/v1/agents/simulations') {
                    return new Response(404, [], '{"error":"not found"}');
                }
                if ($request->getUri()->getHost() !== 'accelerate-us-east4.gcp.stream-io-api.com'
                    || $request->getHeaderLine('api_key') !== 'test-key'
                    || $request->getHeaderLine('stream-auth-type') !== 'jwt'
                    || $request->getHeaderLine('Authorization') === '') {
                    return new Response(401, [], '{"error":"expected an authenticated regional request"}');
                }
                return new Response(200, ['Content-Type' => 'application/json'], '[]');
            }
        };
    }

    protected function tearDown(): void
    {
        foreach ($this->saved as $name => [$env, $dollarEnv, $server]) {
            putenv($env === false ? $name : "{$name}={$env}");
            if ($dollarEnv === null) {
                unset($_ENV[$name]);
            } else {
                $_ENV[$name] = $dollarEnv;
            }
            if ($server === null) {
                unset($_SERVER[$name]);
            } else {
                $_SERVER[$name] = $server;
            }
        }
        $this->saved = [];
    }

    public function testTheDefaultClientReadsAResource(): void
    {
        $this->readSimulations(new Backend());
    }

    public function testAnExplicitRegionReadsAResource(): void
    {
        $this->readSimulations(new Backend(region: 'us-east4'));
    }

    public function testARegionFromTheEnvironmentReadsAResource(): void
    {
        $this->setEnv(Backend::REGION_ENV, 'us-east4');
        $this->readSimulations(new Backend());
    }

    public function testAConflictingUrlAndRegionAreRefused(): void
    {
        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('pass a url or a region, not both');
        new Client(new Backend(url: Backend::DEFAULT_URL, region: 'europe-west4'), $this->http);
    }

    public function testAnInvalidRegionWithAUrlIsRefused(): void
    {
        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('pass a url or a region, not both');
        new Client(new Backend(url: Backend::DEFAULT_URL, region: 'us-east4.evil.com/x'), $this->http);
    }

    public function testARegionalClientCanActForAUser(): void
    {
        $client = new Client(new Backend(region: 'us-east4'), $this->http);

        $user = $client->asUser('jean', 'user-token');

        self::assertSame(Backend::DEFAULT_URL, $user->backend->url);
        self::assertSame('jean', $user->backend->userId);
        self::assertSame('Bearer user-token', $user->backend->headers()['Authorization']);
        $this->readSimulations($user->backend, $user);
        self::assertSame('Bearer user-token', $this->sent[0]->getHeaderLine('Authorization'));
    }

    private function readSimulations(Backend $backend, ?Client $client = null): void
    {
        $client ??= new Client($backend, $this->http);

        self::assertSame([], $client->simulations->list());

        self::assertCount(1, $this->sent);
        $request = $this->sent[0];
        self::assertSame('https', $request->getUri()->getScheme());
        self::assertSame(self::REGIONAL_HOST, $request->getUri()->getHost());
        self::assertSame('test-key', $request->getHeaderLine('api_key'));
        self::assertSame('jwt', $request->getHeaderLine('stream-auth-type'));
        self::assertNotSame('', $request->getHeaderLine('Authorization'));
    }

    /**
     * Sets or, with null, unsets a variable everywhere `Backend::env` looks, remembering what
     * was there first.
     */
    private function setEnv(string $name, ?string $value): void
    {
        if (!array_key_exists($name, $this->saved)) {
            $this->saved[$name] = [getenv($name), $_ENV[$name] ?? null, $_SERVER[$name] ?? null];
        }
        if ($value === null) {
            putenv($name);
            unset($_ENV[$name], $_SERVER[$name]);
            return;
        }
        putenv("{$name}={$value}");
        $_ENV[$name] = $value;
        $_SERVER[$name] = $value;
    }
}
