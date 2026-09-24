<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GuzzleHttp\Client as Guzzle;
use GuzzleHttp\Psr7\HttpFactory;
use PHPUnit\Framework\TestCase;

final class ClientTest extends TestCase
{
    private LocalRouter $router;

    protected function setUp(): void
    {
        $this->router = new LocalRouter();
    }

    protected function tearDown(): void
    {
        $this->router->stop();
    }

    public function testCustomerHeaderAndUser(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);
        $client = new Client(new Backend(url: $this->router->url, customerId: 'examples', userId: 'ada'));

        $client->get('/v1/agents/configs');

        $sent = $this->router->received()[0];
        self::assertSame('examples', $sent->headers['x-customer-id']);
        self::assertSame('ada', $sent->headers['x-stream-user-id']);
        self::assertArrayNotHasKey('authorization', $sent->headers);
    }

    public function testSecretSignsAServerToken(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);
        $client = new Client(new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret'));

        $client->get('/v1/agents/configs');

        $sent = $this->router->received()[0];
        self::assertSame('key', $sent->headers['x-api-key']);
        self::assertSame('server', $sent->headers['stream-auth-type']);
        [$head, $payload, $signature] = explode('.', substr($sent->headers['authorization'], strlen('Bearer ')));
        $expected = rtrim(strtr(base64_encode(hash_hmac('sha256', "{$head}.{$payload}", 'secret', true)), '+/', '-_'), '=');
        self::assertSame($expected, $signature);
        self::assertTrue(Json::asObject(Json::decode((string) base64_decode(strtr($payload, '-_', '+/'), true)))['server']);
    }

    public function testATokenIsSentAsJwt(): void
    {
        $this->router->answer('GET', '/v1/agents/sessions/ses_1', 200, ['id' => 'ses_1']);
        $client = new Client(new Backend(url: $this->router->url, apiKey: 'key', token: 'tok', userId: 'ada'));

        $client->get('/v1/agents/sessions/{id}', ['id' => 'ses_1']);

        $sent = $this->router->received()[0];
        self::assertSame('/v1/agents/sessions/ses_1', $sent->path);
        self::assertSame('Bearer tok', $sent->headers['authorization']);
        self::assertSame('jwt', $sent->headers['stream-auth-type']);
    }

    public function testBehindTheProxyTheCredentialIsSpelledItsWay(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);
        $client = new Client(new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret', authenticate: true));

        $client->get('/v1/agents/configs');

        $sent = $this->router->received()[0];
        self::assertSame('key', $sent->headers['api_key']);
        self::assertSame('jwt', $sent->headers['stream-auth-type']);
        self::assertArrayNotHasKey('x-api-key', $sent->headers);
    }

    public function testPathAndQueryEncoding(): void
    {
        $this->router->answer('GET', '/v1/agents/sessions/a%2Fb', 200, []);
        $client = $this->router->client();

        $client->get('/v1/agents/sessions/{id}', ['id' => 'a/b'], ['limit' => 5, 'skip' => null, 'live' => true, 'q' => 'two words']);

        $sent = $this->router->received()[0];
        self::assertSame('limit=5&live=true&q=two%20words', $sent->query);
    }

    public function testFailureCarriesStatusAndWhatTheRouterSaid(): void
    {
        $this->router->answerWithHeaders('POST', '/v1/agents/sessions', 429, ['error' => 'slow down'], ['Retry-After' => '7']);

        try {
            $this->router->client()->post('/v1/agents/sessions', body: ['text' => true]);
            self::fail('a 429 was returned as success');
        } catch (RouterException $refused) {
            self::assertSame(429, $refused->status);
            self::assertSame('slow down', $refused->said);
            self::assertSame(7, $refused->retryAfter);
            self::assertSame('POST /v1/agents/sessions answered 429: slow down', $refused->getMessage());
        }
    }

    public function testNoContentIsNull(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/rewind', 204);

        self::assertNull($this->router->client()->post('/v1/agents/sessions/{id}/rewind', ['id' => 'ses_1'], body: ['response_id' => 'resp_1']));
    }

    public function testUnreachableIsStatusZero(): void
    {
        $client = new Client(new Backend(url: 'http://127.0.0.1:1', customerId: 'examples'));

        try {
            $client->get('/v1/agents/configs');
            self::fail('a closed port answered');
        } catch (RouterException $failed) {
            self::assertSame(0, $failed->status);
        }
    }

    public function testAnyPsr18ClientWorks(): void
    {
        $this->router->answer('POST', '/v1/agents/guests', 200, ['id' => 'guest_1', 'token' => 'tok']);
        $factory = new HttpFactory();
        $client = new Client(new Backend(url: $this->router->url, customerId: 'examples'), new Guzzle(), $factory, $factory);

        $guest = $client->guestUser(name: 'Ada');

        self::assertSame('guest_1', $guest->id);
        self::assertSame(['name' => 'Ada'], $this->router->received()[0]->json());
    }

    public function testClaimingAGuestIsServerSideOnly(): void
    {
        $client = new Client(new Backend(url: $this->router->url, apiKey: 'key', token: 'tok', userId: 'ada'));

        $this->expectException(ConfigurationException::class);
        $client->claimGuestUser('guest_1', 'ada');
    }

    public function testClaimsAGuest(): void
    {
        $this->router->answer('POST', '/v1/agents/guests/claim', 200, ['guest_id' => 'guest_1', 'user_id' => 'ada', 'sessions_moved' => 2]);

        $claimed = $this->router->client()->claimGuestUser('guest_1', 'ada');

        self::assertSame(2, $claimed->sessionsMoved);
        self::assertSame(['guest_id' => 'guest_1', 'user_id' => 'ada'], $this->router->received()[0]->json());
    }
}
