<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\SimulationRequest;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Rows;
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

    public function testBehindTheProxyActingForSignsTheUsersTokenAndOnBehalfOfKeepsTheServers(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);
        $backend = new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret', authenticate: true);

        (new Client($backend->actingFor('ada')))->get('/v1/agents/configs');
        (new Client($backend->onBehalfOf('ada')))->get('/v1/agents/configs');

        [$acting, $behalf] = $this->router->received();
        self::assertSame('ada', self::claims($acting->headers['authorization'])['user_id']);
        self::assertArrayNotHasKey('x-stream-user-id', $acting->headers);
        self::assertTrue(self::claims($behalf->headers['authorization'])['server']);
        self::assertArrayNotHasKey('user_id', self::claims($behalf->headers['authorization']));
        self::assertSame('ada', $behalf->headers['x-stream-user-id']);
    }

    public function testOnBehalfOfKeepsTheCustomerHeader(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);

        (new Client(new Backend(url: $this->router->url, customerId: 'examples')->onBehalfOf('ada')))->get('/v1/agents/configs');

        $sent = $this->router->received()[0];
        self::assertSame('examples', $sent->headers['x-customer-id']);
        self::assertSame('ada', $sent->headers['x-stream-user-id']);
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

    public function testTruncatesEverythingRememberedAboutAUser(): void
    {
        $this->router->answer('DELETE', '/v1/agents/users/u%201/memories', 204);

        $this->router->client()->memories->truncate('u 1');

        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/users/u%201/memories'));
    }

    public function testTruncatingNobodyIsRefusedBeforeTheRouter(): void
    {
        try {
            $this->router->client()->memories->truncate('');
            self::fail('an empty user id was sent');
        } catch (ConfigurationException) {
            self::assertSame([], $this->router->received());
        }
    }

    public function testSimulations(): void
    {
        $simulation = ['id' => 'sim_1', 'name' => 'lunch', 'config_id' => 'cfg_1', 'scenario' => 'Order a club.', 'assertion' => 'One club.', 'created_at' => Rows::AT, 'updated_at' => Rows::AT];
        $run = ['id' => 'run_1', 'simulation_id' => 'sim_1', 'state' => 'running', 'created_at' => Rows::AT, 'updated_at' => Rows::AT];
        $this->router->answer('POST', '/v1/agents/simulations', 201, $simulation);
        $this->router->answer('PUT', '/v1/agents/simulations/sim_1', 200, ['scenario' => 'Order two.'] + $simulation);
        $this->router->answer('GET', '/v1/agents/simulations', 200, [$simulation]);
        $this->router->answer('POST', '/v1/agents/simulations/sim_1/run', 202, $run);
        $this->router->answer('GET', '/v1/agents/simulation-runs/run_1', 200, ['state' => 'passed'] + $run);
        $this->router->answer('GET', '/v1/agents/simulation-runs', 200, [$run]);
        $this->router->answer('POST', '/v1/agents/simulation-runs/run_1/cancel', 200, ['state' => 'cancelled'] + $run);
        $this->router->answer('DELETE', '/v1/agents/simulations/sim_1', 204);
        $simulations = $this->router->client()->simulations;
        $request = new SimulationRequest(assertion: 'One club.', configId: 'cfg_1', name: 'lunch', scenario: 'Order a club.', variations: 2);

        $created = $simulations->create($request);
        $updated = $simulations->update($created->id, new SimulationRequest(assertion: 'One club.', configId: 'cfg_1', name: 'lunch', scenario: 'Order two.'));
        $listed = $simulations->list();
        $started = $simulations->run($created->id);
        $finished = $simulations->runs->get($started->id);
        $runs = $simulations->runs->list(simulationId: 'sim_1', limit: 5);
        $cancelled = $simulations->runs->cancel($started->id);
        $simulations->delete($created->id);

        self::assertSame('sim_1', $created->id);
        self::assertSame(
            ['assertion' => 'One club.', 'config_id' => 'cfg_1', 'name' => 'lunch', 'scenario' => 'Order a club.', 'variations' => 2],
            $this->router->to('POST', '/v1/agents/simulations')[0]->json(),
        );
        self::assertSame('Order two.', $updated->scenario);
        self::assertSame(['sim_1'], array_map(static fn ($each) => $each->id, $listed));
        self::assertSame('passed', $finished->state);
        self::assertSame(['simulation_id' => 'sim_1', 'limit' => '5'], $this->router->to('GET', '/v1/agents/simulation-runs')[0]->params());
        self::assertSame(['run_1'], array_map(static fn ($each) => $each->id, $runs));
        self::assertSame('cancelled', $cancelled->state);
        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/simulations/sim_1'));
    }

    /**
     * @return array<string, mixed>
     */
    private static function claims(string $authorization): array
    {
        $payload = explode('.', substr($authorization, strlen('Bearer ')))[1];
        return Json::asObject(Json::decode((string) base64_decode(strtr($payload, '-_', '+/'), true)));
    }
}
