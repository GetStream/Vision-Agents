<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\AuthorizeOptions;
use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\ConnectionFilter;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\AuthorizationKind;
use GetStream\VisionAgents\Generated\ConnectionOwnerType;
use GetStream\VisionAgents\Generated\ConnectionStatus;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\NewConnection;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Received;
use GetStream\VisionAgents\Tests\Support\Rows;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class ConnectionsTest extends TestCase
{
    private LocalRouter $router;
    private Client $client;

    protected function setUp(): void
    {
        $this->router = new LocalRouter();
        $this->client = new Client(new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret'));
    }

    protected function tearDown(): void
    {
        $this->router->stop();
    }

    // Catches a user's connection whose owner and X-Stream-User-Id differ, or that names no user
    // in one of them: the router refuses either.
    public function testAUsersConnectionNamesThemAsOwnerAndActingUser(): void
    {
        $this->router->answer('POST', '/v1/agents/connections', 201, self::usersConnection('con_1', 'anya'));

        $created = $this->client->connections->create(new NewConnection('linear', 'work', 'anya'));

        $sent = $this->router->to('POST', '/v1/agents/connections')[0];
        self::assertSame(['connector_id' => 'linear', 'owner' => ['type' => 'user', 'user_id' => 'anya'], 'label' => 'work'], $sent->json());
        self::assertSame('anya', $sent->headers['x-stream-user-id']);
        self::assertSame('con_1', $created->id);
        self::assertSame('anya', $created->owner->userId);
        self::assertSame(ConnectionStatus::Pending, $created->status);
    }

    /**
     * @return iterable<string, array{?string}>
     */
    public static function noUser(): iterable
    {
        yield 'left out' => [null];
        yield 'empty' => [''];
    }

    // Catches an app connection sent with an owner.user_id, or with the user the client otherwise
    // acts for in X-Stream-User-Id, which would make each call depend on who the client was built for.
    #[DataProvider('noUser')]
    public function testAnAppConnectionNamesNoUser(?string $userId): void
    {
        $this->router->answer('POST', '/v1/agents/connections', 201, Rows::connection('con_1', 'github'));
        $this->router->answer('GET', '/v1/agents/connections/con_1', 200, Rows::connection('con_1', 'github'));
        $this->router->answer('POST', '/v1/agents/connections/con_1/authorizations', 201, self::authorization());
        $this->router->answer('GET', '/v1/agents/connections', 200, ['items' => [], 'has_more' => false]);
        $this->router->answer('DELETE', '/v1/agents/connections/con_1', 204);
        $backend = new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret');

        foreach ([$backend->actingFor('bob'), $backend->onBehalfOf('bob')] as $acting) {
            $connections = new Client($acting)->connections;
            $connections->create(new NewConnection('github', userId: $userId));
            $connections->get('con_1', $userId);
            $connections->authorize('con_1', new AuthorizeOptions(userId: $userId));
            $connections->list(new ConnectionFilter($userId));
            $connections->delete('con_1', $userId);
        }

        self::assertSame(['connector_id' => 'github', 'owner' => ['type' => 'app']], $this->router->to('POST', '/v1/agents/connections')[0]->json());
        self::assertSame(['owner_type' => 'app', 'limit' => '200'], $this->router->to('GET', '/v1/agents/connections')[0]->params());
        self::assertCount(10, $this->router->received());
        foreach ($this->router->received() as $sent) {
            self::assertArrayNotHasKey('x-stream-user-id', $sent->headers, "{$sent->method} {$sent->path}");
            self::assertSame('server', $sent->headers['stream-auth-type']);
        }
    }

    // Catches a read, an authorization, a list or a delete that drops the user: the router would
    // then look for the app's connection and answer that the user's does not exist.
    public function testEveryCallNamingAUserSendsItReadsIncluded(): void
    {
        $this->router->answer('GET', '/v1/agents/connections/con_1', 200, self::usersConnection('con_1', 'anya'));
        $this->router->answer('POST', '/v1/agents/connections/con_1/authorizations', 201, self::authorization());
        $this->router->answer('GET', '/v1/agents/connections', 200, ['items' => [self::usersConnection('con_1', 'anya')], 'has_more' => false]);
        $this->router->answer('DELETE', '/v1/agents/connections/con_1', 204);
        $connections = new Client(new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret')->actingFor('bob'))->connections;

        $connections->get('con_1', 'anya');
        $connections->authorize('con_1', new AuthorizeOptions(userId: 'anya'));
        $listed = $connections->list(new ConnectionFilter('anya'));
        $connections->delete('con_1', 'anya');

        self::assertSame(
            ['GET /v1/agents/connections/con_1', 'POST /v1/agents/connections/con_1/authorizations', 'GET /v1/agents/connections', 'DELETE /v1/agents/connections/con_1'],
            array_map(static fn (Received $sent): string => "{$sent->method} {$sent->path}", $this->router->received()),
        );
        foreach ($this->router->received() as $sent) {
            self::assertSame('anya', $sent->headers['x-stream-user-id'], "{$sent->method} {$sent->path}");
        }
        self::assertSame(['owner_type' => 'user', 'limit' => '200'], $this->router->to('GET', '/v1/agents/connections')[0]->params());
        self::assertSame(['con_1'], array_map(static fn ($c): string => $c->id, $listed));
    }

    // Catches a user named behind the proxy by signing a token for them: the router reads a user
    // token as a device, which is refused every connection call.
    public function testBehindTheProxyTheUserIsAHeaderBesideTheServersToken(): void
    {
        $this->router->answer('GET', '/v1/agents/connections/con_1', 200, self::usersConnection('con_1', 'anya'));
        $backend = new Backend(url: $this->router->url, apiKey: 'key', apiSecret: 'secret', authenticate: true);

        new Client($backend)->connections->get('con_1', 'anya');
        new Client($backend->actingFor('bob'))->connections->get('con_1');

        [$named, $none] = $this->router->received();
        self::assertSame('anya', $named->headers['x-stream-user-id']);
        self::assertArrayNotHasKey('x-stream-user-id', $none->headers);
        foreach ([$named, $none] as $sent) {
            self::assertSame('jwt', $sent->headers['stream-auth-type']);
            $claims = self::claims($sent->headers['authorization']);
            self::assertTrue($claims['server']);
            self::assertArrayNotHasKey('user_id', $claims);
        }
    }

    // Catches the customer header lost when a user is named on a router with nothing in front of it.
    public function testWithNoProxyTheCustomerHeaderStaysBesideTheUser(): void
    {
        $this->router->answer('GET', '/v1/agents/connections/con_1', 200, self::usersConnection('con_1', 'anya'));

        $this->router->client()->connections->get('con_1', 'anya');

        $sent = $this->router->received()[0];
        self::assertSame('examples', $sent->headers['x-customer-id']);
        self::assertSame('anya', $sent->headers['x-stream-user-id']);
    }

    // Catches an authorization that drops the return URL, sends an empty one, or comes back without
    // the launch URL, the handoff token or when the attempt ends.
    public function testAuthorizingAnswersWhereAndHowLongTheConsentRuns(): void
    {
        $this->router->answer('POST', '/v1/agents/connections/con_1/authorizations', 201, self::authorization());

        $started = $this->client->connections->authorize('con_1', new AuthorizeOptions('https://app.example.com/settings', 'anya'));
        $this->client->connections->authorize('con_1', new AuthorizeOptions('', 'anya'));

        [$with, $without] = $this->router->to('POST', '/v1/agents/connections/con_1/authorizations');
        self::assertSame(['return_url' => 'https://app.example.com/settings'], $with->json());
        self::assertSame('{}', $without->body);
        self::assertSame('att_1', $started->id);
        self::assertSame(AuthorizationKind::Consent, $started->kind);
        self::assertSame('https://router.example.com/v1/agents/connections/con_1/launch', $started->launchUrl);
        self::assertSame('handoff-1', $started->handoffToken);
        self::assertSame('2026-09-24T10:10:00+00:00', $started->expiresAt->format(DATE_ATOM));
    }

    // Catches a call whose adjacent optional strings take a user id in the label, return URL or
    // connector slot, which files a user's account as the app's: a string is not an options
    // object, so none of these reaches the router.
    public function testAStringWhereTheOptionsObjectGoesIsRefusedBeforeSending(): void
    {
        $connections = $this->client->connections;
        foreach ([
            [[$connections, 'create'], ['linear', 'ada']],
            [[$connections, 'authorize'], ['con_1', 'ada']],
            [[$connections, 'list'], ['ada']],
        ] as [$call, $arguments]) {
            try {
                $call(...$arguments);
                self::fail('a string was taken for an options object');
            } catch (\TypeError) {
            }
        }

        self::assertSame([], $this->router->received());
    }

    // Catches a list that reads one page only, drops the connector or the cursor on a later page,
    // or follows a next cursor on a page that says there is no more.
    public function testListingReadsEveryPageOfOneConnector(): void
    {
        $this->router->answer('GET', '/v1/agents/connections', 200,
            ['items' => [self::usersConnection('con_3', 'anya')], 'has_more' => true, 'next_cursor' => 'after-1'],
            ['items' => [self::usersConnection('con_2', 'anya')], 'has_more' => true, 'next_cursor' => 'after-2'],
            ['items' => [self::usersConnection('con_1', 'anya')], 'has_more' => false, 'next_cursor' => 'after-3'],
        );

        $listed = $this->client->connections->list(new ConnectionFilter('anya', 'linear'));

        self::assertSame(['con_3', 'con_2', 'con_1'], array_map(static fn ($c): string => $c->id, $listed));
        self::assertSame([
            ['owner_type' => 'user', 'connector_id' => 'linear', 'limit' => '200'],
            ['owner_type' => 'user', 'connector_id' => 'linear', 'limit' => '200', 'cursor' => 'after-1'],
            ['owner_type' => 'user', 'connector_id' => 'linear', 'limit' => '200', 'cursor' => 'after-2'],
        ], array_map(static fn (Received $sent): array => $sent->params(), $this->router->received()));
    }

    // Catches a list by connector that reads the connector as the user (the filter fields swapped):
    // an app's list would then go out as a user's, with a header naming the connector.
    public function testAnAppListByConnectorSendsNoUser(): void
    {
        $this->router->answer('GET', '/v1/agents/connections', 200, ['items' => [], 'has_more' => false]);

        $this->client->connections->list(new ConnectionFilter(connectorId: 'linear'));

        $sent = $this->router->to('GET', '/v1/agents/connections')[0];
        self::assertArrayNotHasKey('x-stream-user-id', $sent->headers);
        self::assertSame(['owner_type' => 'app', 'connector_id' => 'linear', 'limit' => '200'], $sent->params());
    }

    // Catches an authorization that reads the return URL as the user: the URL would go out as
    // X-Stream-User-Id.
    public function testAnAppAuthorizationWithAReturnUrlSendsNoUser(): void
    {
        $this->router->answer('POST', '/v1/agents/connections/con_1/authorizations', 201, self::authorization());

        $this->client->connections->authorize('con_1', new AuthorizeOptions('https://app.example.com/settings'));

        $sent = $this->router->to('POST', '/v1/agents/connections/con_1/authorizations')[0];
        self::assertArrayNotHasKey('x-stream-user-id', $sent->headers);
        self::assertSame(['return_url' => 'https://app.example.com/settings'], $sent->json());
    }

    // Catches a create that reads the label as the user: an app's connection would be filed as
    // the account of a user named like its label.
    public function testAnAppConnectionWithALabelIsOwnedByTheApp(): void
    {
        $this->router->answer('POST', '/v1/agents/connections', 201, Rows::connection('con_1', 'linear'));

        $this->client->connections->create(new NewConnection('linear', 'work'));

        $sent = $this->router->to('POST', '/v1/agents/connections')[0];
        self::assertArrayNotHasKey('x-stream-user-id', $sent->headers);
        self::assertSame(['connector_id' => 'linear', 'owner' => ['type' => 'app'], 'label' => 'work'], $sent->json());
    }

    // Catches a list that pages forever when the router hands back a cursor it answered before,
    // or ends on a page that has more but an empty cursor without keeping that page.
    public function testListingStopsOnACursorItAlreadyFollowedOrAnEmptyOne(): void
    {
        $this->router->answer('GET', '/v1/agents/connections', 200,
            ['items' => [Rows::connection('con_2', 'linear')], 'has_more' => true, 'next_cursor' => 'again'],
            ['items' => [Rows::connection('con_1', 'linear')], 'has_more' => true, 'next_cursor' => 'again'],
        );

        $listed = $this->client->connections->list();

        self::assertSame(['con_2', 'con_1'], array_map(static fn ($c): string => $c->id, $listed));
        self::assertCount(2, $this->router->received());

        $other = new LocalRouter();
        try {
            $other->answer('GET', '/v1/agents/connections', 200, ['items' => [Rows::connection('con_1', 'linear')], 'has_more' => true, 'next_cursor' => '']);
            $only = new Client(new Backend(url: $other->url, apiKey: 'key', apiSecret: 'secret'))->connections->list();
            self::assertSame(['con_1'], array_map(static fn ($c): string => $c->id, $only));
            self::assertCount(1, $other->received());
        } finally {
            $other->stop();
        }
    }

    // Catches the router's refusal turned into anything but its error, such as another user's
    // connection read as an empty one.
    public function testAnotherUsersConnectionIsTheRoutersNotFound(): void
    {
        $this->router->answer('GET', '/v1/agents/connections/con_1', 404,
            ['error' => ['message' => 'no such connection', 'type' => 'not_found', 'code' => 'connection_not_found', 'doc_url' => 'https://getstream.io/agents/docs/api/errors/#not_found']]);

        try {
            $this->client->connections->get('con_1', 'bob');
            self::fail('another user read anya\'s connection');
        } catch (RouterException $refused) {
            self::assertSame(404, $refused->status);
            self::assertSame('connection_not_found', $refused->errorCode);
        }
        self::assertSame('bob', $this->router->received()[0]->headers['x-stream-user-id']);
    }

    /**
     * @return iterable<string, array{\Closure(Client): mixed, string}>
     */
    public static function refusedBeforeSending(): iterable
    {
        $device = static fn (Client $c): Client => $c->asUser('anya', 'tok');
        yield 'a device creating' => [static fn (Client $c) => $device($c)->connections->create(new NewConnection('linear', userId: 'anya')), 'server-side'];
        yield 'a device listing the app\'s' => [static fn (Client $c) => $device($c)->connections->list(), 'server-side'];
        yield 'a get with no id' => [static fn (Client $c) => $c->connections->get('', 'anya'), 'needs a id'];
        yield 'an authorization with no id' => [static fn (Client $c) => $c->connections->authorize('', new AuthorizeOptions(userId: 'anya')), 'needs a id'];
        yield 'a delete with no id' => [static fn (Client $c) => $c->connections->delete(''), 'needs a id'];
    }

    // Catches a device's call or an empty id reaching the router, where the empty id would name
    // /v1/agents/connections/, the list rather than a connection.
    #[DataProvider('refusedBeforeSending')]
    public function testRefusedBeforeSending(\Closure $call, string $why): void
    {
        try {
            $call($this->client);
            self::fail('it was sent');
        } catch (ConfigurationException $refused) {
            self::assertStringContainsString($why, $refused->getMessage());
        }
        self::assertSame([], $this->router->received());
    }

    /**
     * @return array<string, mixed>
     */
    private static function usersConnection(string $id, string $userId): array
    {
        return [...Rows::connection($id, 'linear'), 'owner' => ['type' => ConnectionOwnerType::User->value, 'user_id' => $userId], 'status' => 'pending'];
    }

    /**
     * @return array<string, mixed>
     */
    private static function authorization(): array
    {
        return [
            'id' => 'att_1',
            'kind' => 'consent',
            'launch_url' => 'https://router.example.com/v1/agents/connections/con_1/launch',
            'handoff_token' => 'handoff-1',
            'expires_at' => '2026-09-24T10:10:00Z',
        ];
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
