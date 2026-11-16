<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\LimitInterval;
use GetStream\VisionAgents\Generated\LimitLevel;
use GetStream\VisionAgents\Generated\LimitSubject;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Rows;
use GetStream\VisionAgents\Usd;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class SpendLimitsTest extends TestCase
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

    public function testCreatesALimitInMicros(): void
    {
        $this->router->answer('POST', '/v1/spend/limits', 201, Rows::limit(7, 'api_key', 'daily', 10_000_000, ['level' => 'api_key', 'app_id' => '1234', 'api_key' => 'mz7kq2xa9b', 'note' => 'CI key']));

        $limit = $this->client->spendLimits->create(LimitSubject::ApiKey, LimitInterval::Daily, Usd::dollars(10), appId: '1234', apiKey: 'mz7kq2xa9b', note: 'CI key');

        self::assertSame(7, $limit->id);
        self::assertSame(LimitSubject::ApiKey, $limit->subject);
        self::assertSame('$10.00', (string) Usd::micros($limit->limitMicros));
        self::assertSame(
            ['interval' => 'daily', 'limit_micros' => 10_000_000, 'subject' => 'api_key', 'api_key' => 'mz7kq2xa9b', 'app_id' => '1234', 'note' => 'CI key'],
            $this->router->to('POST', '/v1/spend/limits')[0]->json(),
        );
    }

    public function testATeamOrCategoryWithinAnAppIsSentAsTheAppsSubject(): void
    {
        $this->router->answer('POST', '/v1/spend/limits', 201, Rows::limit(1, 'app_team', 'monthly', 1));

        $this->client->spendLimits->create(LimitSubject::Team, LimitInterval::Monthly, Usd::micros(1), appId: '1234', team: 'finance');
        $this->client->spendLimits->create('category', 'weekly', Usd::cents(150), appId: '1234', category: 'project:onboarding', level: LimitLevel::App);

        [$team, $category] = $this->router->to('POST', '/v1/spend/limits');
        self::assertSame(['interval' => 'monthly', 'limit_micros' => 1, 'subject' => 'app_team', 'app_id' => '1234', 'team' => 'finance'], $team->json());
        self::assertSame(['interval' => 'weekly', 'limit_micros' => 1_500_000, 'subject' => 'app_category', 'app_id' => '1234', 'category' => 'project:onboarding', 'level' => 'app'], $category->json());
    }

    /**
     * @return iterable<string, array{callable(Client): mixed, string}>
     */
    public static function refusedBeforeTheRouter(): iterable
    {
        yield 'no subject' => [static fn (Client $c) => $c->spendLimits->create('', LimitInterval::Daily, Usd::dollars(1)), 'name the subject'];
        yield 'no interval' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::Organization, '', Usd::dollars(1)), 'interval'];
        yield 'below zero' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::Organization, LimitInterval::Daily, Usd::parse('-1')), 'below zero'];
        yield 'team subject without a team' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::Team, LimitInterval::Daily, Usd::dollars(1)), 'names its team'];
        yield 'app team without an app' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::AppTeam, LimitInterval::Daily, Usd::dollars(1), team: 'finance'), 'names its app'];
        yield 'a team on another subject' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::User, LimitInterval::Daily, Usd::dollars(1), team: 'finance'), 'team is for the team subject'];
        yield 'a team name the router refuses' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::Team, LimitInterval::Daily, Usd::dollars(1), team: 'fin:ance'), '1 to 59'];
        yield 'a category that is a team' => [static fn (Client $c) => $c->spendLimits->create(LimitSubject::Category, LimitInterval::Daily, Usd::dollars(1), category: 'Team:finance'), 'names a team'];
        yield 'a filter on a bad team' => [static fn (Client $c) => $c->spendLimits->list(team: 'a b'), '1 to 59'];
        yield 'a reset without a reason' => [static fn (Client $c) => $c->spendLimits->reset(LimitSubject::User, '', userId: 'user-42'), 'reason'];
        yield 'raising below zero' => [static fn (Client $c) => $c->spendLimits->setAmount(7, Usd::micros(-1)), 'below zero'];
        yield 'a team limit without an interval' => [static fn (Client $c) => $c->teams->setLimit('finance', '', Usd::dollars(1)), 'interval'];
        yield 'a team limit for a bad team' => [static fn (Client $c) => $c->teams->removeLimit('', LimitInterval::Daily), '1 to 59'];
    }

    /**
     * @param callable(Client): mixed $call
     */
    #[DataProvider('refusedBeforeTheRouter')]
    public function testWhatTheRouterWouldRefuseIsRefusedHere(callable $call, string $said): void
    {
        try {
            $call($this->client);
            self::fail('it was sent');
        } catch (ConfigurationException $refused) {
            self::assertStringContainsString($said, $refused->getMessage());
            self::assertSame([], $this->router->received());
        }
    }

    public function testListsEveryPageWithTheFilters(): void
    {
        $this->router->answer(
            'GET',
            '/v1/spend/limits',
            200,
            ['items' => [Rows::limit(1, 'user', 'daily', 20_000_000)], 'has_more' => true, 'next_cursor' => 'c1'],
            ['items' => [Rows::limit(2, 'user', 'daily', 50_000_000, ['user_id' => 'user-42'])], 'has_more' => false],
        );

        $limits = $this->client->spendLimits->list(subject: LimitSubject::User, interval: LimitInterval::Daily, appId: '1234');

        self::assertSame([1, 2], array_map(static fn ($limit) => $limit->id, $limits));
        [$first, $second] = $this->router->to('GET', '/v1/spend/limits');
        self::assertSame(['subject' => 'user', 'interval' => 'daily', 'app_id' => '1234', 'limit' => '200'], $first->params());
        self::assertSame(['subject' => 'user', 'interval' => 'daily', 'app_id' => '1234', 'limit' => '200', 'cursor' => 'c1'], $second->params());
    }

    public function testAnApiKeyFilterIsSentAsApiKeyId(): void
    {
        $this->router->answer('GET', '/v1/spend/limits', 200, ['items' => [], 'has_more' => false]);

        $this->client->spendLimits->list(apiKey: 'mz7kq2xa9b');

        self::assertSame(['api_key_id' => 'mz7kq2xa9b', 'limit' => '200'], $this->router->received()[0]->params());
    }

    public function testGetsSetsAndDeletesOneLimit(): void
    {
        $this->router->answer('GET', '/v1/spend/limits/7', 200, Rows::limit(7, 'api_key', 'daily', 10_000_000));
        $this->router->answer('PATCH', '/v1/spend/limits/7', 200, Rows::limit(7, 'api_key', 'daily', 25_000_000));
        $this->router->answer('DELETE', '/v1/spend/limits/7', 204);

        $limits = $this->client->spendLimits;
        self::assertSame(10_000_000, $limits->get(7)->limitMicros);
        self::assertSame(25_000_000, $limits->setAmount(7, Usd::dollars(25))->limitMicros);
        $limits->setNote(7, '');
        $limits->delete(7);

        $patches = $this->router->to('PATCH', '/v1/spend/limits/7');
        self::assertSame(['limit_micros' => 25_000_000], $patches[0]->json());
        self::assertSame(['note' => ''], $patches[1]->json());
        self::assertCount(1, $this->router->to('DELETE', '/v1/spend/limits/7'));
    }

    public function testResetsOneUsersWindows(): void
    {
        $this->router->answer('POST', '/v1/spend/limits/usage/reset', 200, Rows::usage('user', ['app_id' => '1234', 'user_id' => 'user-42']));

        $usage = $this->client->spendLimits->reset(LimitSubject::User, 'refunded a runaway job', [LimitInterval::Weekly], appId: '1234', userId: 'user-42');

        self::assertSame('user-42', $usage->userId);
        self::assertSame(
            ['reason' => 'refunded a runaway job', 'subject' => 'user', 'app_id' => '1234', 'intervals' => ['weekly'], 'user_id' => 'user-42'],
            $this->router->received()[0]->json(),
        );
    }

    public function testReadsWhatOneSubjectSpent(): void
    {
        $this->router->answer('GET', '/v1/spend/limits/usage', 200, [Rows::usage('team', ['team' => 'finance'])]);

        $usage = $this->client->teams->usage('finance');

        self::assertSame('finance', $usage->team);
        self::assertSame(['subject' => 'team', 'team' => 'finance'], $this->router->received()[0]->params());
    }

    public function testListsTeams(): void
    {
        $this->router->answer('GET', '/v1/teams', 200, [['name' => 'finance', 'apps' => ['1234'], 'limits' => [Rows::limit(3, 'team', 'monthly', 8_000_000_000, ['team' => 'finance'])], 'spent' => []]]);

        $teams = $this->client->teams->list('1234');

        self::assertSame('finance', $teams[0]->name);
        self::assertSame(8_000_000_000, $teams[0]->limits[0]->limitMicros);
        self::assertSame(['app_id' => '1234'], $this->router->received()[0]->params());
    }

    public function testATeamsLimitIsCreatedWhenItHasNone(): void
    {
        $this->router->answer('GET', '/v1/spend/limits', 200, ['items' => [Rows::limit(9, 'app_team', 'monthly', 1, ['team' => 'finance', 'app_id' => '1234'])], 'has_more' => false]);
        $this->router->answer('POST', '/v1/spend/limits', 201, Rows::limit(3, 'team', 'monthly', 8_000_000_000, ['team' => 'finance']));

        $limit = $this->client->teams->setLimit('finance', LimitInterval::Monthly, Usd::dollars(8_000), note: 'Q4');

        self::assertSame(3, $limit->id);
        self::assertSame(['subject' => 'team', 'interval' => 'monthly', 'team' => 'finance', 'limit' => '200'], $this->router->received()[0]->params());
        self::assertSame(
            ['interval' => 'monthly', 'limit_micros' => 8_000_000_000, 'subject' => 'team', 'note' => 'Q4', 'team' => 'finance'],
            $this->router->to('POST', '/v1/spend/limits')[0]->json(),
        );
        self::assertSame([], $this->router->to('PATCH', '/v1/spend/limits/9'), 'the app\'s limit is another one');
    }

    public function testATeamsLimitIsChangedWhenItHasOne(): void
    {
        $this->router->answer('GET', '/v1/spend/limits', 200, ['items' => [Rows::limit(9, 'app_team', 'monthly', 1, ['team' => 'finance', 'app_id' => '1234'])], 'has_more' => false]);
        $this->router->answer('PATCH', '/v1/spend/limits/9', 200, Rows::limit(9, 'app_team', 'monthly', 8_000_000_000, ['team' => 'finance', 'app_id' => '1234']));

        $this->client->teams->setLimit('finance', LimitInterval::Monthly, Usd::dollars(8_000), appId: '1234');

        self::assertSame(['subject' => 'app_team', 'interval' => 'monthly', 'app_id' => '1234', 'team' => 'finance', 'limit' => '200'], $this->router->received()[0]->params());
        self::assertSame(['limit_micros' => 8_000_000_000], $this->router->to('PATCH', '/v1/spend/limits/9')[0]->json());
        self::assertSame([], $this->router->to('POST', '/v1/spend/limits'));
    }

    public function testATeamsLimitCreatedMeanwhileIsChanged(): void
    {
        $this->router->answer(
            'GET',
            '/v1/spend/limits',
            200,
            ['items' => [], 'has_more' => false],
            ['items' => [Rows::limit(4, 'team', 'weekly', 1, ['team' => 'finance'])], 'has_more' => false],
        );
        $this->router->answer('POST', '/v1/spend/limits', 409, Rows::failure('conflict', 'a limit exists', 'limit_exists'));
        $this->router->answer('PATCH', '/v1/spend/limits/4', 200, Rows::limit(4, 'team', 'weekly', 2_500_000_000, ['team' => 'finance']));

        $limit = $this->client->teams->setLimit('finance', LimitInterval::Weekly, Usd::dollars(2_500));

        self::assertSame(2_500_000_000, $limit->limitMicros);
    }

    public function testRemovingATeamsLimitDeletesItAndNothingWhenThereIsNone(): void
    {
        $this->router->answer(
            'GET',
            '/v1/spend/limits',
            200,
            ['items' => [Rows::limit(5, 'team', 'daily', 1, ['team' => 'finance'])], 'has_more' => false],
            ['items' => [], 'has_more' => false],
        );
        $this->router->answer('DELETE', '/v1/spend/limits/5', 404, Rows::failure('not_found', 'no such limit'));

        $this->client->teams->removeLimit('finance', LimitInterval::Daily);
        $this->client->teams->removeLimit('finance', LimitInterval::Daily);

        self::assertCount(1, $this->router->to('DELETE', '/v1/spend/limits/5'));
    }

    public function testARefusalOtherThanTheRaceIsRaised(): void
    {
        $this->router->answer('GET', '/v1/spend/limits', 200, ['items' => [], 'has_more' => false]);
        $this->router->answer('POST', '/v1/spend/limits', 400, Rows::failure('invalid_request', 'above the organization', 'limit_above_parent'));

        try {
            $this->client->teams->setLimit('finance', LimitInterval::Monthly, Usd::dollars(1_000_000));
            self::fail('the refusal was swallowed');
        } catch (RouterException $refused) {
            self::assertSame('limit_above_parent', $refused->errorCode);
        }
    }
}
