<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\ForkSessionRequest;
use GetStream\VisionAgents\Generated\ImageSource;
use GetStream\VisionAgents\Generated\SessionSortField;
use GetStream\VisionAgents\Generated\SessionState;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Sessions;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Received;
use GetStream\VisionAgents\Tests\Support\Rows;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class SessionTest extends TestCase
{
    private LocalRouter $router;
    private Session $session;

    protected function setUp(): void
    {
        $this->router = new LocalRouter();
        $this->router->answer('POST', '/v1/agents/sessions', 201, Rows::session());
        $this->session = $this->router->client()->agent('jean')->sessions->create(title: 'Pricing', custom: ['plan' => 'pro']);
    }

    protected function tearDown(): void
    {
        $this->router->stop();
    }

    public function testCreateNamesTheAgentAndHoldsItInWriting(): void
    {
        self::assertSame(
            ['agent' => 'jean', 'custom' => ['plan' => 'pro'], 'title' => 'Pricing'],
            $this->router->to('POST', '/v1/agents/sessions')[0]->json(),
        );
        self::assertSame(SessionState::Live, $this->session->created->state);
        self::assertSame('2026-09-24 10:00:00.123456', $this->session->created->createdAt->format('Y-m-d H:i:s.u'));
        self::assertFalse($this->session->voiceStarted());
    }

    public function testCreateCanStartVoice(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions', 201, Rows::session('ses_2', ['call_id' => 'ses_2', 'call_type' => 'default']));

        $session = $this->router->client()->agent('jean')->sessions->create(startVoice: true, id: 'ses_2');

        self::assertSame(['agent' => 'jean', 'id' => 'ses_2', 'start_voice' => true], $this->router->to('POST', '/v1/agents/sessions')[1]->json());
        self::assertTrue($session->voiceStarted());
    }

    public function testCreateCountsTheSessionUnderATeam(): void
    {
        $this->router->client()->agent('jean')->sessions->create(team: 'finance', userId: 'jean');

        self::assertSame(['agent' => 'jean', 'user_id' => 'jean', 'team' => 'finance'], $this->router->to('POST', '/v1/agents/sessions')[1]->json());
    }

    public function testCreateRefusesATeamTheRouterWould(): void
    {
        try {
            $this->router->client()->agent('jean')->sessions->create(team: 'fin ance');
            self::fail('the team was sent');
        } catch (ConfigurationException) {
            self::assertCount(1, $this->router->to('POST', '/v1/agents/sessions'));
        }
    }

    public function testVoiceStartsAndStopsOnTheSameSession(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/voice', 200, Rows::session('ses_1', ['call_id' => 'ses_1', 'call_type' => 'default']));
        $this->router->answer('DELETE', '/v1/agents/sessions/ses_1/voice', 200, Rows::session('ses_1'));

        $started = $this->session->startVoice();
        self::assertSame('ses_1', $started->callId);
        self::assertTrue($this->session->voiceStarted());

        $this->session->stopVoice();
        self::assertFalse($this->session->voiceStarted());
        self::assertCount(1, $this->router->to('POST', '/v1/agents/sessions/ses_1/voice'));
        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/sessions/ses_1/voice'));
    }

    public function testResponsesCreateAndItems(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1'));
        $this->router->answer('GET', '/v1/agents/sessions/ses_1/responses/items', 200, ['items' => [Rows::item('resp_1', 0), Rows::item('resp_1', 1, 'tool_call')], 'has_more' => false]);

        $response = $this->session->responses->create('What does it cost?');
        $items = $response->items->all();

        self::assertSame('resp_1', $response->id());
        $sent = $this->router->to('POST', '/v1/agents/sessions/ses_1/responses')[0]->json();
        self::assertSame(['text', 'request_id'], array_keys($sent));
        self::assertSame('What does it cost?', $sent['text']);
        self::assertSame(['message', 'tool_call'], array_map(static fn ($item) => $item->kind, $items));
        self::assertSame('resp_1', $this->router->to('GET', '/v1/agents/sessions/ses_1/responses/items')[0]->params()['response_id']);
    }

    public function testEveryTextQuestionCarriesAFreshRequestIdAndOneWithImagesNone(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1'));

        $this->session->responses->create('first');
        $this->session->responses->create('second');
        $this->session->responses->create('what is this?', [new ImageSource(url: 'https://example.com/cat.png')]);

        $sent = array_map(static fn ($seen) => $seen->json(), $this->router->to('POST', '/v1/agents/sessions/ses_1/responses'));
        self::assertMatchesRegularExpression('/^[0-9a-f]{32}$/', Json::string($sent[0], 'request_id'));
        self::assertMatchesRegularExpression('/^[0-9a-f]{32}$/', Json::string($sent[1], 'request_id'));
        self::assertNotSame($sent[0]['request_id'], $sent[1]['request_id']);
        self::assertArrayNotHasKey('request_id', $sent[2]);
    }

    public function testRewindSendsTheResponseId(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1'));
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/rewind', 204);

        $this->session->responses->rewind($this->session->responses->create('first'));

        self::assertSame(['response_id' => 'resp_1'], $this->router->to('POST', '/v1/agents/sessions/ses_1/rewind')[0]->json());
    }

    public function testRewindOfAPersistedConversationIsRefused(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/rewind', 400, Rows::failure('invalid_request', 'this conversation is persisted; fork it instead'));

        try {
            $this->session->responses->rewind('resp_1');
            self::fail('the router refused and nothing was raised');
        } catch (RouterException $refused) {
            self::assertSame(400, $refused->status);
            self::assertStringContainsString('fork it instead', $refused->said);
        }
    }

    public function testRewindNeedsAnId(): void
    {
        $this->expectException(ConfigurationException::class);
        $this->session->responses->rewind('');
    }

    public function testForkFromAResponse(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/fork', 201, Rows::session('ses_2'));

        $fork = $this->session->fork(new ForkSessionRequest(responseId: 'resp_1', title: 'Take two'));

        self::assertSame('ses_2', $fork->id());
        self::assertSame(['response_id' => 'resp_1', 'title' => 'Take two'], $this->router->to('POST', '/v1/agents/sessions/ses_1/fork')[0]->json());
    }

    public function testUpdateChangesOneSessionWithOneRequest(): void
    {
        $this->router->answer('PATCH', '/v1/agents/sessions/ses_1', 200, Rows::session('ses_1', ['llm' => 'llm-thinking', 'title' => 'Plans']));

        $updated = $this->session->update(title: 'Plans', llm: 'llm-thinking', thinking: 'high', sts: '');

        self::assertSame('llm-thinking', $updated->llm);
        self::assertSame(
            ['llm' => 'llm-thinking', 'sts' => '', 'thinking' => 'high', 'title' => 'Plans'],
            $this->router->to('PATCH', '/v1/agents/sessions/ses_1')[0]->json(),
        );
    }

    public function testAnEndedSessionIsRenamedById(): void
    {
        $this->router->answer('PATCH', '/v1/agents/sessions/ses_9', 200, Rows::session('ses_9', ['state' => 'ended', 'title' => 'Pricing']));

        $updated = $this->router->client()->agent('jean')->sessions->update('ses_9', title: 'Pricing');

        self::assertSame('Pricing', $updated->title);
        self::assertSame(['title' => 'Pricing'], $this->router->to('PATCH', '/v1/agents/sessions/ses_9')[0]->json());
    }

    public function testCloseStopsIsIdempotentAndIgnoresAGoneSession(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/stop', 404, Rows::failure('not_found', 'no session ses_1', 'session_not_found'));

        $this->session->close();
        $this->session->close();

        self::assertTrue($this->session->closed());
        self::assertCount(1, $this->router->to('POST', '/v1/agents/sessions/ses_1/stop'));
        self::assertSame([], $this->router->to('DELETE', '/v1/agents/sessions/ses_1'), 'closing keeps the conversation');
    }

    public function testDeleteDeletesTheSession(): void
    {
        $this->router->answer('DELETE', '/v1/agents/sessions/ses_1', 204);

        $this->session->delete();
        $this->session->close();

        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/sessions/ses_1'));
        self::assertSame([], $this->router->to('POST', '/v1/agents/sessions/ses_1/stop'), 'a deleted session has nothing left to stop');
    }

    public function testDeleteMemoriesOfOneSession(): void
    {
        $this->router->answer('DELETE', '/v1/agents/sessions/ses_1/memories', 204);
        $this->router->answer('DELETE', '/v1/agents/sessions/ses_2/memories', 204);

        $this->session->deleteMemories();
        $this->router->client()->agent('jean')->sessions->deleteMemories('ses_2');

        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/sessions/ses_1/memories'));
        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/sessions/ses_2/memories'));
    }

    public function testActions(): void
    {
        foreach (['say', 'interrupt'] as $action) {
            $this->router->answer('POST', "/v1/agents/sessions/ses_1/{$action}", 204);
        }

        $this->session->say('Hello.');
        $this->session->interrupt();

        self::assertSame(['text' => 'Hello.'], $this->router->to('POST', '/v1/agents/sessions/ses_1/say')[0]->json());
        self::assertCount(1, $this->router->to('POST', '/v1/agents/sessions/ses_1/interrupt'));
    }

    public function testQueryWithNoFilterAsksForTheAgentAndReadsAPage(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/query', 200, [
            'items' => [Rows::session('ses_1', ['state' => 'something-new', 'pinned_at' => '2026-11-02T08:30:00Z']), Rows::session('ses_2')],
            'has_more' => true,
            'next_cursor' => 'cur_2',
        ]);

        $page = $this->router->client()->agent('jean')->sessions->query();

        self::assertSame('something-new', $page->items[0]->state, 'a state this SDK does not know is kept, not refused');
        self::assertSame('2026-11-02 08:30:00', $page->items[0]->pinnedAt?->format('Y-m-d H:i:s'));
        self::assertNull($page->items[1]->pinnedAt, 'a session that is not pinned has no pinned_at');
        self::assertTrue($page->hasMore);
        self::assertSame('cur_2', $page->nextCursor);
        self::assertSame('{"filter":{"agent":"jean"}}', $this->router->to('POST', '/v1/agents/sessions/query')[0]->body);
    }

    public function testQueryFiltersByProject(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: ['project_id' => 'health']));

        self::assertSame(['filter' => ['agent' => 'jean', 'project_id' => 'health']], $sent->json());
    }

    public function testQueryFiltersByEitherOfTwoProjects(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(
            filter: ['$or' => [['project_id' => 'health'], ['project_id' => ['$eq' => 'legal']]]],
        ));

        self::assertSame(
            ['filter' => ['agent' => 'jean', '$or' => [['project_id' => 'health'], ['project_id' => ['$eq' => 'legal']]]]],
            $sent->json(),
        );
    }

    public function testQueryNestsAndAndOr(): void
    {
        $filter = [
            '$and' => [
                ['user_id' => 'u1'],
                ['$or' => [
                    ['project_id' => 'health'],
                    ['$and' => [['project_id' => ['$in' => ['legal', 'tax']]], ['review_score' => ['$gte' => 4]]]],
                ]],
            ],
        ];

        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: $filter));

        self::assertSame(['filter' => ['agent' => 'jean', ...$filter]], $sent->json());
        self::assertStringContainsString('"$and":[{"user_id":"u1"},{"$or":[', $sent->body, 'the caller\'s $and is the top-level one, not wrapped in another');
    }

    public function testQueryFiltersByAnyOfSeveralProjects(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: ['project_id' => ['$in' => ['health', 'legal']]]));

        self::assertSame('{"filter":{"agent":"jean","project_id":{"$in":["health","legal"]}}}', $sent->body);
    }

    public function testQueryFiltersByReviewScoreAndUser(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: ['review_score' => ['$gte' => 2, '$lt' => 4], 'user_id' => 'u1']));
        $exact = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: ['review_score' => 5, 'user_id' => ['$eq' => 'u1']]));

        self::assertSame(['filter' => ['agent' => 'jean', 'review_score' => ['$gte' => 2, '$lt' => 4], 'user_id' => 'u1']], $sent->json());
        self::assertSame(['filter' => ['agent' => 'jean', 'review_score' => 5, 'user_id' => ['$eq' => 'u1']]], $exact->json());
    }

    public function testQueryIsAlwaysOfTheAgentAsked(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: ['project_id' => 'health', 'agent' => 'somebody-else']));

        self::assertSame(['filter' => ['agent' => 'jean', 'project_id' => 'health']], $sent->json());
    }

    public function testAnEmptyFilterOrConditionIsSentAsAnObject(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(filter: ['project_id' => [], '$or' => [[], ['$and' => [[]]]]]));

        self::assertSame('{"filter":{"agent":"jean","project_id":{},"$or":[{},{"$and":[{}]}]}}', $sent->body);
    }

    /**
     * @param list<array{field: SessionSortField|string, direction?: -1|1}> $sort
     * @param list<array<string, mixed>> $wire
     */
    #[DataProvider('sorts')]
    public function testQuerySorts(array $sort, array $wire): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(sort: $sort));

        self::assertSame(['filter' => ['agent' => 'jean'], 'sort' => $wire], $sent->json());
    }

    /**
     * @return iterable<string, array{list<array{field: SessionSortField|string, direction?: -1|1}>, list<array<string, mixed>>}>
     */
    public static function sorts(): iterable
    {
        yield 'last answer, the default direction' => [[['field' => 'last_message_at']], [['field' => 'last_message_at']]];
        yield 'last answer, newest first' => [[['field' => 'last_message_at', 'direction' => -1]], [['field' => 'last_message_at', 'direction' => -1]]];
        yield 'last answer, oldest first' => [[['field' => SessionSortField::LastMessageAt, 'direction' => 1]], [['field' => 'last_message_at', 'direction' => 1]]];
        yield 'pinned first' => [[['field' => 'pinned']], [['field' => 'pinned']]];
        yield 'pinned first, then newest' => [
            [['field' => SessionSortField::Pinned, 'direction' => -1], ['field' => SessionSortField::LastMessageAt, 'direction' => -1]],
            [['field' => 'pinned', 'direction' => -1], ['field' => 'last_message_at', 'direction' => -1]],
        ];
        yield 'pinned first, then oldest' => [
            [['field' => 'pinned', 'direction' => -1], ['field' => 'last_message_at', 'direction' => 1]],
            [['field' => 'pinned', 'direction' => -1], ['field' => 'last_message_at', 'direction' => 1]],
        ];
        yield 'best reviewed first' => [[['field' => SessionSortField::ReviewScore, 'direction' => -1]], [['field' => 'review_score', 'direction' => -1]]];
        yield 'worst reviewed first' => [[['field' => 'review_score', 'direction' => 1]], [['field' => 'review_score', 'direction' => 1]]];
    }

    public function testQuerySortsATextFilterByRelevance(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(
            filter: ['text' => ['$q' => 'lisbon']],
            sort: [['field' => SessionSortField::Relevance, 'direction' => -1]],
        ));

        self::assertSame(
            ['filter' => ['agent' => 'jean', 'text' => ['$q' => 'lisbon']], 'sort' => [['field' => 'relevance', 'direction' => -1]]],
            $sent->json(),
        );
    }

    public function testQueryPagesWithACursor(): void
    {
        $sent = $this->queried(fn (Sessions $sessions) => $sessions->query(
            filter: ['project_id' => 'health'],
            sort: [['field' => 'pinned']],
            limit: 20,
            cursor: 'cur_1',
        ));

        self::assertSame(
            ['cursor' => 'cur_1', 'filter' => ['agent' => 'jean', 'project_id' => 'health'], 'limit' => 20, 'sort' => [['field' => 'pinned']]],
            $sent->json(),
        );
    }

    public function testSearchIsAQueryWithTextAtTheTopLevel(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/query', 200, ['items' => [], 'has_more' => false]);
        $sessions = $this->router->client()->agent('jean')->sessions;

        $page = $sessions->search('pricing');
        $sessions->search('pricing plans', filter: ['project_id' => 'health'], limit: 5, cursor: 'cur_1');
        $sessions->search('pricing', filter: ['$and' => [['user_id' => 'u1'], ['$or' => [['project_id' => 'a'], ['project_id' => 'b']]]]]);

        [$plain, $narrowed, $nested] = $this->router->to('POST', '/v1/agents/sessions/query');
        self::assertSame([], $page->items);
        self::assertSame(['filter' => ['agent' => 'jean', 'text' => ['$q' => 'pricing']]], $plain->json());
        self::assertSame(
            ['cursor' => 'cur_1', 'filter' => ['agent' => 'jean', 'project_id' => 'health', 'text' => ['$q' => 'pricing plans']], 'limit' => 5],
            $narrowed->json(),
        );
        self::assertSame(
            ['filter' => [
                'agent' => 'jean',
                '$and' => [['user_id' => 'u1'], ['$or' => [['project_id' => 'a'], ['project_id' => 'b']]]],
                'text' => ['$q' => 'pricing'],
            ]],
            $nested->json(),
        );
    }

    public function testSearchRefusesAFilterWithItsOwnText(): void
    {
        try {
            $this->router->client()->agent('jean')->sessions->search('pricing', filter: ['text' => ['$q' => 'plans']]);
            self::fail('a second text was sent');
        } catch (ConfigurationException $refused) {
            self::assertStringContainsString('text', $refused->getMessage());
        }
        self::assertSame([], $this->router->to('POST', '/v1/agents/sessions/query'));
    }

    public function testPinAndUnpinPatchTheSession(): void
    {
        $this->router->answer('PATCH', '/v1/agents/sessions/ses_1', 200, Rows::session('ses_1', ['pinned_at' => '2026-11-02T08:30:00Z']));
        $this->router->answer('PATCH', '/v1/agents/sessions/ses_9', 200, Rows::session('ses_9', ['state' => 'ended']));
        $sessions = $this->router->client()->agent('jean')->sessions;

        $pinned = $this->session->pin();
        $this->session->unpin();
        $this->session->update(title: 'Plans', pinned: true);
        $sessions->pin('ses_9');
        $unpinned = $sessions->unpin('ses_9');
        $sessions->update('ses_9', pinned: false);

        self::assertSame('2026-11-02 08:30:00', $pinned->pinnedAt?->format('Y-m-d H:i:s'));
        self::assertNull($unpinned->pinnedAt);
        self::assertSame(
            ['{"pinned":true}', '{"pinned":false}', '{"title":"Plans","pinned":true}'],
            array_map(static fn (Received $each): string => $each->body, $this->router->to('PATCH', '/v1/agents/sessions/ses_1')),
        );
        self::assertSame(
            ['{"pinned":true}', '{"pinned":false}', '{"pinned":false}'],
            array_map(static fn (Received $each): string => $each->body, $this->router->to('PATCH', '/v1/agents/sessions/ses_9')),
        );
    }

    /**
     * What one query sent.
     *
     * @param \Closure(Sessions): mixed $ask
     */
    private function queried(\Closure $ask): Received
    {
        $this->router->answer('POST', '/v1/agents/sessions/query', 200, ['items' => [], 'has_more' => false]);
        $before = count($this->router->to('POST', '/v1/agents/sessions/query'));
        $ask($this->router->client()->agent('jean')->sessions);
        $sent = $this->router->to('POST', '/v1/agents/sessions/query');
        self::assertCount($before + 1, $sent);
        return $sent[$before];
    }

    public function testCreateTakesAnIdAndAProject(): void
    {
        $this->router->client()->agent('jean')->sessions->create(projectId: 'health', id: '0192f0c0-0000-7000-8000-000000000001');

        $sent = $this->router->to('POST', '/v1/agents/sessions')[1]->json();
        self::assertSame('0192f0c0-0000-7000-8000-000000000001', $sent['id']);
        self::assertSame('health', $sent['project_id']);
        self::assertArrayNotHasKey('project', $sent);
    }

    public function testResponsesAndItemsPageByCursor(): void
    {
        $this->router->answer('GET', '/v1/agents/sessions/ses_1/responses', 200, ['items' => [Rows::response('resp_1')], 'has_more' => false]);
        $this->router->answer(
            'GET',
            '/v1/agents/sessions/ses_1/responses/items',
            200,
            ['items' => [Rows::item('resp_1', 0)], 'has_more' => true, 'next_cursor' => 'cur_2'],
            ['items' => [Rows::item('resp_1', 1)], 'has_more' => false],
        );

        $page = $this->session->responses->list(limit: 5);
        $items = $this->session->responses->items->all();

        self::assertSame('resp_1', $page->items[0]->id);
        self::assertSame(['limit' => '5'], $this->router->to('GET', '/v1/agents/sessions/ses_1/responses')[0]->params());
        self::assertSame([0, 1], array_map(static fn ($item) => $item->ordinal, $items));
        $asked = $this->router->to('GET', '/v1/agents/sessions/ses_1/responses/items');
        self::assertArrayNotHasKey('cursor', $asked[0]->params());
        self::assertSame('cur_2', $asked[1]->params()['cursor']);
    }
}
