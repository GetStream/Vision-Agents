<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\ForkSessionRequest;
use GetStream\VisionAgents\Generated\SessionState;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Rows;
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
            ['agent' => 'jean', 'custom' => ['plan' => 'pro'], 'text' => true, 'title' => 'Pricing'],
            $this->router->to('POST', '/v1/agents/sessions')[0]->json(),
        );
        self::assertSame(SessionState::Live, $this->session->created->state);
        self::assertSame('2026-09-24 10:00:00.123456', $this->session->created->createdAt->format('Y-m-d H:i:s.u'));
    }

    public function testResponsesCreateAndItems(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1'));
        $this->router->answer('GET', '/v1/agents/sessions/ses_1/responses/items', 200, ['items' => [Rows::item('resp_1', 0), Rows::item('resp_1', 1, 'tool_call')], 'has_more' => false]);

        $response = $this->session->responses->create('What does it cost?');
        $items = $response->items->all();

        self::assertSame('resp_1', $response->id());
        self::assertSame(['text' => 'What does it cost?'], $this->router->to('POST', '/v1/agents/sessions/ses_1/responses')[0]->json());
        self::assertSame(['message', 'tool_call'], array_map(static fn ($item) => $item->kind, $items));
        self::assertSame('resp_1', $this->router->to('GET', '/v1/agents/sessions/ses_1/responses/items')[0]->params()['response_id']);
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
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/rewind', 400, ['error' => 'this conversation is persisted; fork it instead']);

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

        $updated = $this->session->update(title: 'Plans', instructions: 'Be brief.', llm: 'llm-thinking', thinking: 'high', sts: '');

        self::assertSame('llm-thinking', $updated->llm);
        self::assertSame(
            ['instructions' => 'Be brief.', 'llm' => 'llm-thinking', 'sts' => '', 'thinking' => 'high', 'title' => 'Plans'],
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
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/stop', 404, ['error' => 'gone']);

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

    public function testQueryPostsTheFilterAndReadsAPage(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/query', 200, ['items' => [Rows::session('ses_1', ['state' => 'something-new'])], 'has_more' => true, 'next_cursor' => 'cur_2']);

        $page = $this->router->client()->agent('jean')->sessions->query(userId: 'u1', state: 'live', agentId: 'jean-7', limit: 10, cursor: 'cur_1');

        self::assertSame('something-new', $page->items[0]->state, 'a state this SDK does not know is kept, not refused');
        self::assertTrue($page->hasMore);
        self::assertSame('cur_2', $page->nextCursor);
        self::assertSame(
            ['cursor' => 'cur_1', 'filter' => ['agent' => 'jean', 'agent_id' => 'jean-7', 'state' => 'live', 'user_id' => 'u1'], 'limit' => 10],
            $this->router->to('POST', '/v1/agents/sessions/query')[0]->json(),
        );
    }

    public function testSearchIsAQueryWithText(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/query', 200, ['items' => [], 'has_more' => false]);

        $page = $this->router->client()->agent('jean')->sessions->search('pricing', modality: 'voice');

        self::assertSame([], $page->items);
        self::assertSame(
            ['filter' => ['agent' => 'jean', 'modality' => 'voice', 'text' => ['$q' => 'pricing']]],
            $this->router->to('POST', '/v1/agents/sessions/query')[0]->json(),
        );
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
