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
            ['text' => true, 'agent' => 'jean', 'title' => 'Pricing', 'custom' => ['plan' => 'pro']],
            $this->router->to('POST', '/v1/agents/sessions')[0]->json(),
        );
        self::assertSame(SessionState::Live, $this->session->created->state);
        self::assertSame('2026-09-24 10:00:00.123456', $this->session->created->createdAt->format('Y-m-d H:i:s.u'));
    }

    public function testResponsesCreateAndItems(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/responses', 201, Rows::response('resp_1'));
        $this->router->answer('GET', '/v1/agents/sessions/ses_1/responses/items', 200, [Rows::item('resp_1', 0), Rows::item('resp_1', 1, 'tool_call')]);

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
        self::assertSame(['title' => 'Take two', 'response_id' => 'resp_1'], $this->router->to('POST', '/v1/agents/sessions/ses_1/fork')[0]->json());
    }

    public function testCloseIsIdempotentAndIgnoresAGoneSession(): void
    {
        $this->router->answer('DELETE', '/v1/agents/sessions/ses_1', 404, ['error' => 'gone']);

        $this->session->close();
        $this->session->close();

        self::assertTrue($this->session->closed());
        self::assertCount(1, $this->router->to('DELETE', '/v1/agents/sessions/ses_1'));
    }

    public function testActions(): void
    {
        foreach (['say', 'respond', 'interrupt'] as $action) {
            $this->router->answer('POST', "/v1/agents/sessions/ses_1/{$action}", 204);
        }
        $this->router->answer('PUT', '/v1/agents/sessions/ses_1/instructions', 204);

        $this->session->say('Hello.');
        $this->session->respond('Tell them the price.');
        $this->session->interrupt();
        $this->session->setInstructions('Be brief.');

        self::assertSame(['text' => 'Hello.'], $this->router->to('POST', '/v1/agents/sessions/ses_1/say')[0]->json());
        self::assertSame(['text' => 'Tell them the price.'], $this->router->to('POST', '/v1/agents/sessions/ses_1/respond')[0]->json());
        self::assertCount(1, $this->router->to('POST', '/v1/agents/sessions/ses_1/interrupt'));
        self::assertSame(['instructions' => 'Be brief.'], $this->router->to('PUT', '/v1/agents/sessions/ses_1/instructions')[0]->json());
    }

    public function testQueryFiltersByAgentAndLabels(): void
    {
        $this->router->answer('GET', '/v1/agents/sessions', 200, [Rows::session('ses_1', ['state' => 'something-new'])]);

        $rows = $this->router->client()->agent('jean')->sessions->query(state: 'running', custom: ['plan' => 'pro'], limit: 10);

        self::assertSame('something-new', $rows[0]->state, 'a state this SDK does not know is kept, not refused');
        self::assertSame(
            ['agent' => 'jean', 'state' => 'running', 'custom' => '{"plan":"pro"}', 'limit' => '10'],
            $this->router->to('GET', '/v1/agents/sessions')[0]->params(),
        );
    }
}
