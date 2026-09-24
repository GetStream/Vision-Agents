<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Edge;
use GetStream\VisionAgents\Folder;
use GetStream\VisionAgents\Generated\Sandbox;
use GetStream\VisionAgents\Harness;
use GetStream\VisionAgents\Inbound\InboundCall;
use GetStream\VisionAgents\Inbound\InboundMessage;
use GetStream\VisionAgents\Json;
use GetStream\VisionAgents\Pipeline;
use GetStream\VisionAgents\Skill;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Rows;
use PHPUnit\Framework\TestCase;

final class AgentTest extends TestCase
{
    private LocalRouter $router;
    private string $dir = '';

    protected function setUp(): void
    {
        $this->router = new LocalRouter();
        $this->router->answer('POST', '/v1/agents/sessions', 201, Rows::session());
    }

    protected function tearDown(): void
    {
        $this->router->stop();
        if ($this->dir !== '') {
            exec('rm -rf ' . escapeshellarg(dirname($this->dir)));
        }
    }

    public function testChatCarriesTheConfiguration(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, [Rows::config('cfg_9', 'support')]);
        $agent = new Agent(
            config: 'support',
            instructions: 'Be brief.',
            pipeline: new Pipeline(llm: 'openai/gpt-5.6', language: 'fr'),
            harness: new Harness(skills: [new Skill('think', 'Work it out', 'Reason it through.', deadline: 30.0)]),
            sandbox: Sandbox::Daytona,
            costTracking: ['env' => 'production'],
            memoryFilter: ['user_id' => 123, 'topic' => 'billing'],
            client: $this->router->client(),
        );

        $session = $agent->chat();

        self::assertSame('ses_1', $session->id());
        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertSame('cfg_9', $sent['config_id']);
        self::assertTrue($sent['text']);
        self::assertSame('support', $sent['user_id']);
        self::assertSame('support', $sent['agent_id']);
        self::assertSame('Be brief.', $sent['instructions']);
        self::assertSame('openai/gpt-5.6', $sent['llm']);
        self::assertSame(['fr'], $sent['languages']);
        self::assertSame('daytona', $sent['sandbox']);
        self::assertSame(['env' => 'production'], $sent['tags']);
        self::assertSame(['user_id' => '123', 'filter' => ['topic' => 'billing']], $sent['memory']);
        self::assertSame([['name' => 'think', 'description' => 'Work it out', 'instructions' => 'Reason it through.', 'deadline_ms' => 30000]], array_map(
            static fn (mixed $skill): array => array_intersect_key(Json::asObject($skill), array_flip(['name', 'description', 'instructions', 'deadline_ms'])),
            Json::list($sent, 'skills'),
        ));
        self::assertArrayNotHasKey('stt', $sent, 'what was not set is left to the config');
    }

    public function testAConfigNameMatchingNothingIsPassedThrough(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);
        $agent = new Agent(config: 'cfg_raw', client: $this->router->client());

        $agent->chat();

        self::assertSame('cfg_raw', $this->router->to('POST', '/v1/agents/sessions')[0]->json()['config_id']);
    }

    public function testDeclaredToolsAreSent(): void
    {
        $agent = new Agent(name: 'Jean Luc', client: $this->router->client());
        $agent->tools->register('get_weather', 'The weather somewhere', ['type' => 'object'], static fn (array $args): string => 'sunny');

        $agent->chat();

        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertSame('jean-luc', $sent['user_id']);
        self::assertSame([['name' => 'get_weather', 'description' => 'The weather somewhere', 'parameters' => ['type' => 'object']]], $sent['tools']);
    }

    public function testJoinCreatesTheStreamCallFirst(): void
    {
        $this->router->answer('POST', '/api/v2/video/call/agent/hello', 201, ['duration' => '1ms', 'created' => true]);
        $agent = new Agent(name: 'jean', client: $this->router->client(), edge: $this->edge());

        $agent->join('hello');

        $received = $this->router->received();
        self::assertSame('/api/v2/video/call/agent/hello', $received[0]->path);
        self::assertSame('jean', Json::object($received[0]->json(), 'data')['created_by_id']);
        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertSame('hello', $sent['call_id']);
        self::assertSame('agent', $sent['call_type']);
        self::assertArrayNotHasKey('text', $sent);
    }

    public function testOutboundCallPlacesTheCallBeforeJoining(): void
    {
        $this->router->answer('POST', '/api/v2/video/call/agent/out', 201, ['duration' => '1ms']);
        $this->router->answer('POST', '/v1/phone/calls', 201, ['vendor_call_id' => 'CA123', 'status' => 'queued']);
        $agent = new Agent(name: 'jean', costTracking: ['team' => 'sales'], client: $this->router->client(), edge: $this->edge());

        $agent->outboundCall('+15550001111', '+15552223333', 'out');

        $paths = array_map(static fn ($r) => $r->path, $this->router->received());
        self::assertSame(['/api/v2/video/call/agent/out', '/v1/phone/calls', '/v1/agents/sessions'], $paths);
        self::assertSame(
            ['from' => '+15550001111', 'to' => '+15552223333', 'call_id' => 'out', 'call_type' => 'agent', 'tags' => ['team' => 'sales']],
            $this->router->to('POST', '/v1/phone/calls')[0]->json(),
        );
        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertTrue($sent['navigating']);
        self::assertSame(['number' => '+15550001111', 'vendor_call_id' => 'CA123'], $sent['phone']);
    }

    public function testAnswerCarriesTheNumberReached(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $agent->answer(InboundCall::fromFrame(['call_id' => 'c1', 'called_number' => '+15550001111']));

        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertSame('c1', $sent['call_id']);
        self::assertSame('default', $sent['call_type']);
        self::assertSame(['number' => '+15550001111'], $sent['phone']);
    }

    public function testReplyAnswersInTheChannel(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $agent->reply(InboundMessage::fromFrame(['channel_id' => 'ch1', 'text' => 'hi', 'agent_id' => 'jean-7']));

        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertTrue($sent['persist_conversation']);
        self::assertSame('agent:ch1', $sent['conversation_id']);
        self::assertSame('jean-7', $sent['agent_id']);
    }

    public function testSyncSendsTheFolderAndSkipsWhenUnchanged(): void
    {
        $this->folder();
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);
        $this->router->answer('GET', '/v1/agents/configs', 200, [Rows::config('cfg_1', 'jean')]);
        $agent = new Agent(folder: $this->dir, client: $this->router->client());

        $first = $agent->sync();
        $second = (new Agent(folder: $this->dir, client: $this->router->client()))->sync();

        self::assertFalse($first->unchanged);
        self::assertTrue($second->unchanged);
        self::assertSame('cfg_1', $second->config->id);
        $syncs = $this->router->to('POST', '/v1/agents/sync');
        self::assertCount(1, $syncs, 'an unchanged folder is not synced again');
        $sent = $syncs[0]->json();
        self::assertSame('jean', $sent['name']);
        self::assertSame('02a7b2c8428f31e3a2b93ca2f5a6ec70', $sent['hash']);
        self::assertSame('openai/gpt-5.6', $sent['llm']);
        self::assertSame('You are Jean.', $sent['instructions']);
        self::assertSame('pricing.md', Json::objects($sent, 'knowledge')[0]['source']);
        self::assertSame('02a7b2c8428f31e3a2b93ca2f5a6ec70', Folder::load($this->dir)->stamp());
    }

    public function testSyncCarriesKnowledgeUrlsInTheSameRequest(): void
    {
        $this->folder();
        file_put_contents($this->dir . '/knowledge/urls.yaml', "- https://example.com/plans\n- url: https://example.com/faq\n  title: FAQ\n");
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);

        (new Agent(folder: $this->dir, client: $this->router->client()))->sync();

        self::assertCount(1, $this->router->received());
        self::assertSame(
            [['url' => 'https://example.com/plans'], ['url' => 'https://example.com/faq', 'title' => 'FAQ']],
            $this->router->to('POST', '/v1/agents/sync')[0]->json()['knowledge_urls'],
        );
    }

    public function testCostTrackingChangesTheHashTheWayGoDoes(): void
    {
        $this->folder();
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);

        (new Agent(folder: $this->dir, costTracking: ['team' => 'a', 'env' => 'b'], client: $this->router->client()))->sync();

        $expected = Folder::fingerprint('02a7b2c8428f31e3a2b93ca2f5a6ec70', '', 'map[env:b team:a]');
        self::assertSame($expected, $this->router->to('POST', '/v1/agents/sync')[0]->json()['hash']);
    }

    private function edge(): Edge
    {
        return new Edge('key', str_repeat('s', 32), $this->router->url);
    }

    private function folder(): void
    {
        $this->dir = sys_get_temp_dir() . '/agent-' . bin2hex(random_bytes(6)) . '/jean';
        foreach ([
            'agent.yaml' => "name: jean\nllm: openai/gpt-5.6\n",
            'instructions.md' => "You are Jean.\n",
            'skills/think.md' => "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n",
            'knowledge/pricing.md' => "# Pricing\n\nA penny.\n",
        ] as $relative => $contents) {
            $path = $this->dir . '/' . $relative;
            if (!is_dir(dirname($path))) {
                mkdir(dirname($path), 0o777, true);
            }
            file_put_contents($path, $contents);
        }
    }
}
