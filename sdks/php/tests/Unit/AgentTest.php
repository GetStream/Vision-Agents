<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Edge;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Folder;
use GetStream\VisionAgents\Generated\AgentConfigPatch;
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
        self::assertArrayNotHasKey('start_voice', $sent);
        self::assertSame('support', $sent['user_id']);
        self::assertSame('support', $sent['agent_id']);
        self::assertArrayNotHasKey('instructions', $sent, 'instructions reach the router through sync');
        self::assertSame('openai/gpt-5.6', $sent['llm']);
        self::assertSame(['fr'], $sent['languages']);
        self::assertSame(['env' => 'production'], $sent['tags']);
        self::assertSame(['filter' => ['topic' => 'billing'], 'user_id' => '123'], $sent['memory']);
        foreach (['sandbox', 'skills', 'subagent', 'tasks'] as $harness) {
            self::assertArrayNotHasKey($harness, $sent, 'the harness is the config\'s, written by sync');
        }
        self::assertArrayNotHasKey('stt', $sent, 'what was not set is left to the config');
    }

    public function testSyncWritesTheHarnessOntoTheConfig(): void
    {
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);
        $agent = new Agent(
            name: 'jean',
            harness: new Harness(name: 'default', subagent: 'openai/gpt-5.6', skills: [new Skill('think', 'Work it out', 'Reason it through.', deadline: 30.0)]),
            sandbox: Sandbox::Daytona,
            client: $this->router->client(),
        );

        $agent->sync();

        $sent = $this->router->to('POST', '/v1/agents/sync')[0]->json();
        self::assertSame('default', $sent['harness']);
        self::assertSame('openai/gpt-5.6', $sent['subagent']);
        self::assertSame('daytona', $sent['sandbox']);
        self::assertSame(
            [['config_id' => '', 'description' => 'Work it out', 'instructions' => 'Reason it through.', 'name' => 'think', 'deadline_ms' => 30000]],
            array_map(static fn (mixed $skill): array => array_intersect_key(Json::asObject($skill), array_flip(['config_id', 'name', 'description', 'instructions', 'deadline_ms'])), Json::list($sent, 'skills')),
        );
    }

    public function testUpdateConfigPatchesTheConfigFoundByName(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, [Rows::config('cfg_9', 'support')]);
        $this->router->answer('PATCH', '/v1/agents/configs/cfg_9', 200, Rows::config('cfg_9', 'support') + ['guardrail' => 'No refunds.']);

        $config = $this->router->client()->agent('support')->updateConfig(new AgentConfigPatch(guardrail: 'No refunds.', visibleTools: ['athena_*']));

        self::assertSame('No refunds.', $config->guardrail);
        self::assertSame('support', $this->router->to('GET', '/v1/agents/configs')[0]->params()['name']);
        self::assertSame(
            ['guardrail' => 'No refunds.', 'visible_tools' => ['athena_*']],
            $this->router->to('PATCH', '/v1/agents/configs/cfg_9')[0]->json(),
        );
    }

    public function testUpdateConfigOfAnAgentNothingIsStoredUnderIsRefused(): void
    {
        $this->router->answer('GET', '/v1/agents/configs', 200, []);

        $this->expectException(ConfigurationException::class);
        $this->router->client()->agent('nobody')->updateConfig(new AgentConfigPatch(guardrail: 'No refunds.'));
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
        self::assertSame([['description' => 'The weather somewhere', 'name' => 'get_weather', 'parameters' => ['type' => 'object']]], $sent['tools']);
    }

    public function testAToolSaysWhoRunsItAndWhatItIsShownAs(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());
        $agent->tools->register('take_photo', 'Takes a photo on the device', [], static fn (array $args): string => '', displayTitle: 'Taking a photo', executor: 'client');

        $agent->chat();

        self::assertSame(
            [['description' => 'Takes a photo on the device', 'name' => 'take_photo', 'display_title' => 'Taking a photo', 'executor' => 'client']],
            $this->router->to('POST', '/v1/agents/sessions')[0]->json()['tools'],
        );
    }

    public function testJoinStartsVoiceOnTheSessionsOwnCall(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $session = $agent->join();

        self::assertSame(['/v1/agents/sessions'], array_map(static fn ($r) => $r->path, $this->router->received()), 'no Stream call is created first');
        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertTrue($sent['start_voice']);
        self::assertArrayNotHasKey('call_id', $sent);
        self::assertSame('ses_1', $session->call()->id);
        self::assertSame('agent', $session->call()->type);
    }

    public function testMonitorUrlNamesTheSessionsCallOnceVoiceIsStarted(): void
    {
        $this->router->answer('POST', '/v1/agents/sessions/ses_1/voice', 200, Rows::session('ses_1', ['call_id' => 'ses_1', 'call_type' => 'agent']));
        $agent = new Agent(name: 'jean', client: $this->router->client(), edge: $this->edge());
        $session = $agent->chat();
        $session->startVoice();

        $url = $agent->monitorUrl($session);

        self::assertStringContainsString('/join/ses_1?', $url);
    }

    public function testASessionHeldInWritingHasNoMonitorUrl(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client(), edge: $this->edge());

        $this->expectException(ConfigurationException::class);
        $agent->monitorUrl($agent->chat());
    }

    public function testResumeCarriesOnTheSessionById(): void
    {
        $this->router->answer('GET', '/v1/agents/sessions/ses_7', 200, Rows::session('ses_7'));
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $session = $agent->resume('ses_7');

        self::assertSame('ses_7', $session->id());
        self::assertSame($agent, $session->agent);
        self::assertSame([], $this->router->to('POST', '/v1/agents/sessions'));
    }

    public function testOutboundCallJoinsTheSessionThePlacedCallNames(): void
    {
        $this->router->answer('POST', '/v1/phone/calls', 201, ['vendor_call_id' => 'CA123', 'status' => 'queued', 'session_id' => 'ses_out']);
        $agent = new Agent(name: 'jean', costTracking: ['team' => 'sales'], client: $this->router->client());

        $agent->outboundCall('+15550001111', '+15552223333');

        $paths = array_map(static fn ($r) => $r->path, $this->router->received());
        self::assertSame(['/v1/phone/calls', '/v1/agents/sessions'], $paths);
        self::assertSame(
            ['from' => '+15550001111', 'to' => '+15552223333', 'tags' => ['team' => 'sales']],
            $this->router->to('POST', '/v1/phone/calls')[0]->json(),
        );
        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertSame('ses_out', $sent['id']);
        self::assertTrue($sent['start_voice']);
        self::assertTrue($sent['navigating']);
        self::assertSame(['number' => '+15550001111', 'vendor_call_id' => 'CA123'], $sent['phone']);
    }

    public function testAnswerJoinsTheSessionTheCallNames(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $agent->answer(InboundCall::fromFrame(['call_id' => 'ses_in', 'session_id' => 'ses_in', 'called_number' => '+15550001111']));

        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertSame('ses_in', $sent['id']);
        self::assertTrue($sent['start_voice']);
        self::assertArrayNotHasKey('call_id', $sent);
        self::assertSame(['number' => '+15550001111'], $sent['phone']);
    }

    public function testACallNamingNoSessionIsRefused(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $this->expectException(ConfigurationException::class);
        $agent->answer(InboundCall::fromFrame(['call_id' => 'c1']));
    }

    public function testReplyAnswersInTheAgentsConversation(): void
    {
        $agent = new Agent(name: 'jean', client: $this->router->client());

        $agent->reply(InboundMessage::fromFrame(['channel_id' => 'ch1', 'text' => 'hi', 'agent_id' => 'jean-7']));

        $sent = $this->router->to('POST', '/v1/agents/sessions')[0]->json();
        self::assertArrayNotHasKey('incognito', $sent);
        self::assertArrayNotHasKey('conversation_id', $sent);
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

    public function testSyncSendsOnlyTheDispatchSettingsWritten(): void
    {
        $this->folder();
        file_put_contents($this->dir . '/agent.yaml', "name: jean\ndispatch:\n  text: enabled\n");
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);

        (new Agent(folder: $this->dir, client: $this->router->client()))->sync();

        self::assertSame(['text' => 'enabled'], $this->router->to('POST', '/v1/agents/sync')[0]->json()['dispatch']);
    }

    public function testSyncLeavesOutWhatAgentYamlSaysNothingAbout(): void
    {
        $this->folder();
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);

        (new Agent(folder: $this->dir, client: $this->router->client()))->sync();

        $sent = $this->router->to('POST', '/v1/agents/sync')[0]->json();
        foreach (['dispatch', 'greeting', 'harness', 'simulations'] as $absent) {
            self::assertArrayNotHasKey($absent, $sent);
        }
    }

    public function testSyncSendsGreetingPluginsHarnessAndSchedules(): void
    {
        $this->folder();
        file_put_contents($this->dir . '/agent.yaml', "name: jean\ngreeting:\n  text: Hello there.\n  mode: variation\nplugins: [sentry]\nharness: default\n");
        file_put_contents($this->dir . '/knowledge/urls.yaml', "- url: https://example.com/plans\n  refresh_hours: 24\n");
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);

        (new Agent(folder: $this->dir, client: $this->router->client()))->sync();

        $sent = $this->router->to('POST', '/v1/agents/sync')[0]->json();
        self::assertSame(['text' => 'Hello there.', 'mode' => 'variation'], $sent['greeting']);
        self::assertSame(['sentry'], $sent['plugins']);
        self::assertArrayNotHasKey('agent_plugins', $sent);
        self::assertSame('default', $sent['harness']);
        self::assertSame([['url' => 'https://example.com/plans', 'refresh_hours' => 24]], $sent['knowledge_urls']);
    }

    public function testSyncSendsTheSimulationsDirectoryEvenWhenEmpty(): void
    {
        $this->folder();
        mkdir($this->dir . '/simulations');
        $this->router->answer('POST', '/v1/agents/sync', 200, ['unchanged' => false, 'config' => Rows::config('cfg_1', 'jean')]);

        (new Agent(folder: $this->dir, client: $this->router->client()))->sync();
        file_put_contents($this->dir . '/simulations/lunch.yaml', "- name: lunch\n  scenario: Order a club.\n  assertion: One club.\n  variations: 3\n");
        (new Agent(folder: $this->dir, client: $this->router->client()))->sync();

        [$empty, $declared] = $this->router->to('POST', '/v1/agents/sync');
        self::assertSame([], $empty->json()['simulations'], 'an empty directory deletes the stored simulations');
        self::assertSame([['assertion' => 'One club.', 'name' => 'lunch', 'scenario' => 'Order a club.', 'variations' => 3]], $declared->json()['simulations']);
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
