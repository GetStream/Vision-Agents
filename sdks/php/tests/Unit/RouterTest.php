<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Exception\RecordingFailedException;
use GetStream\VisionAgents\Generated\KnowledgeUrlState;
use GetStream\VisionAgents\Generated\RecordingStatus;
use GetStream\VisionAgents\Generated\SearchOptions;
use GetStream\VisionAgents\Generated\SttOptions;
use GetStream\VisionAgents\Generated\TtsOptions;
use GetStream\VisionAgents\Knowledge;
use GetStream\VisionAgents\Router;
use GetStream\VisionAgents\Tests\Support\LocalRouter;
use GetStream\VisionAgents\Tests\Support\Rows;
use PHPUnit\Framework\TestCase;

final class RouterTest extends TestCase
{
    private LocalRouter $local;
    private Router $router;

    protected function setUp(): void
    {
        $this->local = new LocalRouter();
        $this->router = new Router('healthcare', ['team' => 'clinical'], $this->local->client());
    }

    protected function tearDown(): void
    {
        $this->local->stop();
    }

    public function testSearch(): void
    {
        $this->local->answer('POST', '/v1/search', 200, ['provider' => 'exa', 'model' => 'exa', 'results' => [['url' => 'https://example.com', 'title' => 'Guidance']]]);

        $answer = $this->router->search('perioperative antibiotic guidance', new SearchOptions(results: 3));

        self::assertSame('Guidance', $answer->results[0]->title);
        self::assertSame(
            ['query' => 'perioperative antibiotic guidance', 'config_id' => 'healthcare', 'options' => ['results' => 3], 'tags' => ['team' => 'clinical']],
            $this->local->received()[0]->json(),
        );
    }

    public function testTranscriptionWaitsForTheJob(): void
    {
        $this->local->answer('POST', '/v1/stt/recordings', 202, Rows::job('queued'));
        $this->local->answer('GET', '/v1/stt/recordings/job_1', 200, Rows::job('running'), Rows::job('completed', ['text' => 'hello there']));

        $done = $this->router->stt->recording('https://example.com/visit.mp3', new SttOptions(diarize: true));

        self::assertSame(RecordingStatus::Completed, $done->status);
        self::assertSame('hello there', $done->text);
        self::assertCount(2, $this->local->to('GET', '/v1/stt/recordings/job_1'));
        $sent = $this->local->to('POST', '/v1/stt/recordings')[0]->json();
        self::assertSame(['url' => 'https://example.com/visit.mp3'], $sent['source']);
        self::assertSame(['diarize' => true], $sent['options']);
    }

    public function testAFailedJobRaises(): void
    {
        $this->local->answer('POST', '/v1/tts/recordings', 202, Rows::job('failed', ['error' => 'every provider refused']));

        $this->expectException(RecordingFailedException::class);
        $this->expectExceptionMessage('every provider refused');
        $this->router->tts->recording('Chapter one.', new TtsOptions(voice: 'custom:reader'));
    }

    public function testACallbackReturnsTheAcceptedJob(): void
    {
        $this->local->answer('POST', '/v1/tts/recordings', 202, Rows::job('queued'));

        $job = $this->router->tts->recording('Chapter one.', callback: 'https://example.com/done');

        self::assertSame(RecordingStatus::Queued, $job->status);
        self::assertSame('https://example.com/done', $this->local->received()[0]->json()['callback']);
    }

    public function testConfiguringCarriesTheOtherModalitiesForward(): void
    {
        $stored = ['id' => 'rc_1', 'name' => 'healthcare', 'created_at' => Rows::AT, 'updated_at' => Rows::AT, 'tts' => ['voice' => 'Kore']];
        $this->local->answer('GET', '/v1/router/configs', 200, [$stored]);
        $this->local->answer('PUT', '/v1/router/configs/rc_1', 200, $stored);

        $this->router->configureStt(new SttOptions(providers: ['deepgram']));

        self::assertSame(
            ['name' => 'healthcare', 'stt' => ['providers' => ['deepgram']], 'tts' => ['voice' => 'Kore']],
            $this->local->to('PUT', '/v1/router/configs/rc_1')[0]->json(),
        );
    }

    public function testKnowledgeAddUrlWaitsForTheRead(): void
    {
        $this->local->answer('POST', '/v1/agents/knowledge/urls', 202, Rows::page('pending'));
        $this->local->answer('GET', '/v1/agents/knowledge/urls/kurl_1', 200, Rows::page('pending'), Rows::page('indexed', 4));
        $knowledge = (new Agent(name: 'jean', client: $this->local->client()))->knowledge();

        $page = $knowledge->addUrl('https://example.com/plans', title: 'Plans');

        self::assertSame(KnowledgeUrlState::Indexed, $page->state);
        self::assertSame(4, $page->passages);
        self::assertSame(
            ['namespace' => 'jean', 'url' => 'https://example.com/plans', 'title' => 'Plans'],
            $this->local->to('POST', '/v1/agents/knowledge/urls')[0]->json(),
        );
    }

    public function testKnowledgeReturnsAPendingPageAtTheTimeout(): void
    {
        $this->local->answer('POST', '/v1/agents/knowledge/urls', 202, Rows::page('pending'));
        $this->local->answer('GET', '/v1/agents/knowledge/urls/kurl_1', 200, Rows::page('pending'));

        $page = (new Knowledge($this->local->client(), 'jean', timeout: 0.3, pollEvery: 0.1))->addUrl('https://example.com/plans');

        self::assertSame(KnowledgeUrlState::Pending, $page->state);
    }
}
