<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Folder;
use GetStream\VisionAgents\Generated\GreetingMode;
use GetStream\VisionAgents\Json;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

final class FolderTest extends TestCase
{
    private string $dir;

    protected function setUp(): void
    {
        $this->dir = sys_get_temp_dir() . '/folder-' . bin2hex(random_bytes(6)) . '/jean';
        $this->write('agent.yaml', "name: jean\nllm: openai/gpt-5.6\n");
        $this->write('instructions.md', "You are Jean.\n");
        $this->write('skills/think.md', "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n");
        $this->write('knowledge/pricing.md', "# Pricing\n\nA penny.\n");
    }

    protected function tearDown(): void
    {
        exec('rm -rf ' . escapeshellarg(dirname($this->dir)));
    }

    public function testHashMatchesTheGoSdk(): void
    {
        self::assertSame('02a7b2c8428f31e3a2b93ca2f5a6ec70', Folder::load($this->dir)->hash());
    }

    public function testAKnowledgeUrlChangesTheHash(): void
    {
        $this->write('knowledge/urls.yaml', "- https://example.com/plans\n");
        $folder = Folder::load($this->dir);

        self::assertNotSame('02a7b2c8428f31e3a2b93ca2f5a6ec70', $folder->hash());
        self::assertSame('https://example.com/plans', $folder->knowledgeUrls[0]->url);
        self::assertCount(1, $folder->knowledge, 'urls.yaml is a declaration, not a document');
    }

    public function testARefreshScheduleHashesTheWayGoDoes(): void
    {
        $this->write('knowledge/urls.yaml', "- url: https://example.com/plans\n  refresh_hours: 24\n");

        $folder = Folder::load($this->dir);

        self::assertSame(24, $folder->knowledgeUrls[0]->refreshHours);
        self::assertSame('eb700759324810e0cf80e9ecdc7e3d4a', $folder->hash());
    }

    public function testSimulationsHashTheWayGoDoes(): void
    {
        $this->write('knowledge/urls.yaml', "- url: https://example.com/plans\n  refresh_hours: 24\n");
        $this->write('simulations/lunch.yaml', <<<'YAML'
            - name: lunch <&> order
              scenario: "Order a turkey club, then swap it for a veggie wrap/ café."
              assertion: The final order is one veggie wrap.
              variations: 3
              tags:
                team: b
                env: a
            - name: dinner
              scenario: Order soup.
              assertion: One soup.
              mode: audio

            YAML);

        $folder = Folder::load($this->dir);

        self::assertSame(['lunch <&> order', 'dinner'], array_map(static fn ($simulation) => $simulation->name, $folder->simulations ?? []));
        self::assertSame('c3bba1b0ab05b6408805ba25511ffbf7', $folder->hash());
    }

    public function testAnEmptySimulationsDirectoryIsAnEmptyListAndHashesTheWayGoDoes(): void
    {
        mkdir($this->dir . '/simulations');

        $folder = Folder::load($this->dir);

        self::assertSame([], $folder->simulations);
        self::assertSame('06272dda88821ce516e631787b335205', $folder->hash());
    }

    public function testNoSimulationsDirectoryIsNull(): void
    {
        self::assertNull(Folder::load($this->dir)->simulations);
    }

    public function testReadsGreetingSubagentAndHarness(): void
    {
        $this->write('agent.yaml', "greeting:\n  text: Hello there.\n  mode: exact\nsubagent: openai/gpt-5.6\nharness: default\n");

        $settings = Folder::load($this->dir)->settings;

        self::assertSame('Hello there.', $settings->greeting?->text);
        self::assertSame(GreetingMode::Exact, $settings->greeting->mode);
        self::assertSame('openai/gpt-5.6', $settings->subagent);
        self::assertSame('default', $settings->harness);
    }

    /**
     * @return iterable<string, array{string, string}>
     */
    public static function refusedFiles(): iterable
    {
        yield 'refresh_hours of zero' => ['knowledge/urls.yaml', "- url: https://example.com/plans\n  refresh_hours: 0\n"];
        yield 'refresh_hours not a whole number' => ['knowledge/urls.yaml', "- url: https://example.com/plans\n  refresh_hours: 1.5\n"];
        yield 'unknown page key' => ['knowledge/urls.yaml', "- url: https://example.com/plans\n  refresh: 24\n"];
        yield 'unknown simulation key' => ['simulations/a.yaml', "- name: a\n  scenario: s\n  assertion: x\n  turns: 3\n"];
        yield 'simulation without an assertion' => ['simulations/a.yaml', "- name: a\n  scenario: s\n"];
        yield 'simulation in an unknown mode' => ['simulations/a.yaml', "- name: a\n  scenario: s\n  assertion: x\n  mode: video\n"];
        yield 'simulations file not a list' => ['simulations/a.yaml', "name: a\n"];
        yield 'speed, which is gone' => ['agent.yaml', "speed: 1.1\n"];
        yield 'greeting as a string' => ['agent.yaml', "greeting: Hello there.\n"];
        yield 'unknown greeting key' => ['agent.yaml', "greeting:\n  text: Hello there.\n  voice: ash\n"];
    }

    #[DataProvider('refusedFiles')]
    public function testRefusesWhatGoRefusesInTheRestOfTheDirectory(string $file, string $contents): void
    {
        $this->write($file, $contents);

        $this->expectException(ConfigurationException::class);
        Folder::load($this->dir);
    }

    public function testASimulationNameIsUniqueAcrossFiles(): void
    {
        $this->write('simulations/a.yaml', "- name: lunch\n  scenario: s\n  assertion: x\n");
        $this->write('simulations/b.yml', "- name: lunch\n  scenario: s\n  assertion: x\n");

        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('also declared in a.yaml');
        Folder::load($this->dir);
    }

    public function testReadsTheDirectory(): void
    {
        $folder = Folder::load($this->dir);

        self::assertSame('jean', $folder->name);
        self::assertSame('openai/gpt-5.6', $folder->settings->llm);
        self::assertSame('You are Jean.', $folder->instructions);
        self::assertSame('think', $folder->skills[0]->name);
        self::assertSame('Work it out', $folder->skills[0]->description);
        self::assertSame('Reason it through.', $folder->skills[0]->instructions);
        self::assertSame(30.0, $folder->skills[0]->deadline);
        self::assertSame('pricing.md', $folder->knowledge[0]->source);
    }

    /**
     * @return iterable<string, array{string}>
     */
    public static function refused(): iterable
    {
        yield 'misspelt key' => ["name: jean\nlmm: openai/gpt-5.6\n"];
        yield 'too many frames' => ["video:\n  max_frames: 9\n"];
        yield 'keyterms not a list' => ["keyterms: Vision Agents\n"];
        yield 'unknown dispatch key' => ["dispatch:\n  outgoing_call: enabled\n"];
    }

    public function testReadsDispatchAsWritten(): void
    {
        $this->write('agent.yaml', "dispatch:\n  incoming_call: enabled\n  text: disabled\n");

        $dispatch = Folder::load($this->dir)->settings->dispatch;

        self::assertSame(['incoming_call' => 'enabled', 'text' => 'disabled'], $dispatch?->toArray());
    }

    #[DataProvider('refused')]
    public function testRefusesWhatGoRefuses(string $declaration): void
    {
        $this->write('agent.yaml', $declaration);

        $this->expectException(ConfigurationException::class);
        Folder::load($this->dir);
    }

    public function testStampRoundTrips(): void
    {
        $folder = Folder::load($this->dir);
        self::assertSame('', $folder->stamp());

        $folder->writeStamp($folder->hash());

        self::assertSame($folder->hash(), $folder->stamp());
        $written = Json::asObject(Json::decode((string) file_get_contents($this->dir . '/.agent_sync')));
        self::assertSame(['hash', 'synced_at'], array_keys($written));
        self::assertMatchesRegularExpression('/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00$/', Json::string($written, 'synced_at'));
    }

    public function testNeedsAgentYaml(): void
    {
        unlink($this->dir . '/agent.yaml');

        $this->expectException(ConfigurationException::class);
        Folder::load($this->dir);
    }

    private function write(string $relative, string $contents): void
    {
        $path = $this->dir . '/' . $relative;
        if (!is_dir(dirname($path))) {
            mkdir(dirname($path), 0o777, true);
        }
        file_put_contents($path, $contents);
    }
}
