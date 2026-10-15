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

    public function testASkillFolderHashesTheWayGoDoes(): void
    {
        $this->write('agent.yaml', "name: jean\nllm: openai/gpt-5.6\nsandbox: daytona\n");
        $this->write('skills/pdf/SKILL.md', "---\ndescription: Fill PDF forms\ndeadline: 2m\n---\nFill the form.\n");
        $this->write('skills/pdf/reference/forms.md', "# Forms\n");
        $this->write('skills/pdf/scripts/fill.py', "print('filled')\n");
        $this->write('skills/pdf/scripts/lib/fill_lib.py', "def fill():\n    return 1\n");
        $this->write('skills/pdf/scripts/__pycache__/fill.cpython-313.pyc', "\xcb\x0d\xff");
        $this->write('skills/pdf/.DS_Store', "\x00\xff");

        $folder = Folder::load($this->dir);

        self::assertSame(['pdf', 'think'], array_map(static fn ($skill) => $skill->name, $folder->skills));
        self::assertSame(120.0, $folder->skills[0]->deadline);
        self::assertSame(
            ['reference/forms.md' => "# Forms\n", 'scripts/fill.py' => "print('filled')\n", 'scripts/lib/fill_lib.py' => "def fill():\n    return 1\n"],
            $folder->skills[0]->files,
        );
        self::assertSame([], $folder->skills[1]->files);
        self::assertSame('e47f7807b97bbcbdee683c6c481d1d36', $folder->hash());
    }

    public function testMovingASkillIntoAFolderKeepsTheHashAndAFileChangesIt(): void
    {
        $before = Folder::load($this->dir)->hash();
        $this->write('skills/think/SKILL.md', (string) file_get_contents($this->dir . '/skills/think.md'));
        unlink($this->dir . '/skills/think.md');

        self::assertSame($before, Folder::load($this->dir)->hash());

        $this->write('skills/think/notes.md', "Check twice.\n");

        self::assertNotSame($before, Folder::load($this->dir)->hash());
    }

    public function testASkillFolderNeedsSkillMd(): void
    {
        $this->write('skills/pdf/reference/forms.md', "# Forms\n");

        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('skills/pdf has no SKILL.md, so it is not a skill');
        Folder::load($this->dir);
    }

    public function testASkillIsEitherAFileOrAFolder(): void
    {
        $this->write('skills/think/SKILL.md', "---\ndescription: Work it out\n---\nReason it through.\n");

        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('skill "think" is both think and think.md');
        Folder::load($this->dir);
    }

    public function testASkillBringsOnlyTextFiles(): void
    {
        $this->write('skills/pdf/SKILL.md', "---\ndescription: Fill PDF forms\n---\nFill the form.\n");
        $this->write('skills/pdf/logo.png', "\x89PNG\xff");

        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage('logo.png is not text, and a skill brings only text files');
        Folder::load($this->dir);
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
        yield 'unknown sandbox_options key' => ["sandbox_options:\n  memory: 4\n"];
        yield 'sandbox timeout past 30m' => ["sandbox_options:\n  timeout: 2h\n"];
        yield 'sandbox size that is not a number' => ["sandbox_options:\n  cpu: two\n"];
        yield 'sandbox network that is not a boolean' => ["sandbox_options:\n  network: maybe\n"];
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

    /**
     * @return iterable<string, array{string, string}>
     */
    public static function refusedConnectors(): iterable
    {
        yield 'null entry' => ["connectors:\n  - \n", 'a connector needs a name'];
        yield 'empty name' => ["connectors:\n  - name: ''\n", 'a connector needs a name'];
        yield 'mapping without a name' => ["connectors:\n  - connector: linear\n", 'a connector needs a name'];
        yield 'empty tools' => ["connectors:\n  - name: linear\n    tools: []\n", 'connector "linear": tools: [] grants nothing; leave tools out for every tool'];
        yield 'connection as a map' => ["connectors:\n  - name: linear\n    connection:\n      id: conn_a\n", 'connector "linear": connection must be a connection id or label'];
        yield 'empty connection' => ["connectors:\n  - name: linear\n    connection: ''\n", 'connector "linear": connection must be a connection id or label'];
        yield 'unknown key' => ["connectors:\n  - name: linear\n    tool: [search]\n", '"connectors.tool" is not something agent.yaml declares'];
        yield 'not a list' => ["connectors: linear\n", 'connectors must be a list'];
        yield 'alias of a plugin' => ["plugins: [linear]\nconnectors: [linear]\n", '"linear" is named under both connectors and plugins'];
    }

    #[DataProvider('refusedConnectors')]
    public function testRefusesConnectorsTheWayGoDoes(string $declaration, string $message): void
    {
        $this->write('agent.yaml', $declaration);

        $this->expectException(ConfigurationException::class);
        $this->expectExceptionMessage($message);
        Folder::load($this->dir);
    }

    public function testReadsConnectorsAsWritten(): void
    {
        $this->write('agent.yaml', "connectors:\n  - linear\n  - name: issues\n    connector: linear\n    tools: [search]\n    connection: prod\n    required: false\n");

        $connectors = Folder::load($this->dir)->settings->connectors;

        self::assertSame(['linear', 'issues'], array_map(static fn ($each) => $each->name, $connectors ?? []));
        self::assertSame([null, ['search']], array_map(static fn ($each) => $each->tools, $connectors ?? []));
        self::assertSame(['', 'linear'], array_map(static fn ($each) => $each->connector, $connectors ?? []));
        self::assertSame([null, 'prod'], array_map(static fn ($each) => $each->connection, $connectors ?? []));
        self::assertSame([null, false], array_map(static fn ($each) => $each->required, $connectors ?? []));
    }

    public function testConnectorsLeftOutAreNullAndAnEmptyListIsEmpty(): void
    {
        self::assertNull(Folder::load($this->dir)->settings->connectors);

        $this->write('agent.yaml', "connectors: []\n");

        self::assertSame([], Folder::load($this->dir)->settings->connectors);
    }

    public function testConnectorsAreHashedAsWritten(): void
    {
        $before = Folder::load($this->dir)->hash();
        $this->write('agent.yaml', "name: jean\nllm: openai/gpt-5.6\nconnectors: [linear]\n");

        self::assertNotSame($before, Folder::load($this->dir)->hash());
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
