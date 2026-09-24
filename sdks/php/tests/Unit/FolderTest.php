<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Unit;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Folder;
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
