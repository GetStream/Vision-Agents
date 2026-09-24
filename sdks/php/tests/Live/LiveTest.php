<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Tests\Live;

use GetStream\VisionAgents\Agent;
use GetStream\VisionAgents\Backend;
use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Folder;
use GetStream\VisionAgents\Generated\ForkSessionRequest;
use GetStream\VisionAgents\Generated\KnowledgeUrlState;
use GetStream\VisionAgents\Router;
use GetStream\VisionAgents\Session;
use GetStream\VisionAgents\Worker\Dispatch;
use PHPUnit\Framework\TestCase;
use Revolt\EventLoop;

/**
 * Against a running router. Skipped unless VISION_AGENTS_URL names one; the customer is
 * VISION_AGENTS_CUSTOMER_ID, `examples` by default.
 */
final class LiveTest extends TestCase
{
    private Client $client;
    /** @var list<Session> */
    private array $opened = [];
    private string $dir = '';

    protected function setUp(): void
    {
        $url = getenv('VISION_AGENTS_URL');
        if (!is_string($url) || $url === '') {
            self::markTestSkipped('set VISION_AGENTS_URL to run against a router');
        }
        $customer = getenv('VISION_AGENTS_CUSTOMER_ID');
        $this->client = new Client(new Backend(url: $url, customerId: is_string($customer) && $customer !== '' ? $customer : 'examples'));
    }

    protected function tearDown(): void
    {
        foreach ($this->opened as $session) {
            $session->close();
        }
        if ($this->dir !== '') {
            exec('rm -rf ' . escapeshellarg(dirname($this->dir)));
        }
    }

    public function testResponsesRewindAndFork(): void
    {
        $agent = new Agent(name: 'php-sdk-live', instructions: 'Answer in at most five words.', client: $this->client);
        $session = $this->open($agent->chat(title: 'php sdk live'));

        $first = $session->responses->create('Name a colour.');
        $this->settle($session);
        $second = $session->responses->create('Name an animal.');
        $this->settle($session);

        self::assertNotSame('', $first->id());
        self::assertNotSame($first->id(), $second->id());
        self::assertNotEmpty($second->items->all(), 'a finished response has items');
        self::assertCount(2, $session->responses->list());

        $session->responses->rewind($first);
        self::assertSame([$first->id()], array_map(static fn ($row) => $row->id, $session->responses->list()));

        $fork = $this->open($session->fork(new ForkSessionRequest(responseId: $first->id(), title: 'php sdk fork')));
        self::assertNotSame($session->id(), $fork->id());
    }

    public function testAPersistedConversationIsForkedNotRewound(): void
    {
        $agent = new Agent(name: 'php-sdk-live', instructions: 'Answer in at most five words.', client: $this->client);
        $session = $this->open($agent->chat(persist: true));
        $response = $session->responses->create('Name a colour.');
        $this->settle($session);

        try {
            $session->responses->rewind($response);
            self::fail('a persisted conversation was rewound');
        } catch (RouterException $refused) {
            self::assertSame(400, $refused->status);
            self::assertStringContainsString('fork', $refused->said);
        }
    }

    public function testWatchSeesTheReply(): void
    {
        $agent = new Agent(name: 'php-sdk-live', instructions: 'Answer in at most five words.', client: $this->client);
        $session = $this->open($agent->chat());
        $watch = $session->watch();
        $watch->respond('Name a fruit.');

        $kinds = [];
        foreach ($watch as $event) {
            $kinds[] = $event->kind;
            if ($event->kind === 'responded' || $event->kind === 'error') {
                break;
            }
        }
        $watch->close();

        self::assertContains('responded', $kinds);
    }

    public function testFolderSyncAndStamp(): void
    {
        $this->dir = sys_get_temp_dir() . '/live-' . bin2hex(random_bytes(6)) . '/php-sdk-live';
        mkdir($this->dir . '/skills', 0o777, true);
        mkdir($this->dir . '/knowledge');
        file_put_contents($this->dir . '/agent.yaml', "name: php-sdk-live\nmode: text\n");
        file_put_contents($this->dir . '/instructions.md', "You are the PHP SDK's live test.\n");
        file_put_contents($this->dir . '/skills/think.md', "---\ndescription: Work it out\ndeadline: 30s\n---\nReason it through.\n");
        file_put_contents($this->dir . '/knowledge/pricing.md', "# Pricing\n\nA penny.\n");
        file_put_contents($this->dir . '/knowledge/urls.yaml', "- https://example.com/\n");

        $first = (new Agent(folder: $this->dir, client: $this->client))->sync();
        $second = (new Agent(folder: $this->dir, client: $this->client))->sync();

        self::assertSame('php-sdk-live', $first->config->name);
        self::assertTrue($second->unchanged);
        self::assertSame($first->config->id, $second->config->id);
        self::assertSame(Folder::load($this->dir)->hash(), Folder::load($this->dir)->stamp());
        self::assertSame($first->config->id, $this->client->agent('php-sdk-live')->config()?->id);
    }

    public function testSearch(): void
    {
        $answer = (new Router(tags: ['sdk' => 'php'], client: $this->client))->search('What is the capital of France?');

        self::assertNotSame('', $answer->provider);
        self::assertNotEmpty($answer->results);
    }

    public function testKnowledgeAddUrl(): void
    {
        $page = (new Agent(name: 'php-sdk-live', client: $this->client))->knowledge()->addUrl('https://example.com/', title: 'Example');

        self::assertSame('php-sdk-live', $page->namespace);
        self::assertSame(KnowledgeUrlState::Indexed, $page->state, (string) $page->error);
    }

    public function testGuests(): void
    {
        $guest = $this->client->guestUser(name: 'PHP Guest');

        self::assertNotSame('', $guest->id);
        self::assertNotSame('', $guest->token);

        $claimed = $this->client->claimGuestUser($guest, 'php-sdk-live-user');
        self::assertSame($guest->id, $claimed->guestId);
    }

    public function testDispatchIsHandedAWorkerId(): void
    {
        $dispatch = new Dispatch(capacity: 1, client: $this->client);
        $dispatch->waitForCall(static fn () => null);
        EventLoop::delay(1.5, $dispatch->stop(...));

        $dispatch->run();

        self::assertNotSame('', $dispatch->workerId);
    }

    /**
     * A response is created running and finishes on its own; this waits for all of them.
     */
    private function settle(Session $session): void
    {
        $deadline = microtime(true) + 60;
        while (microtime(true) < $deadline) {
            $running = array_filter($session->responses->list(), static fn ($row): bool => $row->status === 'running');
            if ($running === []) {
                return;
            }
            usleep(250_000);
        }
        self::fail('the response never finished');
    }

    private function open(Session $session): Session
    {
        $this->opened[] = $session;
        return $session;
    }
}
