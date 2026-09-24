<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\AgentResponseItem;
use Generator;

/**
 * The things turns were made of, in the order they happened.
 *
 * Read rather than watched: this is what the backend wrote down, so it is the same whether the
 * conversation is still going or ended last week. Deltas are not here, since a hundred
 * fragments of one sentence are the sentence.
 */
final readonly class Items
{
    private const int PAGE = 200;

    /**
     * @param string $responseId empty for every turn in the session, set for one response's own
     */
    public function __construct(private Client $client, private string $sessionId, private string $responseId = '')
    {
    }

    /**
     * Every item, oldest first, fetched a page at a time as the loop reaches it.
     *
     * @return Generator<int, AgentResponseItem>
     */
    public function unwind(int $page = self::PAGE): Generator
    {
        $page = max(1, min($page, 1000));
        $offset = 0;
        while (true) {
            $items = $this->list($page, $offset);
            yield from $items;
            // A short page is the last page.
            if (count($items) < $page) {
                return;
            }
            $offset += count($items);
        }
    }

    /**
     * One page, for a caller doing its own paging.
     *
     * @return list<AgentResponseItem>
     */
    public function list(?int $limit = null, ?int $offset = null): array
    {
        $listed = $this->client->get('/v1/agents/sessions/{id}/responses/items', ['id' => $this->sessionId], [
            'response_id' => $this->responseId === '' ? null : $this->responseId,
            'limit' => $limit,
            'offset' => $offset,
        ]);
        return array_map(AgentResponseItem::fromArray(...), Json::objects(['rows' => $listed], 'rows'));
    }

    /**
     * Everything in one array, for a conversation short enough to hold.
     *
     * @return list<AgentResponseItem>
     */
    public function all(): array
    {
        return iterator_to_array($this->unwind(), false);
    }
}
