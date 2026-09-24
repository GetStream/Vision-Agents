<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use DateTimeInterface;
use GetStream\VisionAgents\Generated\CreateSessionRequest;
use GetStream\VisionAgents\Generated\ModelOverwrites;
use GetStream\VisionAgents\Generated\Session as SessionRow;

/**
 * An agent's conversations: the one being held and the ones that were.
 *
 * `query` filters and pages; `search` reads the words a caller titled and described their
 * conversations with. Two methods, because a filter and a phrase combine by narrowing and there
 * is no sensible ranking of an empty phrase.
 */
final readonly class Sessions
{
    public function __construct(private Client $client, private string $agent)
    {
    }

    /**
     * Opens a conversation, held in writing unless a call is named.
     *
     * A field left null is left out, so the config decides it.
     *
     * @param array<string, mixed>|null $custom anything of the caller's own a later query can match on
     */
    public function create(
        ?string $title = null,
        ?string $description = null,
        ?string $project = null,
        ?array $custom = null,
        ?bool $incognito = null,
        ?bool $persist = null,
        ?string $conversationId = null,
        ?ModelOverwrites $modelOverwrites = null,
        ?string $callId = null,
        ?string $userId = null,
    ): Session {
        $request = new CreateSessionRequest(
            conversationId: $conversationId,
            persistConversation: $persist,
            callId: $callId,
            text: $callId === null ? true : null,
            agent: $this->agent,
            incognito: $incognito,
            title: $title,
            description: $description,
            project: $project,
            custom: $custom,
            modelOverwrites: $modelOverwrites,
            userId: $userId,
        );
        $created = $this->client->post('/v1/agents/sessions', body: $request->toArray());
        return new Session($this->client, SessionRow::fromArray(Json::asObject($created)));
    }

    /**
     * The agent's conversations, newest first, the ones that ended included. Rows rather than
     * live handles: reading a conversation back is not holding one.
     *
     * @param 'running'|'closed'|null $state
     * @param array<string, string|int|float|bool>|null $custom labels a session must carry, all of them
     * @return list<SessionRow>
     */
    public function query(
        ?string $project = null,
        ?string $userId = null,
        ?string $state = null,
        ?array $custom = null,
        ?DateTimeInterface $createdAfter = null,
        ?DateTimeInterface $createdBefore = null,
        ?int $limit = null,
        ?int $offset = null,
    ): array {
        return $this->rows('/v1/agents/sessions', self::filter($this->agent, $project, $userId, $state, $custom, $createdAfter, $createdBefore, $limit, $offset));
    }

    /**
     * Finds a conversation by what it was called: the title, description and project. Nothing
     * about an incognito session is searchable, because nothing about it was written down.
     *
     * @param 'running'|'closed'|null $state
     * @param array<string, string|int|float|bool>|null $custom
     * @return list<SessionRow>
     */
    public function search(
        string $text,
        ?string $project = null,
        ?string $userId = null,
        ?string $state = null,
        ?array $custom = null,
        ?int $limit = null,
        ?int $offset = null,
    ): array {
        $query = self::filter($this->agent, $project, $userId, $state, $custom, null, null, $limit, $offset);
        $query['q'] = $text;
        return $this->rows('/v1/agents/sessions/search', $query);
    }

    /**
     * One conversation, whether or not it is still being held.
     */
    public function get(string $id): Session
    {
        return new Session($this->client, SessionRow::fromArray(Json::asObject($this->client->get('/v1/agents/sessions/{id}', ['id' => $id]))));
    }

    /**
     * The turns of a conversation this process is not holding, which is most of them: a page
     * rendering last week's conversation has its id and no session.
     */
    public function responses(string $id): Responses
    {
        return new Responses($this->client, $id);
    }

    /**
     * @param array<string, scalar|null> $query
     * @return list<SessionRow>
     */
    private function rows(string $path, array $query): array
    {
        return array_map(SessionRow::fromArray(...), Json::objects(['rows' => $this->client->get($path, query: $query)], 'rows'));
    }

    /**
     * @param array<string, string|int|float|bool>|null $custom
     * @return array<string, scalar|null>
     */
    private static function filter(
        string $agent,
        ?string $project,
        ?string $userId,
        ?string $state,
        ?array $custom,
        ?DateTimeInterface $createdAfter,
        ?DateTimeInterface $createdBefore,
        ?int $limit,
        ?int $offset,
    ): array {
        return [
            'agent' => $agent,
            'project' => $project,
            'user_id' => $userId,
            'state' => $state,
            'custom' => $custom === null ? null : Json::encode($custom),
            'created_after' => $createdAfter?->format(DateTimeInterface::RFC3339),
            'created_before' => $createdBefore?->format(DateTimeInterface::RFC3339),
            'limit' => $limit,
            'offset' => $offset,
        ];
    }
}
