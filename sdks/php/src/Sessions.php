<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\CreateSessionRequest;
use GetStream\VisionAgents\Generated\ModelOverwrites;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SessionFilter;
use GetStream\VisionAgents\Generated\SessionPage;
use GetStream\VisionAgents\Generated\SessionQuery;
use GetStream\VisionAgents\Generated\TextMatch;
use GetStream\VisionAgents\Generated\UpdateSessionRequest;

/**
 * An agent's conversations: the one being held and the ones that were.
 *
 * `query` filters and pages; `search` reads the words a caller titled and described their
 * conversations with, best match first. Both page by cursor: pass a page's `nextCursor` back
 * with the same filters for the next one.
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
     * @param ?string $id a UUID to hold the session by; null lets the router generate one, and
     *     one already taken is refused with 409
     * @param array<string, mixed>|null $custom anything of the caller's own a later query can match on
     */
    public function create(
        ?string $title = null,
        ?string $description = null,
        ?string $projectId = null,
        ?array $custom = null,
        ?bool $incognito = null,
        ?string $conversationId = null,
        ?ModelOverwrites $modelOverwrites = null,
        ?string $callId = null,
        ?string $userId = null,
        ?string $id = null,
    ): Session {
        $request = new CreateSessionRequest(
            id: $id,
            conversationId: $conversationId,
            callId: $callId,
            text: $callId === null ? true : null,
            agent: $this->agent,
            incognito: $incognito,
            title: $title,
            description: $description,
            projectId: $projectId,
            custom: $custom,
            modelOverwrites: $modelOverwrites,
            userId: $userId,
        );
        $created = $this->client->post('/v1/agents/sessions', body: $request->toArray());
        return new Session($this->client, SessionRow::fromArray(Json::asObject($created)));
    }

    /**
     * A page of the agent's conversations, most recently updated first, the ones that ended
     * included. Rows rather than live handles: reading a conversation back is not holding one.
     *
     * @param 'text'|'voice'|'video'|null $modality how the user took part
     * @param 'live'|'ended'|null $state
     * @param ?string $agentId the agent id the sessions were created with
     * @param ?int $limit up to 200; null is the router's 25
     */
    public function query(
        ?string $projectId = null,
        ?string $userId = null,
        ?string $modality = null,
        ?string $state = null,
        ?string $agentId = null,
        ?int $limit = null,
        ?string $cursor = null,
    ): SessionPage {
        return $this->page(null, $projectId, $userId, $modality, $state, $agentId, $limit, $cursor);
    }

    /**
     * Finds a conversation by what it was called: the title, description and opening question.
     * Nothing about an incognito session is searchable, because nothing about it was written
     * down. A search covers every project.
     *
     * @param 'text'|'voice'|'video'|null $modality
     * @param 'live'|'ended'|null $state
     */
    public function search(
        string $text,
        ?string $userId = null,
        ?string $modality = null,
        ?string $state = null,
        ?string $agentId = null,
        ?int $limit = null,
        ?string $cursor = null,
    ): SessionPage {
        return $this->page($text, null, $userId, $modality, $state, $agentId, $limit, $cursor);
    }

    /**
     * One conversation, whether or not it is still being held.
     */
    public function get(string $id): Session
    {
        return new Session($this->client, SessionRow::fromArray(Json::asObject($this->client->get('/v1/agents/sessions/{id}', ['id' => $id]))));
    }

    /**
     * Changes one conversation, whether or not it is still being held, and returns it as it
     * now is. One that ended takes only a title, description and custom labels; instructions,
     * models and voice need it running, and apply from its next turn. A field left null is
     * left as it is; an empty `sts` makes the session a cascade again and an empty `voice`
     * returns to the provider's default.
     *
     * @param array<string, mixed>|null $custom
     */
    public function update(
        string $id,
        ?string $title = null,
        ?string $description = null,
        ?array $custom = null,
        ?string $instructions = null,
        ?string $llm = null,
        ?string $stt = null,
        ?string $tts = null,
        ?string $sts = null,
        ?string $voice = null,
        ?string $thinking = null,
        ?float $temperature = null,
        ?int $maxOutputTokens = null,
        ?string $verbosity = null,
    ): SessionRow {
        $request = new UpdateSessionRequest(
            custom: $custom,
            description: $description,
            instructions: $instructions,
            llm: $llm,
            maxOutputTokens: $maxOutputTokens,
            sts: $sts,
            stt: $stt,
            temperature: $temperature,
            thinking: $thinking,
            title: $title,
            tts: $tts,
            verbosity: $verbosity,
            voice: $voice,
        );
        return SessionRow::fromArray(Json::asObject($this->client->patch('/v1/agents/sessions/{id}', ['id' => $id], $request->toArray())));
    }

    /**
     * Deletes a conversation, running or ended: it is stopped, and its turns and what it
     * remembered are deleted with it. The user's other memories are kept.
     */
    public function delete(string $id): void
    {
        $this->client->delete('/v1/agents/sessions/{id}', ['id' => $id]);
    }

    /**
     * Deletes what one conversation remembered, running or ended, and leaves the rest of the
     * user's memories alone. Server side only.
     */
    public function deleteMemories(string $id): void
    {
        $this->client->delete('/v1/agents/sessions/{id}/memories', ['id' => $id]);
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
     * One function for the listing and the search, so the two cannot drift apart in which
     * filters they honour. Text makes it a search.
     */
    private function page(
        ?string $text,
        ?string $projectId,
        ?string $userId,
        ?string $modality,
        ?string $state,
        ?string $agentId,
        ?int $limit,
        ?string $cursor,
    ): SessionPage {
        $query = new SessionQuery(
            cursor: $cursor,
            filter: new SessionFilter(
                agent: $this->agent,
                agentId: $agentId,
                modality: $modality,
                projectId: $projectId,
                state: $state,
                text: $text === null ? null : new TextMatch($text),
                userId: $userId,
            ),
            limit: $limit,
        );
        return SessionPage::fromArray(Json::asObject($this->client->post('/v1/agents/sessions/query', body: $query->toArray())));
    }
}
