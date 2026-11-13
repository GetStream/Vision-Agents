<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\CreateSessionRequest;
use GetStream\VisionAgents\Generated\ModelOverwrites;
use GetStream\VisionAgents\Generated\Session as SessionRow;
use GetStream\VisionAgents\Generated\SessionPage;
use GetStream\VisionAgents\Generated\SessionSortField;
use GetStream\VisionAgents\Generated\UpdateSessionRequest;

/**
 * An agent's conversations: the one being held and the ones that were.
 *
 * `query` filters, sorts and pages; `search` reads the words a caller titled and described
 * their conversations with, best match first. Both take the filter as the router reads it, an
 * array with the wire's own keys, and both are narrowed to this agent whatever the filter
 * says. Both page by cursor: pass a page's `nextCursor` back with the same filter and sort for
 * the next one.
 *
 * A filter is field keys and/or `$and` / `$or`; the keys of one array are ANDed. `$and` and
 * `$or` take 1 to 10 filters and nest up to 3 deep. The fields, and there are no others:
 *
 * - `project_id`: `'health'`, `['$eq' => 'health']` or `['$in' => ['health', 'legal']]` (1 to 50)
 * - `text`: `['$q' => 'words']`, once, at the top level or directly in the top-level `$and`,
 *   never inside an `$or`
 * - `review_score`: `4`, or any of `$eq`, `$gt`, `$gte`, `$lt`, `$lte`, ANDed:
 *   `['$gte' => 2, '$lt' => 4]`. Server side only
 * - `user_id`: `'u1'` or `['$eq' => 'u1']`, at the top level or directly in the top-level
 *   `$and`. Server side only
 * - `agent`: set by this SDK to the agent the sessions were asked of
 *
 * Anything else is refused by the router with a 400 that names it.
 *
 * @phpstan-type SortRule array{field: SessionSortField|string, direction?: -1|1}
 */
final readonly class Sessions
{
    public function __construct(private Client $client, private string $agent)
    {
    }

    /**
     * Opens a conversation, held in writing unless `startVoice` has the agent join the
     * session's own call, `agent:<session id>`.
     *
     * A field left null is left out, so the config decides it.
     *
     * @param ?string $id up to 64 letters, digits, - and _ to hold the session by; null lets the
     *     router generate one, and one already taken is refused with 409
     * @param array<string, mixed>|null $custom anything of the caller's own a later query can match on
     * @param ?string $team the team the conversation's work is counted under, its team spend
     *     limits included, for a backend opening one for a team. Null is the team the caller's
     *     token names, else the agent's `tags: team`. Server side only: a device sending one is
     *     refused with 403, and a token signed for another team with 400
     */
    public function create(
        ?string $title = null,
        ?string $description = null,
        ?string $projectId = null,
        ?array $custom = null,
        ?bool $incognito = null,
        ?ModelOverwrites $modelOverwrites = null,
        ?bool $startVoice = null,
        ?string $userId = null,
        ?string $id = null,
        ?string $team = null,
    ): Session {
        $request = new CreateSessionRequest(
            team: $team === null ? null : UsageCategories::checkTeam($team),
            id: $id,
            startVoice: $startVoice,
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
     * A page of the agent's conversations, the ones that ended included. Rows rather than live
     * handles: reading a conversation back is not holding one.
     *
     *     $sessions->query(filter: ['$or' => [['project_id' => 'health'], ['project_id' => 'legal']]]);
     *     $sessions->query(sort: [['field' => 'pinned'], ['field' => 'last_message_at', 'direction' => -1]]);
     *
     * @param array<string, mixed> $filter in the wire's shape, see the class comment. Its top
     *     level `agent` is always this agent
     * @param list<SortRule>|null $sort null is the agent's last answer, newest first.
     *     `direction` is -1, descending and the default, or 1. One of: `last_message_at`
     *     either way; `pinned`, descending, alone or followed by `last_message_at`;
     *     `review_score` either way, server side only; `relevance`, descending, only with a
     *     `text` filter
     * @param ?int $limit 1 to 200; null is the router's 25
     * @param ?string $cursor the previous page's `nextCursor`, good only with the filter and
     *     sort it came from
     */
    public function query(array $filter = [], ?array $sort = null, ?int $limit = null, ?string $cursor = null): SessionPage
    {
        $body = [];
        if ($cursor !== null) {
            $body['cursor'] = $cursor;
        }
        // The agent first, so it is this one whatever the caller's filter names. The caller's
        // keys stay at the top level beside it: text and user_id are only read there.
        $body['filter'] = self::filterObject(['agent' => $this->agent] + $filter);
        if ($limit !== null) {
            $body['limit'] = $limit;
        }
        if ($sort !== null) {
            $body['sort'] = array_map(self::sortRule(...), $sort);
        }
        return SessionPage::fromArray(Json::asObject($this->client->post('/v1/agents/sessions/query', body: $body)));
    }

    /**
     * Finds a conversation by what it was called: the title, description, project and agent
     * name, best match first, which is why it takes no sort. Nothing about an incognito
     * session is searchable, because nothing about it was written down.
     *
     *     $sessions->search('pricing', filter: ['project_id' => 'health'], limit: 5);
     *
     * @param array<string, mixed> $filter as `query` takes it, without a top-level `text`:
     *     the words are this call's
     *
     * @throws ConfigurationException when the filter has a top-level `text` of its own
     */
    public function search(string $text, array $filter = [], ?int $limit = null, ?string $cursor = null): SessionPage
    {
        if (array_key_exists('text', $filter)) {
            throw new ConfigurationException('search() sets the filter\'s text itself; leave text out of the filter, or call query()');
        }
        return $this->query($filter + ['text' => ['$q' => $text]], null, $limit, $cursor);
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
     * now is. One that ended takes only a title, description, custom labels and a pin; models
     * and voice need it running, and apply from its next turn. A field left null is
     * left as it is; an empty `sts` makes the session a cascade again and an empty `voice`
     * returns to the provider's default.
     *
     * @param array<string, mixed>|null $custom
     * @param ?bool $pinned true pins the session and false unpins it; the row's `pinnedAt` says
     *     when, and is null for one that is not pinned. A pin does not move a session in
     *     `last_message_at` order: sort by `pinned` to list the pinned first
     */
    public function update(
        string $id,
        ?string $title = null,
        ?string $description = null,
        ?array $custom = null,
        ?string $llm = null,
        ?string $stt = null,
        ?string $tts = null,
        ?string $sts = null,
        ?string $voice = null,
        ?string $thinking = null,
        ?float $temperature = null,
        ?int $maxOutputTokens = null,
        ?string $verbosity = null,
        ?bool $pinned = null,
    ): SessionRow {
        $request = new UpdateSessionRequest(
            custom: $custom,
            description: $description,
            llm: $llm,
            maxOutputTokens: $maxOutputTokens,
            pinned: $pinned,
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
     * Pins a conversation, running or ended, and returns it as it now is. Pinning one that is
     * already pinned keeps the time it was first pinned.
     */
    public function pin(string $id): SessionRow
    {
        return $this->update($id, pinned: true);
    }

    /**
     * Unpins a conversation, and returns it as it now is.
     */
    public function unpin(string $id): SessionRow
    {
        return $this->update($id, pinned: false);
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
     * A filter as JSON has to carry it: every filter and every condition an object, so one
     * with nothing in it is sent as `{}` where PHP's own encoding of an empty array is `[]`.
     * `$and` and `$or` stay lists of filters, and an operator's operand (the list of `$in`)
     * is sent as it was given.
     *
     * @param array<array-key, mixed> $filter
     */
    private static function filterObject(array $filter): \stdClass
    {
        $out = new \stdClass();
        foreach ($filter as $key => $value) {
            $key = (string) $key;
            if (($key === '$and' || $key === '$or') && is_array($value)) {
                $out->{$key} = array_map(
                    static fn (mixed $each): mixed => is_array($each) ? self::filterObject($each) : $each,
                    array_values($value),
                );
            } elseif (is_array($value)) {
                $out->{$key} = (object) $value;
            } else {
                $out->{$key} = $value;
            }
        }
        return $out;
    }

    /**
     * @param array<array-key, mixed> $rule
     *
     * @return array<string, mixed>
     */
    private static function sortRule(array $rule): array
    {
        $field = $rule['field'] ?? null;
        $out = ['field' => $field instanceof SessionSortField ? $field->value : $field];
        if (array_key_exists('direction', $rule)) {
            $out['direction'] = $rule['direction'];
        }
        return $out;
    }
}
