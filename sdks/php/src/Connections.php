<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\Authorization;
use GetStream\VisionAgents\Generated\AuthorizationRequest;
use GetStream\VisionAgents\Generated\Connection;
use GetStream\VisionAgents\Generated\ConnectionOwner;
use GetStream\VisionAgents\Generated\ConnectionOwnerType;
use GetStream\VisionAgents\Generated\ConnectionPage;
use GetStream\VisionAgents\Generated\ConnectionRequest;

/**
 * The accounts the app and its users hold at connectors, such as a user's Linear: what an
 * agent's tools reach that account through. Server side only.
 *
 * Every call takes the user as `userId`, sent as the owner and as `X-Stream-User-Id` alike,
 * so the two never differ: a parameter of `get` and `delete`, a field of the options object the
 * other calls take. No user is the app's own connection, and sends no header, whoever
 * the client otherwise acts for. A user's connection is reached only naming that user; the
 * app's is reached by every call, one naming a user included.
 *
 *     $connection = $client->connections->create(new NewConnection('linear', userId: 'ada'));
 *     $consent = $client->connections->authorize($connection->id, new AuthorizeOptions('https://app.example.com/settings', userId: 'ada'));
 */
final readonly class Connections
{
    /** The page size `list` reads in, the most the router answers (`limit` maximum 200). */
    private const int PAGE = 200;

    public function __construct(private Client $client)
    {
    }

    /**
     * Stores a pending connection to a connector, the user's when `userId` is given and else
     * the app's.
     */
    public function create(NewConnection $connection): Connection
    {
        $userId = self::set($connection->userId);
        $owner = $userId === null
            ? new ConnectionOwner(ConnectionOwnerType::App)
            : new ConnectionOwner(ConnectionOwnerType::User, $userId);
        $body = new ConnectionRequest(connectorId: $connection->connectorId, owner: $owner, label: self::set($connection->label));
        return Connection::fromArray(Json::asObject($this->as($userId)->post('/v1/agents/connections', body: $body->toArray())));
    }

    /**
     * Starts the provider's consent for a connection: a first one for a pending connection, a
     * reconnect for one connected before. Open the `launchUrl` it returns in a popup from a page
     * of the return URL's origin and post it the `handoffToken`; the attempt ends at `expiresAt`.
     */
    public function authorize(string $id, AuthorizeOptions $options = new AuthorizeOptions()): Authorization
    {
        $body = new AuthorizationRequest(returnUrl: self::set($options->returnUrl));
        return Authorization::fromArray(Json::asObject($this->as(self::set($options->userId))->post('/v1/agents/connections/{id}/authorizations', ['id' => $id], body: $body->toArray())));
    }

    /**
     * One connection of the user's, or of the app's. Another user's is not found, the same as
     * one that does not exist.
     */
    public function get(string $id, ?string $userId = null): Connection
    {
        return Connection::fromArray(Json::asObject($this->as(self::set($userId))->get('/v1/agents/connections/{id}', ['id' => $id])));
    }

    /**
     * Every connection of the filter's user, or of the app's when it names none, newest first,
     * read page by page. A user's list holds only that user's connections, not the app's.
     *
     * @return list<Connection>
     */
    public function list(ConnectionFilter $filter = new ConnectionFilter()): array
    {
        $userId = self::set($filter->userId);
        $client = $this->as($userId);
        $query = [
            'owner_type' => ($userId === null ? ConnectionOwnerType::App : ConnectionOwnerType::User)->value,
            'connector_id' => self::set($filter->connectorId),
            'limit' => self::PAGE,
        ];
        $connections = [];
        $asked = [];
        do {
            $page = ConnectionPage::fromArray(Json::asObject($client->get('/v1/agents/connections', query: $query)));
            array_push($connections, ...$page->items);
            $cursor = $page->hasMore ? self::set($page->nextCursor) : null;
            // A cursor answered before would page forever.
            if ($cursor === null || isset($asked[$cursor])) {
                return $connections;
            }
            $asked[$cursor] = true;
            $query['cursor'] = $cursor;
        } while (true);
    }

    /**
     * Disconnects a connection of the user's, or of the app's, and drops its credentials. One an
     * agent config binds as its fixed connection is refused with a 409.
     */
    public function delete(string $id, ?string $userId = null): void
    {
        $this->as(self::set($userId))->delete('/v1/agents/connections/{id}', ['id' => $id]);
    }

    private function as(?string $userId): Client
    {
        return $this->client->withBackend($this->client->backend->speakingFor($userId ?? ''));
    }

    private static function set(?string $value): ?string
    {
        return $value === '' ? null : $value;
    }
}
