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
 * Every call takes the user as its `userId`, sent as the owner and as `X-Stream-User-Id` alike,
 * so the two never differ. No user is the app's own connection, and sends no header, whoever
 * the client otherwise acts for. A user's connection is reached only naming that user; the
 * app's is reached by every call, one naming a user included.
 *
 *     $connection = $client->connections->create('linear', userId: 'ada');
 *     $consent = $client->connections->authorize($connection->id, 'https://app.example.com/settings', userId: 'ada');
 */
final readonly class Connections
{
    /** The page size `list` reads in, the most the router answers (`limit` maximum 200). */
    private const int PAGE = 200;

    public function __construct(private Client $client)
    {
    }

    /**
     * Stores a pending connection to a connector, the user's when `$userId` is given and else
     * the app's.
     *
     * @param string $connectorId a built-in, such as linear, or one of the app's own
     * @param ?string $label tells the owner's connections to the same connector apart
     */
    public function create(string $connectorId, ?string $label = null, ?string $userId = null): Connection
    {
        $userId = self::set($userId);
        $owner = $userId === null
            ? new ConnectionOwner(ConnectionOwnerType::App)
            : new ConnectionOwner(ConnectionOwnerType::User, $userId);
        $body = new ConnectionRequest(connectorId: $connectorId, owner: $owner, label: self::set($label));
        return Connection::fromArray(Json::asObject($this->as($userId)->post('/v1/agents/connections', body: $body->toArray())));
    }

    /**
     * Starts the provider's consent for a connection: a first one for a pending connection, a
     * reconnect for one connected before. Open the `launchUrl` it returns in a popup from a page
     * of the return URL's origin and post it the `handoffToken`; the attempt ends at `expiresAt`.
     *
     * @param ?string $returnUrl the app's page the browser goes back to when the consent ends,
     *     with connection_id and status added to its query; none goes back to the dashboard
     */
    public function authorize(string $id, ?string $returnUrl = null, ?string $userId = null): Authorization
    {
        $body = new AuthorizationRequest(returnUrl: self::set($returnUrl));
        return Authorization::fromArray(Json::asObject($this->as(self::set($userId))->post('/v1/agents/connections/{id}/authorizations', ['id' => $id], body: $body->toArray())));
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
     * Every connection of the user's, or of the app's when no user is given, newest first, read
     * page by page. A user's list holds only that user's connections, not the app's.
     *
     * @param ?string $connectorId keeps one connector's
     * @return list<Connection>
     */
    public function list(?string $connectorId = null, ?string $userId = null): array
    {
        $userId = self::set($userId);
        $client = $this->as($userId);
        $query = [
            'owner_type' => ($userId === null ? ConnectionOwnerType::App : ConnectionOwnerType::User)->value,
            'connector_id' => self::set($connectorId),
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
