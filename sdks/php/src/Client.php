<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\ClaimGuestRequest;
use GetStream\VisionAgents\Generated\ClaimGuestResult;
use GetStream\VisionAgents\Generated\GuestUser;
use GetStream\VisionAgents\Generated\GuestUserRequest;
use JsonException;
use Psr\Http\Client\ClientExceptionInterface;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestFactoryInterface;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\StreamFactoryInterface;

/**
 * The router held once: where it is, who is calling it, and one way to send it a request.
 *
 * One method per HTTP method, not one per endpoint. The shapes are generated into
 * `Generated\`, so a wrapper per operation would say nothing they do not, and a new endpoint
 * would need one written before it could be called:
 *
 *     $configs = $client->get('/v1/agents/configs', query: ['name' => 'docs']);
 *     $client->delete('/v1/agents/sessions/{id}', ['id' => $id]);
 *
 * Synchronous, over any PSR-18 client. Inside a worker, where requests should suspend a fiber
 * rather than stall the loop, install amphp/http-client-psr7 and it is preferred.
 */
final readonly class Client
{
    public Backend $backend;
    private ClientInterface $http;
    private RequestFactoryInterface $requests;
    private StreamFactoryInterface $streams;

    public function __construct(
        ?Backend $backend = null,
        ?ClientInterface $http = null,
        ?RequestFactoryInterface $requests = null,
        ?StreamFactoryInterface $streams = null,
    ) {
        $this->backend = $backend ?? new Backend();
        $this->requests = $requests ?? Http::requests();
        $this->streams = $streams ?? Http::streams();
        $this->http = $http ?? Http::client();
    }

    /**
     * The same router and transport, acting for one end user with the token that proves it.
     */
    public function asUser(string $userId, string $token): self
    {
        return $this->withBackend($this->backend->asUser($userId, $token));
    }

    /**
     * The same router and transport, acting for a guest, which is what their conversations
     * belong to.
     */
    public function asGuest(GuestUser $guest): self
    {
        return $this->asUser($guest->id, $guest->token);
    }

    public function withBackend(Backend $backend): self
    {
        return new self($backend, $this->http, $this->requests, $this->streams);
    }

    /**
     * An agent configured in the backend, addressed by the name a person knows it as.
     *
     * No request is made: a name that matches nothing is refused when a conversation is opened.
     */
    public function agent(string $name): AgentHandle
    {
        return new AgentHandle($this, $name);
    }

    /**
     * Mints a guest so somebody can talk to an agent before they sign up.
     *
     * Nothing is remembered here. A PHP process is a backend, and one that remembered a guest
     * would hand the next visitor the previous one's conversations; which visitor is which is
     * the caller's to know, in their own session or cookie.
     *
     * @param string $id a guest to reuse, for somebody coming back; empty mints a new one
     * @param array<string, mixed>|null $custom anything of the caller's own to keep against them
     */
    public function guestUser(string $id = '', string $name = '', ?array $custom = null): GuestUser
    {
        $body = new GuestUserRequest($id === '' ? null : $id, $name === '' ? null : $name, $custom);
        return GuestUser::fromArray(Json::asObject($this->post('/v1/agents/guests', body: $body->toArray())));
    }

    /**
     * Moves a guest's conversations onto the account they turned out to be.
     *
     * Server side only: the backend that just authenticated the account is the only thing that
     * knows which guest it was, so this refuses before asking rather than reading a 403 after.
     */
    public function claimGuestUser(GuestUser|string $guest, string $userId): ClaimGuestResult
    {
        if (!$this->backend->serverSide()) {
            throw new ConfigurationException('claiming a guest is server side only: it is the backend that just authenticated the account that knows which guest it was');
        }
        $guestId = $guest instanceof GuestUser ? $guest->id : $guest;
        if ($guestId === '' || $userId === '') {
            throw new ConfigurationException('claiming a guest needs the guest and the account');
        }
        $body = new ClaimGuestRequest($guestId, $userId);
        return ClaimGuestResult::fromArray(Json::asObject($this->post('/v1/agents/guests/claim', body: $body->toArray())));
    }

    /**
     * @param array<string, string> $params values for the `{name}` segments of the path
     * @param array<string, scalar|null> $query dropped when null
     */
    public function get(string $path, array $params = [], array $query = []): mixed
    {
        return $this->request('GET', $path, $params, $query);
    }

    /**
     * @param array<string, string> $params
     * @param array<string, scalar|null> $query
     * @param array<mixed>|\stdClass|null $body
     */
    public function post(string $path, array $params = [], array $query = [], array|\stdClass|null $body = null): mixed
    {
        return $this->request('POST', $path, $params, $query, $body);
    }

    /**
     * @param array<string, string> $params
     * @param array<mixed>|\stdClass|null $body
     */
    public function put(string $path, array $params = [], array|\stdClass|null $body = null): mixed
    {
        return $this->request('PUT', $path, $params, [], $body);
    }

    /**
     * @param array<string, string> $params
     * @param array<mixed>|\stdClass|null $body
     */
    public function patch(string $path, array $params = [], array|\stdClass|null $body = null): mixed
    {
        return $this->request('PATCH', $path, $params, [], $body);
    }

    /**
     * @param array<string, string> $params
     */
    public function delete(string $path, array $params = []): mixed
    {
        return $this->request('DELETE', $path, $params);
    }

    /**
     * Sends one request and returns the decoded body, or null for one with none.
     *
     * @param array<string, string> $params
     * @param array<string, scalar|null> $query
     * @param array<mixed>|\stdClass|null $body
     * @throws RouterException when the router answers a failure or cannot be reached
     */
    public function request(string $method, string $path, array $params = [], array $query = [], array|\stdClass|null $body = null): mixed
    {
        $operation = "{$method} {$path}";
        $url = $this->backend->url . self::expand($path, $params) . self::queryOf($query);

        $request = $this->requests->createRequest($method, $url)->withHeader('Accept', 'application/json');
        foreach ($this->backend->headers() as $name => $value) {
            $request = $request->withHeader($name, $value);
        }
        if ($body !== null) {
            $request = $request
                ->withHeader('Content-Type', 'application/json')
                ->withBody($this->streams->createStream(Json::encode($body === [] ? new \stdClass() : $body)));
        }

        try {
            $response = $this->http->sendRequest($request);
        } catch (ClientExceptionInterface $failed) {
            throw new RouterException(0, $operation, $failed->getMessage(), '', 0, $failed);
        }
        return self::answer($operation, $response);
    }

    private static function answer(string $operation, ResponseInterface $response): mixed
    {
        $status = $response->getStatusCode();
        $text = (string) $response->getBody();
        if ($status >= 400) {
            $said = '';
            try {
                $said = Json::string(Json::asObject(Json::decode($text)), 'error');
            } catch (JsonException) {
                $said = trim(substr($text, 0, 200));
            }
            throw new RouterException($status, $operation, $said, substr($text, 0, 4096), (int) $response->getHeaderLine('Retry-After'));
        }
        if ($status === 204 || trim($text) === '') {
            return null;
        }
        try {
            return Json::decode($text);
        } catch (JsonException $unreadable) {
            throw new RouterException($status, $operation, 'the router answered with something that is not JSON', substr($text, 0, 4096), 0, $unreadable);
        }
    }

    /**
     * @param array<string, string> $params
     */
    private static function expand(string $path, array $params): string
    {
        return (string) preg_replace_callback('/\{([a-z0-9_]+)\}/i', static function (array $match) use ($params, $path): string {
            $value = $params[$match[1]] ?? '';
            if ($value === '') {
                throw new ConfigurationException("{$path} needs a {$match[1]}");
            }
            return rawurlencode($value);
        }, $path);
    }

    /**
     * @param array<string, scalar|null> $query
     */
    private static function queryOf(array $query): string
    {
        $pairs = [];
        foreach ($query as $name => $value) {
            if ($value === null) {
                continue;
            }
            $pairs[$name] = is_bool($value) ? ($value ? 'true' : 'false') : (string) $value;
        }
        return $pairs === [] ? '' : '?' . http_build_query($pairs, '', '&', PHP_QUERY_RFC3986);
    }
}
