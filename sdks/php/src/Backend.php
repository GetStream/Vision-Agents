<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;

/**
 * Where the router is and who is calling it.
 *
 * Three ways to say who that is, and which one a deployment takes is a property of the
 * deployment rather than a choice: a customer id for a router with nothing in front of it, a
 * key and secret for a process the customer runs, and a key and a token for somebody acting
 * as one end user.
 *
 * Immutable: `asUser` returns a new one, because a process usually holds both its own
 * credential and one per user, and switching the user on a shared one would make which user a
 * request was for depend on when it happened to run.
 */
final readonly class Backend
{
    public const string DEFAULT_URL = 'http://localhost:8080';
    public const string URL_ENV = 'STREAM_ACCELERATION_URL';
    public const string CUSTOMER_ENV = 'STREAM_ACCELERATION_CUSTOMER_ID';
    public const string API_KEY_ENV = 'STREAM_API_KEY';
    public const string API_SECRET_ENV = 'STREAM_API_SECRET';
    public const string AUTHENTICATE_ENV = 'STREAM_ACCELERATION_AUTHENTICATE';

    /** Short, because a token is minted per request. */
    private const int TOKEN_VALIDITY_SECONDS = 3600;

    public string $url;
    public string $customerId;
    public string $apiKey;
    public string $userId;
    public bool $authenticate;
    private string $apiSecret;
    private string $token;

    /**
     * Every argument falls back to the environment, so a process deployed next to a router
     * needs none of them.
     *
     * @param ?string $url the router's base URL; `STREAM_ACCELERATION_URL`, then localhost
     * @param ?string $customerId who the work is billed to, for a router that trusts the header
     * @param ?string $apiKey the public half of a Stream credential; `STREAM_API_KEY`
     * @param ?string $apiSecret the secret that makes this a backend; `STREAM_API_SECRET`
     * @param string $token a token minted for `$userId` to hold, in place of the secret
     * @param string $userId the end user this is acting for, if it is acting for one
     * @param ?bool $authenticate whether the router sits behind Stream's authenticating proxy,
     *     which wants the credential spelled `api_key` and `stream-auth-type: jwt`
     */
    public function __construct(
        ?string $url = null,
        ?string $customerId = null,
        ?string $apiKey = null,
        ?string $apiSecret = null,
        string $token = '',
        string $userId = '',
        ?bool $authenticate = null,
    ) {
        $this->url = rtrim($url ?? self::env(self::URL_ENV) ?? self::DEFAULT_URL, '/');
        $this->customerId = $customerId ?? self::env(self::CUSTOMER_ENV) ?? '';
        // Naming a customer is choosing how a router with nothing in front of it is reached,
        // and a key that happens to be in the environment does not overrule the choice.
        $this->apiKey = $apiKey ?? ($customerId !== null ? '' : (self::env(self::API_KEY_ENV) ?? ''));
        // A token handed in is the caller's answer to who they are, so an ambient secret does
        // not turn it back into a backend.
        $this->apiSecret = $apiSecret ?? ($token !== '' || $this->apiKey === '' ? '' : (self::env(self::API_SECRET_ENV) ?? ''));
        $this->token = $token;
        $this->userId = $userId;
        $this->authenticate = $authenticate ?? self::flag(self::env(self::AUTHENTICATE_ENV));

        if ($this->authenticate && $this->apiKey === '') {
            throw new ConfigurationException('a router behind the proxy is reached with a credential; pass apiKey, or ' . self::API_KEY_ENV);
        }
    }

    /**
     * Whether this speaks for a process the customer runs rather than for a device.
     *
     * Only a server-side caller reaches the operations the spec does not mark client
     * accessible, which is everything about how an agent is configured.
     */
    public function serverSide(): bool
    {
        return $this->apiSecret !== '' || ($this->apiKey === '' && $this->customerId !== '');
    }

    /**
     * The same router, acting for one end user and holding the token that proves it.
     *
     * A token is the whole credential, so the result is no longer a backend.
     */
    public function asUser(string $userId, string $token): self
    {
        if ($userId === '') {
            throw new ConfigurationException('a user needs an id');
        }
        if ($token === '') {
            throw new ConfigurationException("there is no token for {$userId} to hold");
        }
        return new self($this->url, $this->apiKey === '' ? $this->customerId : '', $this->apiKey, '', $token, $userId, $this->authenticate);
    }

    /**
     * The same backend, saying which of its users it is acting for, so the sessions it opens
     * belong to that user and their own device can reach them afterwards.
     */
    public function actingFor(string $userId): self
    {
        return new self($this->url, $this->apiKey === '' ? $this->customerId : '', $this->apiKey, $this->apiSecret, $this->token, $userId, $this->authenticate);
    }

    /**
     * What every request and every socket handshake carries.
     *
     * Minted per call, so a worker idle longer than a token lasts does not wake up holding an
     * expired one.
     *
     * @return array<string, string>
     */
    public function headers(): array
    {
        if ($this->apiKey === '') {
            if ($this->customerId === '') {
                throw new ConfigurationException(
                    'who is calling is not set; pass customerId or ' . self::CUSTOMER_ENV
                    . ' for a router that trusts one, or apiKey with apiSecret or a token',
                );
            }
            $headers = ['X-Customer-Id' => $this->customerId];
            if ($this->userId !== '') {
                $headers['X-Stream-User-Id'] = $this->userId;
            }
            return $headers;
        }
        if ($this->apiSecret === '' && $this->token === '') {
            throw new ConfigurationException('apiKey needs the secret it belongs to, or a token minted with it');
        }

        if ($this->authenticate) {
            // `jwt` whoever the token is for: the proxy works out the caller from the token it
            // verified, and refuses a request that claims to be `server`.
            return [
                'api_key' => $this->apiKey,
                'stream-auth-type' => 'jwt',
                'Authorization' => 'Bearer ' . $this->proxyToken(),
            ];
        }

        $headers = ['X-Api-Key' => $this->apiKey];
        if ($this->apiSecret !== '') {
            $headers['Authorization'] = 'Bearer ' . self::sign(['server' => true], $this->apiSecret);
            $headers['Stream-Auth-Type'] = 'server';
            if ($this->userId !== '') {
                $headers['X-Stream-User-Id'] = $this->userId;
            }
            return $headers;
        }
        $headers['Authorization'] = 'Bearer ' . $this->token;
        $headers['Stream-Auth-Type'] = 'jwt';
        return $headers;
    }

    /**
     * The WebSocket URL for a path. The credentials are in the headers, not here: a server
     * can send headers on a handshake, which the query string only exists to work around.
     *
     * @param array<string, string> $query
     */
    public function socketUrl(string $path, array $query = []): string
    {
        $url = preg_replace('#^http#', 'ws', $this->url) . $path;
        return $query === [] ? $url : $url . '?' . http_build_query($query);
    }

    /**
     * Signs a Stream token, HS256.
     *
     * @param array<string, mixed> $claims
     */
    public static function sign(array $claims, string $secret, int $validitySeconds = self::TOKEN_VALIDITY_SECONDS): string
    {
        if ($secret === '') {
            throw new ConfigurationException('a token cannot be signed without a secret');
        }
        $issued = time();
        $payload = ['iat' => $issued, 'exp' => $issued + $validitySeconds, ...$claims];
        $signing = self::base64url(Json::encode(['alg' => 'HS256', 'typ' => 'JWT'])) . '.' . self::base64url(Json::encode($payload));
        return $signing . '.' . self::base64url(hash_hmac('sha256', $signing, $secret, true));
    }

    /**
     * An environment variable, from wherever this runtime put it.
     */
    public static function env(string $name): ?string
    {
        $value = $_ENV[$name] ?? $_SERVER[$name] ?? getenv($name);
        return is_string($value) && $value !== '' ? $value : null;
    }

    /**
     * The proxy has no header to read a backend's choice of user from, so with a user named
     * the token has to be that user's.
     */
    private function proxyToken(): string
    {
        if ($this->token !== '') {
            return $this->token;
        }
        return $this->userId !== ''
            ? self::sign(['user_id' => $this->userId], $this->apiSecret)
            : self::sign(['server' => true], $this->apiSecret);
    }

    private static function base64url(string $bytes): string
    {
        return rtrim(strtr(base64_encode($bytes), '+/', '-_'), '=');
    }

    private static function flag(?string $value): bool
    {
        return $value === '1' || strtolower((string) $value) === 'true';
    }
}
