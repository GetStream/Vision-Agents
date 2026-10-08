<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\Exceptions\StreamApiException;
use GetStream\Exceptions\StreamException;
use GetStream\GeneratedModels\CallRequest;
use GetStream\GeneratedModels\GetOrCreateCallRequest;
use GetStream\VideoClient;
use GetStream\VisionAgents\Edge\StreamTransport;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use Psr\Http\Client\ClientInterface;

/**
 * The Stream call an agent is asked to join, through Stream's own PHP SDK.
 *
 * The acceleration backend joins a call that already exists, so what is needed here is creating
 * one and a link a person can open to be on the other end of it. No media crosses this class.
 *
 * Server side only, because it holds the app secret.
 */
final readonly class Edge
{
    public const string DEFAULT_CALL_TYPE = 'agent';
    public const string STREAM_API = 'https://chat.stream-io-api.com';
    public const string DEFAULT_MONITOR_URL = 'https://getstream.io/video/demos';

    public VideoClient $video;
    private string $apiKey;
    private string $monitorUrl;

    /**
     * @param ?string $apiKey the Stream app calls are created in; `STREAM_API_KEY`
     * @param ?string $apiSecret its secret, which signs the tokens; `STREAM_API_SECRET`
     * @param string $baseUrl where Stream's API is
     * @param ?string $monitorUrl another deployment of the demo page to link to
     */
    public function __construct(
        ?string $apiKey = null,
        ?string $apiSecret = null,
        string $baseUrl = self::STREAM_API,
        ?string $monitorUrl = null,
        ?ClientInterface $http = null,
    ) {
        $this->apiKey = $apiKey ?? Backend::env(Backend::API_KEY_ENV) ?? '';
        $secret = $apiSecret ?? Backend::env(Backend::API_SECRET_ENV) ?? '';
        if ($this->apiKey === '' || $secret === '') {
            throw new ConfigurationException(Backend::API_KEY_ENV . ' and ' . Backend::API_SECRET_ENV . ' are required to create a call');
        }
        $this->monitorUrl = rtrim($monitorUrl ?? Backend::env('EXAMPLE_BASE_URL') ?? self::DEFAULT_MONITOR_URL, '/');
        $this->video = new VideoClient($this->apiKey, $secret, $baseUrl, new StreamTransport($http ?? Http::client(), Http::requests(), Http::streams()));
    }

    /**
     * Creates the call the backend will join, or returns the one already under that id.
     *
     * An empty id names a new call after a random one, which is what a one-off conversation
     * wants.
     */
    public function createCall(Call $call, string $createdBy): Call
    {
        if ($createdBy === '') {
            throw new ConfigurationException('a call needs somebody to have created it');
        }
        $named = new Call($call->id === '' ? bin2hex(random_bytes(8)) : $call->id, $call->type === '' ? self::DEFAULT_CALL_TYPE : $call->type);
        try {
            $this->video->getOrCreateCall($named->type, $named->id, new GetOrCreateCallRequest(data: new CallRequest(createdByID: $createdBy)));
        } catch (StreamApiException $refused) {
            throw new RouterException($refused->getStatusCode(), "POST /api/v2/video/call/{$named->type}/{$named->id}", $refused->getMessage(), '', 0, $refused);
        } catch (StreamException $failed) {
            throw new RouterException(0, "POST /api/v2/video/call/{$named->type}/{$named->id}", $failed->getMessage(), '', 0, $failed);
        }
        return $named;
    }

    /**
     * A token for somebody to join a call as. Signed here, so this makes no request.
     */
    public function token(string $userId, int $validitySeconds = 3600): string
    {
        if ($userId === '') {
            throw new ConfigurationException('a token needs a user to name');
        }
        return $this->video->createUserToken($userId, [], $validitySeconds);
    }

    /**
     * A link a person can open to join a call from a browser and hear the agent.
     */
    public function monitorUrl(Call $call, string $userId, string $userName = ''): string
    {
        if ($call->id === '') {
            throw new ConfigurationException('there is no call to watch');
        }
        return $this->monitorUrl . '/join/' . rawurlencode($call->id) . '?' . http_build_query([
            'api_key' => $this->apiKey,
            'token' => $this->token($userId),
            'skip_lobby' => 'true',
            'user_name' => $userName === '' ? $userId : $userName,
        ]);
    }
}
