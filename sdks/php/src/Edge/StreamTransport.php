<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Edge;

use GetStream\Exceptions\StreamApiException;
use GetStream\Exceptions\StreamTransportException;
use GetStream\Http\HttpClientInterface;
use GetStream\StreamResponse;
use GetStream\VisionAgents\Json;
use JsonException;
use Psr\Http\Client\ClientExceptionInterface;
use Psr\Http\Client\ClientInterface;
use Psr\Http\Message\RequestFactoryInterface;
use Psr\Http\Message\StreamFactoryInterface;

/**
 * Stream's PHP SDK, sent over the same PSR-18 client as everything else here.
 *
 * Its own transport is Guzzle, which blocks; inside a worker that would stall every other call
 * for the length of one request to Stream.
 *
 * @internal
 */
final readonly class StreamTransport implements HttpClientInterface
{
    public function __construct(
        private ClientInterface $http,
        private RequestFactoryInterface $requests,
        private StreamFactoryInterface $streams,
    ) {
    }

    /**
     * @param array<mixed> $headers
     * @param array<mixed> $options
     * @return StreamResponse<mixed>
     */
    public function request(string $method, string $url, array $headers = [], mixed $body = null, array $options = []): StreamResponse
    {
        $request = $this->requests->createRequest($method, $url);
        foreach ($headers as $name => $value) {
            if (is_string($name) && $name !== '' && is_string($value)) {
                $request = $request->withHeader($name, $value);
            }
        }
        if ($body !== null) {
            $request = $request->withBody($this->streams->createStream(is_string($body) ? $body : Json::encode($body)));
        }

        try {
            $response = $this->http->sendRequest($request);
        } catch (ClientExceptionInterface $failed) {
            throw new StreamTransportException($failed->getMessage(), StreamTransportException::ERROR_TYPE_UNKNOWN, $failed);
        }

        $raw = (string) $response->getBody();
        $data = null;
        try {
            $data = $raw === '' ? null : Json::decode($raw);
        } catch (JsonException) {
            $data = $raw;
        }
        $status = $response->getStatusCode();
        if ($status >= 400) {
            $said = is_array($data) && is_string($data['message'] ?? null) ? $data['message'] : "Stream answered {$status}";
            throw new StreamApiException($said, $status, 0, [], false, $raw);
        }
        $lowered = [];
        foreach ($response->getHeaders() as $name => $values) {
            $lowered[strtolower($name)] = implode(', ', $values);
        }
        return new StreamResponse($status, $lowered, $data, $raw);
    }
}
