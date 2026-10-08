<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

use GetStream\VisionAgents\Json;
use JsonException;
use RuntimeException;
use Throwable;

/**
 * The router answered with a failure, or never answered at all.
 *
 * A request that never arrived has status 0, because retrying a network failure and retrying a
 * 500 are different decisions. The status is not the exception code: PHP's integer code is for
 * the caller's own use and means nothing here.
 */
final class RouterException extends RuntimeException implements VisionAgentsException
{
    /**
     * @param int $status the HTTP status, or 0 when no response arrived
     * @param string $operation what was being asked, as METHOD /path
     * @param string $said what the router said went wrong: the envelope's message, or the body
     *     or status text when something else answered
     * @param string $body the response body, cut to 4 KB
     * @param int $retryAfter the seconds a 429 said to wait, or 0
     * @param string|null $type the kind of failure, an `ErrorType` value or one a newer router
     *     added; null when the body was not the router's envelope
     * @param string|null $errorCode what to branch on (`not_configured`, `validation_failed`,
     *     ...); the router adds codes, so an unknown one is expected. Not `code`, which is
     *     Exception's own integer
     * @param string|null $docUrl where the code is explained
     * @param string|null $requestId the response's X-Request-Id, what to quote to support: a 500
     *     says only "something went wrong"
     */
    public function __construct(
        public readonly int $status,
        public readonly string $operation,
        public readonly string $said,
        public readonly string $body = '',
        public readonly int $retryAfter = 0,
        ?Throwable $previous = null,
        public readonly ?string $type = null,
        public readonly ?string $errorCode = null,
        public readonly ?string $docUrl = null,
        public readonly ?string $requestId = null,
    ) {
        $prefix = $status === 0 ? "{$operation} never reached the router" : "{$operation} answered {$status}";
        parent::__construct($said === '' ? $prefix : "{$prefix}: {$said}", 0, $previous);
    }

    /**
     * The failure a response that was not a success reports.
     *
     * The router answers `{"error": {"message", "type", "code", "doc_url"}}`, but a proxy's
     * HTML, an empty body or an older router's `{"error": "..."}` is not that, so those keep
     * their text and leave the rest null rather than fail to parse in place of the router's
     * failure.
     *
     * @internal
     * @param string $reason the status text, said when the body says nothing
     */
    public static function answered(int $status, string $operation, string $text, string $reason = '', string $requestId = '', int $retryAfter = 0, ?Throwable $previous = null): self
    {
        $error = null;
        try {
            $decoded = Json::decode($text);
            $error = is_array($decoded) ? ($decoded['error'] ?? null) : null;
        } catch (JsonException) {
        }
        $detail = Json::asObject($error);
        $said = Json::string($detail, 'message');
        if ($said === '') {
            $detail = [];
            $said = is_string($error) ? $error : trim(substr($text, 0, 200));
        }
        return new self(
            $status,
            $operation,
            $said === '' ? $reason : $said,
            substr($text, 0, 4096),
            $retryAfter,
            $previous,
            type: self::present(Json::string($detail, 'type')),
            errorCode: self::present(Json::string($detail, 'code')),
            docUrl: self::present(Json::string($detail, 'doc_url')),
            requestId: self::present($requestId),
        );
    }

    private static function present(string $value): ?string
    {
        return $value === '' ? null : $value;
    }
}
