<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

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
     * @param string $said what the router said went wrong, from its error body where it had one
     * @param string $body the response body, cut to 4 KB
     * @param int $retryAfter the seconds a 429 said to wait, or 0
     */
    public function __construct(
        public readonly int $status,
        public readonly string $operation,
        public readonly string $said,
        public readonly string $body = '',
        public readonly int $retryAfter = 0,
        ?Throwable $previous = null,
    ) {
        $prefix = $status === 0 ? "{$operation} never reached the router" : "{$operation} answered {$status}";
        parent::__construct($said === '' ? $prefix : "{$prefix}: {$said}", 0, $previous);
    }
}
