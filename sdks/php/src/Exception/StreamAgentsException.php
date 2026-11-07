<?php

declare(strict_types=1);

namespace GetStream\StreamAgents\Exception;

use Throwable;

/**
 * Everything this SDK throws on purpose implements this, so one catch covers it.
 */
interface StreamAgentsException extends Throwable
{
}
