<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Exception;

use InvalidArgumentException;

/**
 * What was asked for cannot be done as configured, and was refused before any request.
 */
final class ConfigurationException extends InvalidArgumentException implements VisionAgentsException
{
}
