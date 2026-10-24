<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use Amp\CancelledException;
use Amp\TimeoutCancellation;
use Closure;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\SessionTool;
use GetStream\VisionAgents\Generated\SessionToolApproval;
use JsonException;
use ReflectionFunction;
use RuntimeException;

use function Amp\async;

/**
 * The caller's own functions, which the model is offered and this process runs.
 *
 *     $agent->tools->register('get_weather', 'The weather somewhere', [
 *         'type' => 'object',
 *         'properties' => ['city' => ['type' => 'string']],
 *         'required' => ['city'],
 *     ], fn (array $args): string => weatherIn($args['city']));
 *
 * A tool is answered over the session's socket, so it only runs while the session is watched.
 */
final class Tools
{
    /** The `{argument}` placeholders in an approval question, as the router fills them in. */
    private const PLACEHOLDER = '/\{([A-Za-z_][A-Za-z0-9_]*)\}/';

    /** @var array<string, array{description: string, parameters: array<string, mixed>, run: Closure, takesCall: bool, displayTitle: ?string, executor: ?string, interruptible: bool, timeout: ?float, approval: ?string}> */
    private array $tools = [];

    /**
     * @param array<string, mixed> $parameters a JSON Schema for the arguments
     * @param callable(array<string, mixed>, ToolCall): mixed $run given the decoded arguments,
     *     and the call it runs for when it takes a second argument; what it returns is sent
     *     back as the output, JSON encoded unless it is already a string
     * @param ?string $displayTitle what the reply's tool call attachment shows, at most 80 characters
     * @param 'server'|'client'|null $executor who runs it: `server`, the router's default, or
     *     `client`, a person's device
     * @param bool $interruptible whether the person talking over the agent cancels a call in flight
     * @param ?float $timeout how many seconds one call may take, in place of the session's tool
     *     timeout; only a tool that waits on Amp, as with `delay()` or Amp's HTTP client, can be
     *     timed out while it runs
     * @param ?string $approval a question a person answers before each call runs, where each
     *     `{name}` is one of the parameters
     */
    public function register(
        string $name,
        string $description,
        array $parameters,
        callable $run,
        ?string $displayTitle = null,
        ?string $executor = null,
        bool $interruptible = false,
        ?float $timeout = null,
        ?string $approval = null,
    ): self {
        if ($name === '' || $description === '') {
            throw new ConfigurationException('a tool needs a name and a description, since the description is all the model chooses it by');
        }
        if ($timeout !== null && $timeout <= 0) {
            throw new ConfigurationException("{$name} has a timeout that is not positive");
        }
        $run = $run(...);
        $reflected = new ReflectionFunction($run);
        $this->tools[$name] = [
            'description' => $description,
            'parameters' => $parameters,
            'run' => $run,
            'takesCall' => $reflected->isVariadic() || $reflected->getNumberOfParameters() >= 2,
            'displayTitle' => $displayTitle,
            'executor' => $executor,
            'interruptible' => $interruptible,
            'timeout' => $timeout,
            'approval' => self::checkApproval($name, $approval, $parameters),
        ];
        return $this;
    }

    /**
     * What the session is opened with.
     *
     * @return list<SessionTool>
     */
    public function declared(): array
    {
        $declared = [];
        foreach ($this->tools as $name => $tool) {
            $declared[] = new SessionTool(
                description: $tool['description'],
                name: $name,
                displayTitle: $tool['displayTitle'],
                executor: $tool['executor'],
                parameters: $tool['parameters'] === [] ? null : $tool['parameters'],
                // Only what differs from the router's defaults.
                approval: $tool['approval'] === null ? null : new SessionToolApproval(title: $tool['approval']),
                interruptible: $tool['interruptible'] ? true : null,
                timeoutMs: $tool['timeout'] === null ? null : (int) round($tool['timeout'] * 1000),
            );
        }
        return $declared;
    }

    /**
     * Whether a tool is registered under this name.
     */
    public function has(string $name): bool
    {
        return isset($this->tools[$name]);
    }

    /**
     * Runs one tool with the arguments the model sent, which are a JSON string on the wire.
     *
     * @throws ConfigurationException when there is no tool by that name or the arguments are not JSON
     * @throws RuntimeException when the call takes longer than the tool's timeout, which stops it
     */
    public function call(string $name, string $arguments, ?ToolCall $call = null): string
    {
        $tool = $this->tools[$name] ?? null;
        if ($tool === null) {
            throw new ConfigurationException("{$name} was asked for, and no tool by that name is registered");
        }
        try {
            $decoded = $arguments === '' ? [] : Json::decode($arguments);
        } catch (JsonException) {
            throw new ConfigurationException("the arguments to {$name} are not JSON");
        }
        $call ??= new ToolCall();
        $run = $tool['run'];
        $running = $tool['takesCall']
            ? static fn (): mixed => $run(Json::asObject($decoded), $call)
            : static fn (): mixed => $run(Json::asObject($decoded));
        if ($tool['timeout'] === null) {
            $output = $running();
        } else {
            $future = async($running);
            // A call that runs on past its timeout may still fail later, with nobody waiting.
            $future->ignore();
            try {
                $output = $future->await(new TimeoutCancellation($tool['timeout']));
            } catch (CancelledException) {
                $call->stop();
                throw new RuntimeException(sprintf('%s took longer than %ss', $name, $tool['timeout']));
            }
        }
        return is_string($output) ? $output : Json::encode($output);
    }

    /**
     * Refuses an approval question the router could not fill in from the arguments.
     *
     * @param array<string, mixed> $parameters
     */
    private static function checkApproval(string $name, ?string $approval, array $parameters): ?string
    {
        if ($approval === null) {
            return null;
        }
        $approval = trim($approval);
        if ($approval === '') {
            throw new ConfigurationException("{$name}'s approval question cannot be blank");
        }
        $properties = Json::object($parameters, 'properties');
        preg_match_all(self::PLACEHOLDER, $approval, $placeholders);
        foreach ($placeholders[1] as $argument) {
            if (!array_key_exists($argument, $properties)) {
                throw new ConfigurationException("{$name}'s approval names {{$argument}}, which is not one of its parameters");
            }
        }
        return $approval;
    }
}
