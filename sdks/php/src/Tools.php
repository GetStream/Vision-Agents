<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use Closure;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\SessionTool;
use JsonException;

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
    /** @var array<string, array{description: string, parameters: array<string, mixed>, run: Closure(array<string, mixed>): mixed}> */
    private array $tools = [];

    /**
     * @param array<string, mixed> $parameters a JSON Schema for the arguments
     * @param callable(array<string, mixed>): mixed $run given the decoded arguments; what it
     *     returns is sent back as the output, JSON encoded unless it is already a string
     */
    public function register(string $name, string $description, array $parameters, callable $run): self
    {
        if ($name === '' || $description === '') {
            throw new ConfigurationException('a tool needs a name and a description, since the description is all the model chooses it by');
        }
        $this->tools[$name] = ['description' => $description, 'parameters' => $parameters, 'run' => $run(...)];
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
            $declared[] = new SessionTool($name, $tool['description'], $tool['parameters'] === [] ? null : $tool['parameters']);
        }
        return $declared;
    }

    /**
     * Runs one tool with the arguments the model sent, which are a JSON string on the wire.
     *
     * @throws ConfigurationException when there is no tool by that name or the arguments are not JSON
     */
    public function call(string $name, string $arguments): string
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
        $output = ($tool['run'])(Json::asObject($decoded));
        return is_string($output) ? $output : Json::encode($output);
    }
}
