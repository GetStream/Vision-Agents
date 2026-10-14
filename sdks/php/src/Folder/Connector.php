<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Folder;

use GetStream\VisionAgents\Client;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\AgentConnectorBinding;
use GetStream\VisionAgents\Generated\AgentConnectorSelection;
use GetStream\VisionAgents\Generated\AgentConnectorSelectionType;
use GetStream\VisionAgents\Generated\ConnectionPage;
use GetStream\VisionAgents\Json;

/**
 * One connector agent.yaml binds: the alias alone, or a mapping naming it with the connector,
 * the tools and a connection.
 */
final readonly class Connector
{
    private const array KEYS = ['name', 'connector', 'tools', 'connection', 'required'];

    /**
     * @param string $connector the connector id; empty means the name
     * @param ?list<string> $tools names or path.Match patterns; null is every tool
     * @param ?string $connection an app connection's id or label; null is chosen per session
     */
    public function __construct(
        public string $name,
        public string $connector = '',
        public ?array $tools = null,
        public ?string $connection = null,
        public ?bool $required = null,
    ) {
    }

    /**
     * Reads one entry of `connectors`, parsed. A key nobody knows is refused, as everywhere in
     * agent.yaml. Parsed with Yaml::PARSE_OBJECT_FOR_MAP, so a mapping is an object and `{}`
     * is never mistaken for an empty list.
     */
    public static function fromYaml(mixed $entry): self
    {
        if ($entry === null) {
            throw new ConfigurationException('a connector needs a name');
        }
        if (is_scalar($entry)) {
            $entry = (object) ['name' => $entry];
        }
        if (!$entry instanceof \stdClass) {
            throw new ConfigurationException('a connector is a name or a mapping');
        }

        $values = ['name' => null, 'connector' => null, 'tools' => null, 'connection' => null, 'required' => null];
        foreach (get_object_vars($entry) as $key => $value) {
            $key = (string) $key;
            if (!in_array($key, self::KEYS, true)) {
                throw new ConfigurationException("\"connectors.{$key}\" is not something agent.yaml declares");
            }
            $values[$key] = $value;
        }

        $name = self::text('connectors.name', $values['name']);
        if ($name === '') {
            throw new ConfigurationException('a connector needs a name');
        }
        $connector = self::text('connectors.connector', $values['connector']);

        $tools = null;
        if ($values['tools'] !== null) {
            if (!is_array($values['tools']) || !array_is_list($values['tools'])) {
                throw new ConfigurationException("connector \"{$name}\": tools is a list");
            }
            if ($values['tools'] === []) {
                throw new ConfigurationException("connector \"{$name}\": tools: [] grants nothing; leave tools out for every tool");
            }
            $tools = array_map(static fn (mixed $each): string => self::text('connectors.tools', $each), $values['tools']);
        }

        $connection = null;
        if ($values['connection'] !== null) {
            if (!is_scalar($values['connection']) || self::text('connectors.connection', $values['connection']) === '') {
                throw new ConfigurationException("connector \"{$name}\": connection must be a connection id or label");
            }
            $connection = self::text('connectors.connection', $values['connection']);
        }

        if ($values['required'] !== null && !is_bool($values['required'])) {
            throw new ConfigurationException('connectors.required must be true or false');
        }

        return new self($name, $connector, $tools, $connection, $values['required']);
    }

    /**
     * The binding the sync takes. A fixed connection is looked up, so a label that matches no
     * connection or several is refused here, before anything is stored.
     */
    public function toBinding(Client $client): AgentConnectorBinding
    {
        $connectorId = $this->connector === '' ? $this->name : $this->connector;
        $selection = new AgentConnectorSelection(AgentConnectorSelectionType::Session);
        if ($this->connection !== null) {
            $selection = new AgentConnectorSelection(
                AgentConnectorSelectionType::Fixed,
                $this->fixedConnection($client, $connectorId, $this->connection),
            );
        }
        return new AgentConnectorBinding(
            connection: $selection,
            connectorId: $connectorId,
            name: $this->name,
            tools: [],
            required: $this->required,
            toolPatterns: $this->tools ?? ['*'],
        );
    }

    /**
     * The app connection a token names, by id or by the one label that matches. An id wins over
     * a label. Every page is read first: a label on an early page must not hide the id on a
     * later one, and a label is only unique once the list ends.
     */
    private function fixedConnection(Client $client, string $connectorId, string $token): string
    {
        $connections = [];
        $cursor = null;
        $asked = [];
        do {
            $asked[$cursor ?? ''] = true;
            $query = ['owner_type' => 'app', 'connector_id' => $connectorId];
            if ($cursor !== null) {
                $query['cursor'] = $cursor;
            }
            $page = ConnectionPage::fromArray(Json::asObject($client->get('/v1/agents/connections', query: $query)));
            array_push($connections, ...$page->items);
            $cursor = $page->nextCursor === '' ? null : $page->nextCursor;
        } while ($cursor !== null && !isset($asked[$cursor]));

        $labelled = [];
        foreach ($connections as $connection) {
            if ($connection->id === $token) {
                return $connection->id;
            }
            if ($connection->label === $token) {
                $labelled[] = $connection->id;
            }
        }
        if (count($labelled) === 1) {
            return $labelled[0];
        }
        if ($labelled === []) {
            throw new ConfigurationException("agents: connector \"{$this->name}\": no app connection with id or label \"{$token}\"");
        }
        throw new ConfigurationException("agents: connector \"{$this->name}\": \"{$token}\" matches " . count($labelled) . ' app connections');
    }

    private static function text(string $key, mixed $value): string
    {
        if ($value === null) {
            return '';
        }
        if (!is_scalar($value)) {
            throw new ConfigurationException("{$key} is a single value");
        }
        return is_bool($value) ? ($value ? 'true' : 'false') : (string) $value;
    }
}
