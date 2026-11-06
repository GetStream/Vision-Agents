<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Folder;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Folder;
use GetStream\VisionAgents\Generated\AgentDispatch;
use GetStream\VisionAgents\Generated\DispatchSetting;
use GetStream\VisionAgents\Generated\FloorControl;
use GetStream\VisionAgents\Generated\Greeting;
use GetStream\VisionAgents\Generated\GreetingMode;
use GetStream\VisionAgents\Generated\SandboxOptions;
use GetStream\VisionAgents\Generated\SessionVideo;

/**
 * What agent.yaml declares: what the agent is called and what it runs on.
 *
 * A field left out leaves whatever the stored config already has, so a model chosen in the
 * dashboard survives a sync that says nothing about it.
 */
final readonly class Declaration
{
    private const array STRINGS = ['name', 'description', 'mode', 'stt', 'tts', 'voice', 'llm', 'harness', 'subagent', 'search', 'sandbox'];
    private const array LISTS = ['keyterms'];

    /**
     * @param ?string $sts null when the declaration says nothing, empty when it turns it off
     * @param string $voice the voice the agent speaks in: a provider's own voice id, or one of your voices by id or by
     *     the name you gave it. Prefix it with `custom:` to mean only yours. A native (sts) agent
     *     takes the vendor's own voice name instead
     * @param list<string> $keyterms
     * @param array<string, string> $tags
     * @param ?list<Connector> $connectors null when the declaration says nothing, which leaves
     *     the stored bindings; empty when it lists none, which removes them
     * @param ?SandboxOptions $sandboxOptions how the sandbox is built and what code in it may
     *     reach; outbound network is blocked unless network or an allow list opens it
     * @param ?FloorControl $floorControl the model that controls the agent's floor, and how long
     *     a caller may wait once they stop (max_wait_ms), as written: null when the declaration
     *     says nothing, which leaves the stored choice; empty for the default flow controller again
     */
    public function __construct(
        public string $name = '',
        public string $description = '',
        public string $mode = '',
        public string $stt = '',
        public string $tts = '',
        public ?string $sts = null,
        public string $voice = '',
        public string $llm = '',
        public string $harness = '',
        public string $subagent = '',
        public string $search = '',
        public ?Greeting $greeting = null,
        public string $sandbox = '',
        public array $keyterms = [],
        public array $tags = [],
        public ?SessionVideo $video = null,
        public ?AgentDispatch $dispatch = null,
        public ?array $connectors = null,
        public ?SandboxOptions $sandboxOptions = null,
        public ?FloorControl $floorControl = null,
    ) {
    }

    /**
     * Reads agent.yaml, parsed. A key nobody knows is refused rather than dropped, since a
     * misspelt llm that goes quietly is a config running on a model the file does not name.
     *
     * @param mixed $mapped the same file parsed with Yaml::PARSE_OBJECT_FOR_MAP, which
     *     `connectors` is read from: there `{}` is an object and not an empty list, so it is
     *     refused rather than sent as [] and wiping the stored bindings. `floor_control` is read
     *     from it too, where `{}` is the default flow controller and `[]` is not a mapping
     */
    public static function fromYaml(mixed $parsed, mixed $mapped): self
    {
        if ($parsed === null) {
            return new self();
        }
        if (!is_array($parsed) || array_is_list($parsed)) {
            throw new ConfigurationException('agent.yaml is a mapping of settings');
        }

        $values = [];
        $connectors = null;
        $floorControl = null;
        foreach ($parsed as $key => $value) {
            $key = (string) $key;
            if (in_array($key, self::STRINGS, true)) {
                $values[$key] = self::scalar($key, $value);
            } elseif ($key === 'sts') {
                $values['sts'] = $value === null ? null : self::scalar($key, $value);
            } elseif ($key === 'greeting') {
                $values['greeting'] = self::greeting($value);
            } elseif (in_array($key, self::LISTS, true)) {
                $values[$key] = self::strings($key, $value);
            } elseif ($key === 'tags') {
                $values['tags'] = self::tags($value);
            } elseif ($key === 'video') {
                $values['video'] = self::video($value);
            } elseif ($key === 'dispatch') {
                $values['dispatch'] = self::dispatch($value);
            } elseif ($key === 'sandbox_options') {
                $values['sandbox_options'] = self::sandboxOptions($value);
            } elseif ($key === 'plugins') {
                self::refuseReplaced($mapped instanceof \stdClass ? $mapped->plugins : $value);
            } elseif ($key === 'connectors') {
                $connectors = self::connectors($mapped instanceof \stdClass ? $mapped->connectors : $value);
            } elseif ($key === 'floor_control') {
                $floorControl = self::floorControl($mapped instanceof \stdClass ? $mapped->floor_control : $value);
            } else {
                throw new ConfigurationException("\"{$key}\" is not something agent.yaml declares");
            }
        }

        return new self(
            name: self::pick($values, 'name'),
            description: self::pick($values, 'description'),
            mode: self::pick($values, 'mode'),
            stt: self::pick($values, 'stt'),
            tts: self::pick($values, 'tts'),
            sts: array_key_exists('sts', $values) && is_string($values['sts']) ? $values['sts'] : null,
            voice: self::pick($values, 'voice'),
            llm: self::pick($values, 'llm'),
            harness: self::pick($values, 'harness'),
            subagent: self::pick($values, 'subagent'),
            search: self::pick($values, 'search'),
            greeting: ($values['greeting'] ?? null) instanceof Greeting ? $values['greeting'] : null,
            sandbox: self::pick($values, 'sandbox'),
            keyterms: self::pickList($values, 'keyterms'),
            tags: self::pickTags($values),
            video: ($values['video'] ?? null) instanceof SessionVideo ? $values['video'] : null,
            dispatch: ($values['dispatch'] ?? null) instanceof AgentDispatch ? $values['dispatch'] : null,
            connectors: $connectors,
            sandboxOptions: ($values['sandbox_options'] ?? null) instanceof SandboxOptions ? $values['sandbox_options'] : null,
            floorControl: $floorControl,
        );
    }

    /**
     * Refuses a `plugins` that names anything: connectors replaced it. An empty or null one was
     * never sent, so the file loads as it did.
     */
    private static function refuseReplaced(mixed $value): void
    {
        if ($value !== null && $value !== []) {
            throw new ConfigurationException('plugins is no longer read: plugins were replaced by connectors, so name the provider under connectors');
        }
    }

    private static function scalar(string $key, mixed $value): string
    {
        if ($value === null) {
            return '';
        }
        if (!is_scalar($value)) {
            throw new ConfigurationException("{$key} is a single value");
        }
        return is_bool($value) ? ($value ? 'true' : 'false') : (string) $value;
    }

    /**
     * @return list<string>
     */
    private static function strings(string $key, mixed $value): array
    {
        if ($value === null) {
            return [];
        }
        if (!is_array($value) || !array_is_list($value)) {
            throw new ConfigurationException("{$key} is a list");
        }
        return array_map(static fn (mixed $each): string => self::scalar($key, $each), $value);
    }

    /**
     * @return array<string, string>
     */
    private static function tags(mixed $value): array
    {
        if ($value === null) {
            return [];
        }
        if (!is_array($value) || (array_is_list($value) && $value !== [])) {
            throw new ConfigurationException('tags is a mapping of labels');
        }
        $tags = [];
        foreach ($value as $name => $label) {
            $tags[(string) $name] = self::scalar('tags', $label);
        }
        return $tags;
    }

    private static function video(mixed $value): SessionVideo
    {
        if ($value === null) {
            $value = [];
        }
        if (!is_array($value) || (array_is_list($value) && $value !== [])) {
            throw new ConfigurationException('video is a mapping');
        }
        $source = null;
        $frames = 0;
        foreach ($value as $key => $each) {
            if ($key === 'source') {
                $source = self::scalar('video.source', $each);
            } elseif ($key === 'max_frames') {
                if (!is_int($each) && $each !== null) {
                    throw new ConfigurationException('video.max_frames must be an integer from 1 to 8');
                }
                $frames = $each ?? 0;
            } else {
                throw new ConfigurationException("\"video.{$key}\" is not something agent.yaml declares");
            }
        }
        // Zero reads as one, the way the Go SDK reads it.
        $frames = $frames === 0 ? 1 : $frames;
        if ($frames < 1 || $frames > 8) {
            throw new ConfigurationException('video.max_frames must be an integer from 1 to 8');
        }
        return new SessionVideo(maxFrames: $frames, source: $source === '' ? null : $source);
    }

    /**
     * What the agent opens the call with: its `text`, and a `mode` of exact or variation.
     */
    private static function greeting(mixed $value): ?Greeting
    {
        if ($value === null) {
            return null;
        }
        if (!is_array($value) || (array_is_list($value) && $value !== [])) {
            throw new ConfigurationException('greeting is a mapping of text and mode');
        }
        $text = '';
        $mode = null;
        foreach ($value as $key => $each) {
            if ($key === 'text') {
                $text = self::scalar('greeting.text', $each);
            } elseif ($key === 'mode') {
                $named = self::scalar('greeting.mode', $each);
                $mode = $named === '' ? null : (GreetingMode::tryFrom($named) ?? $named);
            } else {
                throw new ConfigurationException("\"greeting.{$key}\" is not something agent.yaml declares");
            }
        }
        return new Greeting(text: $text, mode: $mode);
    }

    /**
     * Null when the key has no value, which sends nothing; a list, even an empty one, is the
     * whole of the agent's bindings.
     *
     * @return ?list<Connector>
     */
    private static function connectors(mixed $value): ?array
    {
        if ($value === null) {
            return null;
        }
        if (!is_array($value) || !array_is_list($value)) {
            throw new ConfigurationException('connectors must be a list');
        }
        return array_map(Connector::fromYaml(...), $value);
    }

    /**
     * The model that controls the agent's floor and, optionally, `max_wait_ms`, as written: an
     * empty mapping is the default flow controller again, and a key whose value is null is left
     * out. Only the shape is checked here; which models there are, and whether a wait names one,
     * is the router's to say.
     */
    private static function floorControl(mixed $value): ?FloorControl
    {
        if ($value === null) {
            return null;
        }
        if (!$value instanceof \stdClass) {
            throw new ConfigurationException('floor_control is a mapping of model and, optionally, max_wait_ms');
        }
        $model = null;
        $wait = null;
        foreach (get_object_vars($value) as $key => $each) {
            if ($key === 'model') {
                if (!is_string($each) && $each !== null) {
                    throw new ConfigurationException('floor_control.model must be a string');
                }
                $model = $each;
            } elseif ($key === 'max_wait_ms') {
                if (!is_int($each) && $each !== null) {
                    throw new ConfigurationException('floor_control.max_wait_ms must be an integer');
                }
                $wait = $each;
            } else {
                throw new ConfigurationException("\"{$key}\" is not a floor_control setting; model and max_wait_ms are");
            }
        }
        return new FloorControl(maxWaitMs: $wait, model: $model);
    }

    /**
     * Each setting is `enabled` or `disabled`, passed through for the router to judge.
     */
    private static function dispatch(mixed $value): ?AgentDispatch
    {
        if ($value === null) {
            return null;
        }
        if (!is_array($value) || (array_is_list($value) && $value !== [])) {
            throw new ConfigurationException('dispatch is a mapping');
        }
        $settings = ['incoming_call' => null, 'text' => null];
        foreach ($value as $key => $each) {
            if (!array_key_exists($key, $settings)) {
                throw new ConfigurationException("\"dispatch.{$key}\" is not something agent.yaml declares");
            }
            $setting = self::scalar("dispatch.{$key}", $each);
            $settings[$key] = $setting === '' ? null : (DispatchSetting::tryFrom($setting) ?? $setting);
        }
        return new AgentDispatch(incomingCall: $settings['incoming_call'], text: $settings['text']);
    }

    private static function sandboxOptions(mixed $value): ?SandboxOptions
    {
        if ($value === null) {
            return null;
        }
        if (!is_array($value) || (array_is_list($value) && $value !== [])) {
            throw new ConfigurationException('sandbox_options is a mapping');
        }
        $options = [];
        foreach ($value as $key => $each) {
            $name = "sandbox_options.{$key}";
            if (in_array($key, ['setup', 'allowed_domains', 'network_allow_list'], true)) {
                $options[$key] = self::strings($name, $each);
            } elseif (in_array($key, ['cpu', 'memory_gb', 'disk_gb'], true)) {
                if (!is_int($each) || $each < 0) {
                    throw new ConfigurationException("{$name} must be a whole number");
                }
                $options[$key] = $each;
            } elseif ($key === 'network') {
                if (!is_bool($each)) {
                    throw new ConfigurationException("{$name} must be true or false");
                }
                $options[$key] = $each;
            } elseif ($key === 'timeout') {
                try {
                    $seconds = Folder::deadline(self::scalar($name, $each));
                } catch (ConfigurationException) {
                    $seconds = 0.0;
                }
                if ($seconds <= 0 || $seconds > 30 * 60) {
                    throw new ConfigurationException("{$name} must be a duration up to 30m");
                }
                $options['timeout_ms'] = (int) round($seconds * 1000);
            } elseif ($key === 'image') {
                $options[$key] = self::scalar($name, $each);
            } else {
                throw new ConfigurationException("\"{$name}\" is not something agent.yaml declares");
            }
        }
        return SandboxOptions::fromArray($options);
    }

    /**
     * @param array<string, mixed> $values
     */
    private static function pick(array $values, string $key): string
    {
        return is_string($values[$key] ?? null) ? $values[$key] : '';
    }

    /**
     * @param array<string, mixed> $values
     * @return list<string>
     */
    private static function pickList(array $values, string $key): array
    {
        $list = $values[$key] ?? [];
        return is_array($list) ? array_values(array_filter($list, is_string(...))) : [];
    }

    /**
     * @param array<string, mixed> $values
     * @return array<string, string>
     */
    private static function pickTags(array $values): array
    {
        $tags = [];
        foreach (is_array($values['tags'] ?? null) ? $values['tags'] : [] as $name => $label) {
            if (is_string($label)) {
                $tags[(string) $name] = $label;
            }
        }
        return $tags;
    }
}
