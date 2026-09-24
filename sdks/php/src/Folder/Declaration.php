<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Folder;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\SessionVideo;

/**
 * What agent.yaml declares: what the agent is called and what it runs on.
 *
 * A field left out leaves whatever the stored config already has, so a model chosen in the
 * dashboard survives a sync that says nothing about it.
 */
final readonly class Declaration
{
    private const array STRINGS = ['name', 'description', 'mode', 'stt', 'tts', 'voice', 'llm', 'subagent', 'search', 'greeting', 'sandbox'];
    private const array LISTS = ['plugins', 'keyterms'];

    /**
     * @param ?string $sts null when the declaration says nothing, empty when it turns it off
     * @param list<string> $plugins
     * @param list<string> $keyterms
     * @param array<string, string> $tags
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
        public string $subagent = '',
        public string $search = '',
        public string $greeting = '',
        public string $sandbox = '',
        public array $plugins = [],
        public array $keyterms = [],
        public array $tags = [],
        public ?SessionVideo $video = null,
    ) {
    }

    /**
     * Reads agent.yaml, parsed. A key nobody knows is refused rather than dropped, since a
     * misspelt llm that goes quietly is a config running on a model the file does not name.
     */
    public static function fromYaml(mixed $parsed): self
    {
        if ($parsed === null) {
            return new self();
        }
        if (!is_array($parsed) || array_is_list($parsed)) {
            throw new ConfigurationException('agent.yaml is a mapping of settings');
        }

        $values = [];
        foreach ($parsed as $key => $value) {
            $key = (string) $key;
            if (in_array($key, self::STRINGS, true)) {
                $values[$key] = self::scalar($key, $value);
            } elseif ($key === 'sts') {
                $values['sts'] = $value === null ? null : self::scalar($key, $value);
            } elseif (in_array($key, self::LISTS, true)) {
                $values[$key] = self::strings($key, $value);
            } elseif ($key === 'tags') {
                $values['tags'] = self::tags($value);
            } elseif ($key === 'video') {
                $values['video'] = self::video($value);
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
            subagent: self::pick($values, 'subagent'),
            search: self::pick($values, 'search'),
            greeting: self::pick($values, 'greeting'),
            sandbox: self::pick($values, 'sandbox'),
            plugins: self::pickList($values, 'plugins'),
            keyterms: self::pickList($values, 'keyterms'),
            tags: self::pickTags($values),
            video: ($values['video'] ?? null) instanceof SessionVideo ? $values['video'] : null,
        );
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
        return new SessionVideo($source === '' ? null : $source, $frames);
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
