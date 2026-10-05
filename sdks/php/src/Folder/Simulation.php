<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Folder;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\SimulationDeclaration;
use GetStream\VisionAgents\Json;

/**
 * One conversation a simulations/*.yaml file declares, run against the agent the directory is.
 * A file holds a list of them, so related ones can share a file.
 *
 *     simulations/lunch.yaml:
 *         - name: lunch order with a change
 *           scenario: Order a turkey club, then swap it for a veggie wrap.
 *           assertion: The final order is one veggie wrap.
 *           variations: 3
 */
final readonly class Simulation
{
    private const array STRINGS = ['name', 'scenario', 'assertion', 'mode', 'caller_target', 'judge_target', 'caller_stt', 'caller_tts', 'caller_voice'];
    private const array INTS = ['variations', 'max_turns'];

    /**
     * @param ?array<string, string> $tags null when the file names none
     */
    public function __construct(
        public string $name,
        public string $scenario,
        public string $assertion,
        public string $mode = '',
        public int $variations = 0,
        public int $maxTurns = 0,
        public string $callerTarget = '',
        public string $judgeTarget = '',
        public string $callerStt = '',
        public string $callerTts = '',
        public string $callerVoice = '',
        public ?array $tags = null,
    ) {
    }

    /**
     * One entry of a simulations file. A key nobody knows is refused, as in agent.yaml.
     */
    public static function fromYaml(mixed $entry): self
    {
        if (!is_array($entry) || array_is_list($entry)) {
            throw new ConfigurationException('a simulation is a mapping');
        }
        $strings = array_fill_keys(self::STRINGS, '');
        $ints = array_fill_keys(self::INTS, 0);
        $tags = null;
        foreach ($entry as $key => $value) {
            $key = (string) $key;
            if (array_key_exists($key, $strings)) {
                if ($value !== null && !is_scalar($value)) {
                    throw new ConfigurationException("{$key} is a single value");
                }
                $strings[$key] = is_bool($value) ? ($value ? 'true' : 'false') : (string) $value;
            } elseif (array_key_exists($key, $ints)) {
                if ($value !== null && !is_int($value)) {
                    throw new ConfigurationException("{$key} is a whole number");
                }
                $ints[$key] = $value ?? 0;
            } elseif ($key === 'tags') {
                if ($value !== null && (!is_array($value) || (array_is_list($value) && $value !== []))) {
                    throw new ConfigurationException('tags is a mapping of labels');
                }
                $tags = [];
                foreach ($value ?? [] as $name => $label) {
                    $tags[(string) $name] = is_scalar($label) ? (string) $label : '';
                }
            } else {
                throw new ConfigurationException("\"{$key}\" is not something a simulation declares");
            }
        }

        $simulation = new self(
            name: $strings['name'],
            scenario: $strings['scenario'],
            assertion: $strings['assertion'],
            mode: $strings['mode'],
            variations: $ints['variations'],
            maxTurns: $ints['max_turns'],
            callerTarget: $strings['caller_target'],
            judgeTarget: $strings['judge_target'],
            callerStt: $strings['caller_stt'],
            callerTts: $strings['caller_tts'],
            callerVoice: $strings['caller_voice'],
            tags: $tags,
        );
        if ($simulation->name === '') {
            throw new ConfigurationException('a simulation needs a name');
        }
        if ($simulation->scenario === '') {
            throw new ConfigurationException("simulation \"{$simulation->name}\" needs a scenario");
        }
        if ($simulation->assertion === '') {
            throw new ConfigurationException("simulation \"{$simulation->name}\" needs an assertion");
        }
        if (!in_array($simulation->mode, ['', 'text', 'audio'], true)) {
            throw new ConfigurationException("simulation \"{$simulation->name}\" is text or audio, not \"{$simulation->mode}\"");
        }
        return $simulation;
    }

    public function toDeclaration(): SimulationDeclaration
    {
        return new SimulationDeclaration(
            assertion: $this->assertion,
            name: $this->name,
            scenario: $this->scenario,
            callerStt: self::set($this->callerStt),
            callerTarget: self::set($this->callerTarget),
            callerTts: self::set($this->callerTts),
            callerVoice: self::set($this->callerVoice),
            judgeTarget: self::set($this->judgeTarget),
            maxTurns: $this->maxTurns > 0 ? $this->maxTurns : null,
            mode: self::set($this->mode),
            tags: $this->tags === null || $this->tags === [] ? null : $this->tags,
            variations: $this->variations > 0 ? $this->variations : null,
        );
    }

    /**
     * What Go's `json.Marshal` writes for its `agents.Simulation`, which is what the fingerprint
     * is taken over: every field, in Go's order, `<`, `>` and `&` escaped the way Go escapes
     * them.
     */
    public function fingerprint(): string
    {
        $tags = $this->tags;
        if ($tags !== null) {
            ksort($tags, SORT_STRING);
        }
        $encoded = Json::encode([
            'name' => $this->name,
            'scenario' => $this->scenario,
            'assertion' => $this->assertion,
            'mode' => $this->mode,
            'variations' => $this->variations,
            'max_turns' => $this->maxTurns,
            'caller_target' => $this->callerTarget,
            'judge_target' => $this->judgeTarget,
            'caller_stt' => $this->callerStt,
            'caller_tts' => $this->callerTts,
            'caller_voice' => $this->callerVoice,
            'tags' => $tags === null ? null : (object) $tags,
        ]);
        return strtr($encoded, ['<' => '\u003c', '>' => '\u003e', '&' => '\u0026', "\u{2028}" => '\u2028', "\u{2029}" => '\u2029']);
    }

    private static function set(string $value): ?string
    {
        return $value === '' ? null : $value;
    }
}
