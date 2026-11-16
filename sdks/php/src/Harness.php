<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Generated\Harness as HarnessName;

/**
 * What stands between what a caller said and the model that answers them.
 *
 * The loop runs in the backend and is part of the agent's stored config, never of a session:
 * `Agent::sync` writes it, and every session opened under that config runs it.
 */
final readonly class Harness
{
    /**
     * @param HarnessName|string $name which harness the backend runs; empty is `default`, the
     *     only one there is
     * @param string $subagent the model delegated work runs on
     * @param list<Skill> $skills skills of your own, stored on the config in place of the
     *     built-in set
     */
    public function __construct(
        public HarnessName|string $name = '',
        public string $subagent = '',
        public array $skills = [],
    ) {
    }

    /**
     * @param list<Skill> $skills
     */
    public function withSkills(array $skills): self
    {
        return new self($this->name, $this->subagent, $skills);
    }
}
