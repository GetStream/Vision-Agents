<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;

/**
 * What stands between what a caller said and the model that answers them.
 *
 * The loop runs in the backend, so this is configuration rather than behaviour: it is written
 * into the session and the decisions are taken there.
 */
final readonly class Harness
{
    /**
     * @param ?bool $useSkills offer the backend's built-in skills; false turns delegation off
     * @param string $subagent the model delegated work runs on
     * @param list<Skill> $skills skills of your own, replacing the built-in set
     * @param ?int $tasks how much delegated work may run at once
     */
    public function __construct(
        public ?bool $useSkills = null,
        public string $subagent = '',
        public array $skills = [],
        public ?int $tasks = null,
    ) {
        if ($tasks !== null && $tasks < 0) {
            throw new ConfigurationException('tasks cannot be negative');
        }
    }

    /**
     * @param list<Skill> $skills
     */
    public function withSkills(array $skills): self
    {
        return new self($this->useSkills ?? true, $this->subagent, $skills, $this->tasks);
    }
}
