<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\SessionSkill;
use GetStream\VisionAgents\Generated\SkillRequest;

/**
 * A kind of work worth handing to the slower model.
 *
 * There is nothing behind a skill but a better model and more time. What it declares is the
 * description the fast model chooses by, and the instructions the slow one answers under.
 */
final readonly class Skill
{
    /**
     * @param string $description the one line the fast model sees
     * @param string $instructions the full prompt, which only the subagent sees
     * @param float $deadline seconds the work may run before it is abandoned; 0 leaves the backend's
     */
    public function __construct(
        public string $name,
        public string $description,
        public string $instructions,
        public bool $captureVideo = false,
        public float $deadline = 0.0,
    ) {
        if ($name === '') {
            throw new ConfigurationException('a skill needs a name');
        }
        if ($description === '') {
            throw new ConfigurationException("{$name} needs a description, since it is all the fast model sees");
        }
        if ($instructions === '') {
            throw new ConfigurationException("{$name} needs instructions, since they are what the subagent answers under");
        }
        if ($deadline < 0) {
            throw new ConfigurationException("{$name} cannot be given less than no time");
        }
    }

    public function toSession(): SessionSkill
    {
        return new SessionSkill($this->name, $this->description, $this->instructions, null, $this->captureVideo, $this->deadlineMs());
    }

    /**
     * The config the skill belongs to is written by the same sync request, so the router fills
     * its id in.
     */
    public function toSync(): SkillRequest
    {
        return new SkillRequest('', $this->name, $this->description, $this->instructions, $this->captureVideo, $this->deadlineMs());
    }

    private function deadlineMs(): ?int
    {
        return $this->deadline > 0 ? (int) round($this->deadline * 1000) : null;
    }
}
