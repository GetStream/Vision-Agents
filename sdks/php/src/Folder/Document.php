<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Folder;

use GetStream\VisionAgents\Generated\KnowledgeDocument;

/**
 * One file from an agent's knowledge directory, as it will be ingested.
 */
final readonly class Document
{
    /**
     * @param string $source the path relative to the knowledge directory, which a passage is keyed and cited by
     */
    public function __construct(public string $source, public string $text)
    {
    }

    public function toKnowledge(): KnowledgeDocument
    {
        return new KnowledgeDocument($this->source, $this->text);
    }
}
