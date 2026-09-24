<?php

declare(strict_types=1);

namespace GetStream\VisionAgents\Folder;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\KnowledgeUrlDeclaration;

/**
 * One page from knowledge/urls.yaml. A page is a subscription rather than a copy: what a
 * crawler makes of it is what ends up in the knowledge base.
 */
final readonly class KnowledgeUrl
{
    public function __construct(public string $url, public string $title = '', public string $description = '')
    {
    }

    /**
     * A page written either way: the url on its own, or a mapping naming it alongside what it
     * is. Unknown keys are refused, so a misspelt one is reported rather than dropped.
     */
    public static function fromYaml(mixed $entry): self
    {
        if (is_string($entry)) {
            return self::checked($entry, '', '');
        }
        if (!is_array($entry) || array_is_list($entry)) {
            throw new ConfigurationException('a page is a url, or a mapping naming one');
        }
        $fields = ['url' => '', 'title' => '', 'description' => ''];
        foreach ($entry as $key => $value) {
            if (!array_key_exists((string) $key, $fields)) {
                throw new ConfigurationException("\"{$key}\" is not something a page says; url, title and description are");
            }
            $fields[(string) $key] = is_scalar($value) ? (string) $value : '';
        }
        return self::checked($fields['url'], $fields['title'], $fields['description']);
    }

    public function toDeclaration(): KnowledgeUrlDeclaration
    {
        return new KnowledgeUrlDeclaration($this->url, $this->title === '' ? null : $this->title, $this->description === '' ? null : $this->description);
    }

    private static function checked(string $url, string $title, string $description): self
    {
        if (!str_starts_with($url, 'http://') && !str_starts_with($url, 'https://')) {
            throw new ConfigurationException("\"{$url}\" is not an http or https url");
        }
        return new self($url, $title, $description);
    }
}
