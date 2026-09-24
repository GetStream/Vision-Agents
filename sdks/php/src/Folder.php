<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use FilesystemIterator;
use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Folder\Declaration;
use GetStream\VisionAgents\Folder\Document;
use GetStream\VisionAgents\Folder\KnowledgeUrl;
use JsonException;
use RecursiveDirectoryIterator;
use RecursiveIteratorIterator;
use SplFileInfo;
use Symfony\Component\Yaml\Exception\ParseException;
use Symfony\Component\Yaml\Yaml;

/**
 * An agent written down as a directory.
 *
 *     agents/jean/
 *       agent.yaml         what makes it an agent: its name and what it runs on
 *       instructions.md
 *       guardrail.md
 *       skills/think.md    frontmatter naming what the fast model sees, then the prompt
 *       knowledge/pricing.md
 *       knowledge/urls.yaml
 *
 * The fingerprint is taken the way the Go and Python SDKs take it, so an `.agent_sync` any of
 * them wrote is understood by the others.
 */
final readonly class Folder
{
    public const string AGENT_FILE = 'agent.yaml';
    public const string STAMP_FILE = '.agent_sync';
    public const string INSTRUCTIONS_FILE = 'instructions.md';
    public const string GUARDRAIL_FILE = 'guardrail.md';
    public const string SKILLS_DIR = 'skills';
    public const string KNOWLEDGE_DIR = 'knowledge';
    public const string KNOWLEDGE_URLS_FILE = 'urls.yaml';

    /** What a knowledge directory is read from. A model looks things up in prose, not in a binary. */
    private const array READABLE = ['md', 'mdx', 'txt', 'rst', 'yaml', 'yml'];

    /**
     * @param string $declaration agent.yaml as written, trimmed, which the fingerprint is taken over
     * @param list<Skill> $skills in name order
     * @param list<Document> $knowledge in path order
     * @param list<KnowledgeUrl> $knowledgeUrls in the order urls.yaml lists them
     */
    public function __construct(
        public string $path,
        public string $name,
        public string $declaration,
        public Declaration $settings,
        public string $instructions = '',
        public string $guardrail = '',
        public array $skills = [],
        public array $knowledge = [],
        public array $knowledgeUrls = [],
    ) {
    }

    /**
     * Reads an agent directory.
     *
     * agent.yaml is what makes a directory an agent, so it is required. Everything else is
     * optional.
     */
    public static function load(string $path): self
    {
        if (!is_dir($path)) {
            throw new ConfigurationException("{$path} is not an agent directory");
        }
        $path = rtrim($path, '/');
        $declarationFile = $path . '/' . self::AGENT_FILE;
        if (!is_file($declarationFile)) {
            throw new ConfigurationException("{$path} has no " . self::AGENT_FILE . ', so it is not an agent directory');
        }
        $declaration = self::read($declarationFile);
        try {
            $settings = Declaration::fromYaml(Yaml::parse($declaration));
        } catch (ParseException | ConfigurationException $bad) {
            throw new ConfigurationException("{$declarationFile}: {$bad->getMessage()}", 0, $bad);
        }

        return new self(
            path: $path,
            name: $settings->name !== '' ? $settings->name : basename($path),
            declaration: trim($declaration),
            settings: $settings,
            instructions: is_file($path . '/' . self::INSTRUCTIONS_FILE) ? trim(self::read($path . '/' . self::INSTRUCTIONS_FILE)) : '',
            guardrail: is_file($path . '/' . self::GUARDRAIL_FILE) ? trim(self::read($path . '/' . self::GUARDRAIL_FILE)) : '',
            skills: self::loadSkills($path . '/' . self::SKILLS_DIR),
            knowledge: self::loadKnowledge($path . '/' . self::KNOWLEDGE_DIR),
            knowledgeUrls: self::loadKnowledgeUrls($path . '/' . self::KNOWLEDGE_DIR . '/' . self::KNOWLEDGE_URLS_FILE),
        );
    }

    /**
     * A fingerprint of the directory. The same files give the same hash here, in Go and in
     * Python.
     */
    public function hash(): string
    {
        return self::fingerprint($this->declaration, $this->instructions, $this->guardrail, $this->skills, $this->knowledge, $this->knowledgeUrls);
    }

    /**
     * The fingerprint the directory was last synced under, or empty when it never was or the
     * stamp cannot be read.
     */
    public function stamp(): string
    {
        $raw = @file_get_contents($this->path . '/' . self::STAMP_FILE);
        if ($raw === false) {
            return '';
        }
        try {
            return Json::string(Json::asObject(Json::decode($raw)), 'hash');
        } catch (JsonException) {
            return '';
        }
    }

    /**
     * Records what was synced and when, so the next sync can do nothing.
     */
    public function writeStamp(string $hash): void
    {
        $stamp = Json::encode(['hash' => $hash, 'synced_at' => gmdate('Y-m-d\TH:i:s') . '+00:00']) . "\n";
        if (@file_put_contents($this->path . '/' . self::STAMP_FILE, $stamp) === false) {
            throw new ConfigurationException('could not write ' . $this->path . '/' . self::STAMP_FILE);
        }
    }

    /**
     * The Go SDK's `fingerprint`, byte for byte: an MD5 over the declaration, the prompts, then
     * the skills and documents in name order and the pages in the order they were declared.
     *
     * @param list<Skill> $skills
     * @param list<Document> $knowledge
     * @param list<KnowledgeUrl> $pages
     */
    public static function fingerprint(string $declaration, string $instructions, string $guardrail, array $skills = [], array $knowledge = [], array $pages = []): string
    {
        $hasher = hash_init('md5');
        hash_update($hasher, $declaration . "\n" . $instructions . "\n" . $guardrail);

        usort($skills, static fn (Skill $a, Skill $b): int => strcmp($a->name, $b->name));
        foreach ($skills as $skill) {
            // Written the way Python prints a bool and a float, which is what keeps the SDKs'
            // fingerprints of one directory the same.
            hash_update($hasher, "\nskill:{$skill->name}\n{$skill->description}\n{$skill->instructions}" . ($skill->captureVideo ? 'True' : 'False') . "\n");
            if ($skill->deadline > 0) {
                hash_update($hasher, Json::encode($skill->deadline));
            }
        }

        usort($knowledge, static fn (Document $a, Document $b): int => strcmp($a->source, $b->source));
        foreach ($knowledge as $document) {
            hash_update($hasher, "\nknowledge:{$document->source}\n{$document->text}");
        }
        foreach ($pages as $page) {
            hash_update($hasher, "\nurl:{$page->url}\n{$page->title}\n{$page->description}");
        }
        return hash_final($hasher);
    }

    /**
     * @return list<Skill>
     */
    private static function loadSkills(string $path): array
    {
        if (!is_dir($path)) {
            return [];
        }
        $files = glob($path . '/*.md');
        $files = $files === false ? [] : $files;
        sort($files, SORT_STRING);
        $skills = [];
        foreach ($files as $file) {
            if (!is_file($file)) {
                continue;
            }
            try {
                $skills[] = self::parseSkill(basename($file, '.md'), self::read($file));
            } catch (ConfigurationException $bad) {
                throw new ConfigurationException("{$file}: {$bad->getMessage()}", 0, $bad);
            }
        }
        return $skills;
    }

    /**
     * A skill file: frontmatter between --- lines naming it, then the instructions. A deadline
     * is a Go duration, so "30s" and "2m" read the way they look, or a bare number of seconds.
     */
    private static function parseSkill(string $name, string $content): Skill
    {
        $description = '';
        $captureVideo = false;
        $deadline = 0.0;
        $body = $content;

        $trimmed = (string) preg_replace('/^[\x{feff} \t\r\n]+/u', '', $content);
        if (str_starts_with($trimmed, '---')) {
            $rest = ltrim(substr($trimmed, 3), "\r\n");
            $end = strpos($rest, "\n---");
            if ($end !== false) {
                $frontmatter = substr($rest, 0, $end);
                $body = ltrim(substr($rest, $end + 4), "-\r\n");
                foreach (explode("\n", $frontmatter) as $line) {
                    $line = trim($line);
                    if ($line === '' || str_starts_with($line, '#')) {
                        continue;
                    }
                    $colon = strpos($line, ':');
                    if ($colon === false) {
                        throw new ConfigurationException("\"{$line}\" is not a key and a value");
                    }
                    $value = trim(trim(substr($line, $colon + 1)), "\"'");
                    switch (trim(substr($line, 0, $colon))) {
                        case 'name':
                            $name = $value;
                            break;
                        case 'description':
                            $description = $value;
                            break;
                        case 'capture_video':
                            if ($value !== 'true' && $value !== 'false') {
                                throw new ConfigurationException('capture_video must be true or false');
                            }
                            $captureVideo = $value === 'true';
                            break;
                        case 'deadline':
                            $deadline = self::deadline($value);
                            break;
                    }
                }
            }
        }

        if ($description === '') {
            throw new ConfigurationException('a skill needs a description, since it is all the fast model sees');
        }
        $instructions = trim($body);
        if ($instructions === '') {
            throw new ConfigurationException('a skill needs instructions, since they are what the subagent answers under');
        }
        return new Skill($name, $description, $instructions, $captureVideo, $deadline);
    }

    /**
     * A Go duration such as 1m30s or 500ms, or a bare number of seconds, in seconds.
     */
    private static function deadline(string $value): float
    {
        if (is_numeric($value)) {
            return (float) $value;
        }
        $units = ['ns' => 1e-9, 'us' => 1e-6, 'µs' => 1e-6, 'ms' => 1e-3, 's' => 1.0, 'm' => 60.0, 'h' => 3600.0];
        if (preg_match_all('/(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)/u', $value, $parts, PREG_SET_ORDER) === 0
            || implode('', array_column($parts, 0)) !== $value) {
            throw new ConfigurationException("\"{$value}\" is not a deadline");
        }
        $seconds = 0.0;
        foreach ($parts as $part) {
            $seconds += (float) $part[1] * $units[$part[2]];
        }
        return $seconds;
    }

    /**
     * @return list<Document>
     */
    private static function loadKnowledge(string $path): array
    {
        if (!file_exists($path)) {
            return [];
        }
        if (!is_dir($path)) {
            throw new ConfigurationException("{$path} is not a directory");
        }
        $documents = [];
        $files = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($path, FilesystemIterator::SKIP_DOTS));
        foreach ($files as $file) {
            if (!$file instanceof SplFileInfo || !$file->isFile() || !in_array(strtolower($file->getExtension()), self::READABLE, true)) {
                continue;
            }
            // The declaration of which pages to read is not itself something to look things up
            // in. Only the one at the root is; deeper, urls.yaml is a document like any other.
            $source = substr($file->getPathname(), strlen($path) + 1);
            if ($source === self::KNOWLEDGE_URLS_FILE) {
                continue;
            }
            $text = self::read($file->getPathname());
            if (trim($text) === '') {
                continue;
            }
            $documents[] = new Document(str_replace(DIRECTORY_SEPARATOR, '/', $source), $text);
        }
        usort($documents, static fn (Document $a, Document $b): int => strcmp($a->source, $b->source));
        return $documents;
    }

    /**
     * A bad url is refused here rather than when it is subscribed, since a directory that
     * cannot be turned into a knowledge base is worth hearing about before anything is written.
     *
     * @return list<KnowledgeUrl>
     */
    private static function loadKnowledgeUrls(string $path): array
    {
        if (!is_file($path)) {
            return [];
        }
        try {
            $parsed = Yaml::parse(self::read($path));
            if ($parsed === null) {
                return [];
            }
            if (!is_array($parsed) || !array_is_list($parsed)) {
                throw new ConfigurationException('urls.yaml is a list of pages');
            }
            return array_map(KnowledgeUrl::fromYaml(...), $parsed);
        } catch (ParseException | ConfigurationException $bad) {
            throw new ConfigurationException("{$path}: {$bad->getMessage()}", 0, $bad);
        }
    }

    private static function read(string $file): string
    {
        $content = @file_get_contents($file);
        if ($content === false) {
            throw new ConfigurationException("could not read {$file}");
        }
        return $content;
    }
}
