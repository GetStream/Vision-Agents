<?php

declare(strict_types=1);

/*
 * Generates src/Generated from the router's OpenAPI spec: one final readonly class per object
 * schema and one backed enum per enum schema. Operations are not generated; the request layer in
 * Client is hand-written.
 *
 *     php bin/generate.php ../../acceleration/api/openapi.yaml           writes src/Generated
 *     php bin/generate.php ../../acceleration/api/openapi.yaml --check   fails if it is stale
 */

require __DIR__ . '/../vendor/autoload.php';

use Symfony\Component\Yaml\Yaml;

const TARGET = __DIR__ . '/../src/Generated';
const NS = 'GetStream\\VisionAgents\\Generated';

/** @var list<string> $args */
$args = $_SERVER['argv'];
$spec = $args[1] ?? '';
$check = in_array('--check', $args, true);
if ($spec === '' || !is_file($spec)) {
    fwrite(STDERR, "usage: php bin/generate.php path/to/openapi.yaml [--check]\n");
    exit(2);
}

/** @var array{components: array{schemas: array<string, array<string, mixed>>}} $document */
$document = Yaml::parseFile($spec);
$schemas = $document['components']['schemas'];

$files = [];
foreach ($schemas as $name => $schema) {
    if (isset($schema['enum'])) {
        $files[$name . '.php'] = enumOf($name, $schema);
    } elseif (($schema['type'] ?? null) === 'object' && isset($schema['properties'])) {
        foreach (classOf($name, $schema, $schemas) as $class => $source) {
            $files[$class . '.php'] = $source;
        }
    }
}
ksort($files);

if ($check) {
    $stale = [];
    foreach ($files as $file => $source) {
        if (@file_get_contents(TARGET . '/' . $file) !== $source) {
            $stale[] = $file;
        }
    }
    foreach (existing() as $existing) {
        if (!isset($files[basename($existing)])) {
            $stale[] = basename($existing);
        }
    }
    if ($stale !== []) {
        fwrite(STDERR, "src/Generated is stale; run composer generate:\n  " . implode("\n  ", $stale) . "\n");
        exit(1);
    }
    echo "src/Generated is up to date (" . count($files) . " files)\n";
    exit(0);
}

if (!is_dir(TARGET)) {
    mkdir(TARGET, 0o755, true);
}
foreach (existing() as $existing) {
    unlink($existing);
}
foreach ($files as $file => $source) {
    file_put_contents(TARGET . '/' . $file, $source);
}
echo 'wrote ' . count($files) . " files to src/Generated\n";

/**
 * @param array<string, mixed> $schema
 */
function enumOf(string $name, array $schema): string
{
    $cases = '';
    foreach (lst($schema['enum']) as $value) {
        $value = str($value);
        $cases .= '    case ' . caseName($value) . ' = ' . var_export($value, true) . ";\n";
    }
    return fileHeader() . docblock(str($schema['description'] ?? ''), '') . "enum {$name}: string\n{\n{$cases}}\n";
}

/**
 * @param array<string, mixed> $schema
 * @param array<string, array<string, mixed>> $schemas
 * @return array<string, string>
 */
function classOf(string $name, array $schema, array $schemas): array
{
    $out = [];
    $required = lst($schema['required'] ?? []);
    $fields = [];
    foreach (arr($schema['properties']) as $wire => $property) {
        $field = fieldOf($name, $wire, arr($property), $schemas, $out);
        $field['required'] = in_array($wire, $required, true);
        $fields[] = $field;
    }
    // Required parameters come first so a caller can pass them positionally.
    usort($fields, static fn (array $a, array $b): int => $b['required'] <=> $a['required']);

    $params = [];
    $hydrate = [];
    $serialize = [];
    foreach ($fields as $field) {
        $doc = $field['doc'] !== $field['type'] ? "        /** @var {$field['doc']}" . ($field['required'] ? '' : '|null') . " */\n" : '';
        $description = $field['description'] !== '' ? '        // ' . oneLine($field['description']) . "\n" : '';
        $params[] = $description . $doc . '        public ' . ($field['required'] ? $field['type'] : nullable($field['type']))
            . ' $' . $field['php'] . ($field['required'] ? '' : ' = null');
        $read = sprintf($field['read'], var_export($field['wire'], true));
        $hydrate[] = '            ' . $field['php'] . ': ' . ($field['required'] || $field['type'] === 'mixed' ? $read : "array_key_exists(" . var_export($field['wire'], true) . ", \$data) && \$data[" . var_export($field['wire'], true) . "] !== null ? {$read} : null");
        $value = '$this->' . $field['php'];
        $serialize[] = $field['required']
            ? "        \$out[" . var_export($field['wire'], true) . '] = ' . sprintf($field['write'], $value) . ";\n"
            : "        if ({$value} !== null) {\n            \$out[" . var_export($field['wire'], true) . '] = ' . sprintf($field['write'], $value) . ";\n        }\n";
    }

    $source = fileHeader()
        . "use GetStream\\VisionAgents\\Json;\n\n"
        . docblock(str($schema['description'] ?? ''), '')
        . "final readonly class {$name}\n{\n"
        . "    public function __construct(\n" . implode(",\n", $params) . ",\n    ) {\n    }\n\n"
        . "    /**\n     * @param array<mixed> \$data\n     */\n"
        . "    public static function fromArray(array \$data): self\n    {\n        return new self(\n" . implode(",\n", $hydrate) . ",\n        );\n    }\n\n"
        . "    /**\n     * The request body, leaving out every field that was not set.\n     *\n     * @return array<string, mixed>\n     */\n"
        . "    public function toArray(): array\n    {\n        \$out = [];\n" . implode('', $serialize) . "        return \$out;\n    }\n}\n";
    $out[$name] = $source;
    return $out;
}

/**
 * How one property is typed, read off the wire and written back to it.
 *
 * @param array<string, mixed> $property
 * @param array<string, array<string, mixed>> $schemas
 * @param array<string, string> $out nested classes an inline object needs
 * @return array{wire: string, php: string, type: string, doc: string, read: string, write: string, description: string}
 */
function fieldOf(string $owner, string $wire, array $property, array $schemas, array &$out): array
{
    $typed = typeOf($owner, $wire, $property, $schemas, $out);
    return [
        'wire' => $wire,
        'php' => camel($wire),
        'type' => $typed[0],
        'doc' => $typed[1],
        'read' => $typed[2],
        'write' => $typed[3],
        'description' => str($property['description'] ?? ''),
    ];
}

/**
 * @param array<string, mixed> $property
 * @param array<string, array<string, mixed>> $schemas
 * @param array<string, string> $out
 * @return array{0: string, 1: string, 2: string, 3: string} php type, phpdoc type, read with %s as the key, write with %s as the value
 */
function typeOf(string $owner, string $wire, array $property, array $schemas, array &$out): array
{
    if (isset($property['$ref'])) {
        $ref = substr(str($property['$ref']), strlen('#/components/schemas/'));
        $target = $schemas[$ref];
        if (isset($target['enum'])) {
            // The enum when the value is one this SDK knows, the string when the router has
            // learnt a new one since.
            return [$ref . '|string', $ref . '|string', "Json::enum(\$data, %s, {$ref}::class)", 'Json::enumValue(%s)'];
        }
        if (($target['type'] ?? null) === 'object' && isset($target['properties'])) {
            return [$ref, $ref, "{$ref}::fromArray(Json::object(\$data, %s))", '%s->toArray()'];
        }
        return ['mixed', 'mixed', '$data[%s] ?? null', '%s'];
    }

    $type = $property['type'] ?? null;
    if (isset($property['enum'])) {
        return ['string', 'string', 'Json::string($data, %s)', '%s'];
    }
    switch ($type) {
        case 'string':
            if (($property['format'] ?? '') === 'date-time') {
                return ['\\DateTimeImmutable', '\\DateTimeImmutable', 'Json::date($data, %s)', 'Json::dateValue(%s)'];
            }
            return ['string', 'string', 'Json::string($data, %s)', '%s'];
        case 'integer':
            return ['int', 'int', 'Json::int($data, %s)', '%s'];
        case 'number':
            return ['float', 'float', 'Json::float($data, %s)', '%s'];
        case 'boolean':
            return ['bool', 'bool', 'Json::bool($data, %s)', '%s'];
        case 'array':
            $items = arr($property['items'] ?? []);
            if (isset($items['$ref'])) {
                $ref = substr(str($items['$ref']), strlen('#/components/schemas/'));
                $target = $schemas[$ref];
                if (isset($target['enum'])) {
                    return ['array', "list<{$ref}|string>", "Json::enums(\$data, %s, {$ref}::class)", 'Json::enumValues(%s)'];
                }
                if (($target['type'] ?? null) === 'object' && isset($target['properties'])) {
                    return ['array', "list<{$ref}>", "array_map({$ref}::fromArray(...), Json::objects(\$data, %s))", 'array_map(static fn (' . $ref . ' $each): array => $each->toArray(), %s)'];
                }
                return ['array', 'list<mixed>', 'Json::list($data, %s)', '%s'];
            }
            if (isset($items['properties'])) {
                $nested = $owner . pascal($wire) . 'Item';
                foreach (classOf($nested, $items, $schemas) as $class => $source) {
                    $out[$class] = $source;
                }
                return ['array', "list<{$nested}>", "array_map({$nested}::fromArray(...), Json::objects(\$data, %s))", 'array_map(static fn (' . $nested . ' $each): array => $each->toArray(), %s)'];
            }
            return match ($items['type'] ?? null) {
                'string' => ['array', 'list<string>', 'Json::strings($data, %s)', '%s'],
                'integer' => ['array', 'list<int>', 'Json::ints($data, %s)', '%s'],
                'object' => ['array', 'list<array<string, mixed>>', 'Json::objects($data, %s)', '%s'],
                default => ['array', 'list<mixed>', 'Json::list($data, %s)', '%s'],
            };
        case 'object':
            if (isset($property['properties'])) {
                $nested = $owner . pascal($wire);
                foreach (classOf($nested, $property, $schemas) as $class => $source) {
                    $out[$class] = $source;
                }
                return [$nested, $nested, "{$nested}::fromArray(Json::object(\$data, %s))", '%s->toArray()'];
            }
            $values = $property['additionalProperties'] ?? true;
            if (is_array($values) && ($values['type'] ?? null) === 'string') {
                return ['array', 'array<string, string>', 'Json::stringMap($data, %s)', '%s'];
            }
            return ['array', 'array<string, mixed>', 'Json::object($data, %s)', 'Json::objectValue(%s)'];
    }
    return ['mixed', 'mixed', '$data[%s] ?? null', '%s'];
}

function nullable(string $type): string
{
    if ($type === 'mixed') {
        return 'mixed';
    }
    return str_contains($type, '|') ? $type . '|null' : '?' . $type;
}

function fileHeader(): string
{
    return "<?php\n\n// Generated by bin/generate.php from acceleration/api/openapi.yaml. Do not edit.\n\ndeclare(strict_types=1);\n\nnamespace " . NS . ";\n\n";
}

function docblock(string $text, string $indent): string
{
    $text = trim($text);
    if ($text === '') {
        return '';
    }
    $lines = explode("\n", wordwrap(str_replace('*/', '* /', preg_replace('/\s+/', ' ', $text) ?? $text), 92));
    return $indent . "/**\n" . implode('', array_map(static fn (string $line): string => $indent . ' * ' . rtrim($line) . "\n", $lines)) . $indent . " */\n";
}

function oneLine(string $text): string
{
    $flat = trim(preg_replace('/\s+/', ' ', $text) ?? $text);
    return strlen($flat) > 110 ? rtrim(substr($flat, 0, 107)) . '...' : $flat;
}

function camel(string $wire): string
{
    $pascal = pascal($wire);
    return strtolower($pascal[0]) . substr($pascal, 1);
}

function pascal(string $wire): string
{
    return str_replace(' ', '', ucwords(str_replace(['_', '-', '.'], ' ', $wire)));
}

function caseName(string $value): string
{
    if ($value === '') {
        return 'None';
    }
    $name = pascal(preg_replace('/[^A-Za-z0-9]+/', '_', $value) ?? $value);
    return ctype_digit($name[0]) ? 'V' . $name : $name;
}

/**
 * @return list<string>
 */
function existing(): array
{
    $found = glob(TARGET . '/*.php');
    return $found === false ? [] : $found;
}

function str(mixed $value): string
{
    return is_scalar($value) ? (string) $value : '';
}

/**
 * @return array<string, mixed>
 */
function arr(mixed $value): array
{
    if (!is_array($value)) {
        return [];
    }
    /** @var array<string, mixed> $value */
    return $value;
}

/**
 * @return list<mixed>
 */
function lst(mixed $value): array
{
    return is_array($value) ? array_values($value) : [];
}
