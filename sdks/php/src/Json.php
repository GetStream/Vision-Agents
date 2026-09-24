<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use DateTimeImmutable;
use DateTimeInterface;
use Exception;
use JsonException;
use stdClass;

/**
 * Reading JSON the router wrote, and writing what it reads.
 *
 * Tolerant on the way in: a field of the wrong type reads as its zero value rather than
 * failing the whole response, the way Go's decoder leaves a missing field zero. The generated
 * models go through here, so the tolerance is in one place.
 *
 * @internal
 */
final class Json
{
    /**
     * @throws JsonException
     */
    public static function encode(mixed $value): string
    {
        return json_encode($value, JSON_THROW_ON_ERROR | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE | JSON_PRESERVE_ZERO_FRACTION);
    }

    /**
     * @throws JsonException
     */
    public static function decode(string $text): mixed
    {
        return json_decode($text, true, 512, JSON_THROW_ON_ERROR);
    }

    /**
     * @param array<mixed> $data
     */
    public static function string(array $data, string $key): string
    {
        $value = $data[$key] ?? null;
        return is_string($value) ? $value : (is_int($value) || is_float($value) ? (string) $value : '');
    }

    /**
     * @param array<mixed> $data
     */
    public static function int(array $data, string $key): int
    {
        $value = $data[$key] ?? null;
        return is_int($value) ? $value : (is_float($value) ? (int) $value : 0);
    }

    /**
     * @param array<mixed> $data
     */
    public static function float(array $data, string $key): float
    {
        $value = $data[$key] ?? null;
        return is_int($value) || is_float($value) ? (float) $value : 0.0;
    }

    /**
     * @param array<mixed> $data
     */
    public static function bool(array $data, string $key): bool
    {
        return ($data[$key] ?? null) === true;
    }

    /**
     * An RFC 3339 timestamp. Go writes as many fractional digits as a value needs, up to nine,
     * and PHP reads six, so the rest are cut rather than failing the parse.
     *
     * @param array<mixed> $data
     */
    public static function date(array $data, string $key): DateTimeImmutable
    {
        return self::parseDate(self::string($data, $key)) ?? new DateTimeImmutable('@0');
    }

    public static function parseDate(string $text): ?DateTimeImmutable
    {
        if ($text === '') {
            return null;
        }
        $trimmed = preg_replace('/(\.\d{6})\d+/', '$1', $text) ?? $text;
        try {
            return new DateTimeImmutable($trimmed);
        } catch (Exception) {
            return null;
        }
    }

    /**
     * @param array<mixed> $data
     * @return array<string, mixed>
     */
    public static function object(array $data, string $key): array
    {
        return self::asObject($data[$key] ?? null);
    }

    /**
     * @return array<string, mixed>
     */
    public static function asObject(mixed $value): array
    {
        if (!is_array($value)) {
            return [];
        }
        $out = [];
        foreach ($value as $name => $each) {
            $out[(string) $name] = $each;
        }
        return $out;
    }

    /**
     * @param array<mixed> $data
     * @return list<array<string, mixed>>
     */
    public static function objects(array $data, string $key): array
    {
        $out = [];
        foreach (self::list($data, $key) as $each) {
            if (is_array($each)) {
                $out[] = self::asObject($each);
            }
        }
        return $out;
    }

    /**
     * @param array<mixed> $data
     * @return list<mixed>
     */
    public static function list(array $data, string $key): array
    {
        $value = $data[$key] ?? null;
        return is_array($value) ? array_values($value) : [];
    }

    /**
     * @param array<mixed> $data
     * @return list<string>
     */
    public static function strings(array $data, string $key): array
    {
        return array_values(array_filter(self::list($data, $key), is_string(...)));
    }

    /**
     * @param array<mixed> $data
     * @return list<int>
     */
    public static function ints(array $data, string $key): array
    {
        return array_values(array_filter(self::list($data, $key), is_int(...)));
    }

    /**
     * @param array<mixed> $data
     * @return array<string, string>
     */
    public static function stringMap(array $data, string $key): array
    {
        $out = [];
        foreach (self::object($data, $key) as $name => $value) {
            if (is_string($value)) {
                $out[$name] = $value;
            }
        }
        return $out;
    }

    /**
     * The enum case for a value this SDK knows, or the value itself for one the router learnt
     * after it shipped.
     *
     * @template T of \BackedEnum
     * @param array<mixed> $data
     * @param class-string<T> $enum
     * @return T|string
     */
    public static function enum(array $data, string $key, string $enum): \BackedEnum|string
    {
        $value = self::string($data, $key);
        return $enum::tryFrom($value) ?? $value;
    }

    /**
     * @template T of \BackedEnum
     * @param array<mixed> $data
     * @param class-string<T> $enum
     * @return list<T|string>
     */
    public static function enums(array $data, string $key, string $enum): array
    {
        return array_map(static fn (string $value): \BackedEnum|string => $enum::tryFrom($value) ?? $value, self::strings($data, $key));
    }

    public static function enumValue(\BackedEnum|string $value): string|int
    {
        return $value instanceof \BackedEnum ? $value->value : $value;
    }

    /**
     * @param list<\BackedEnum|string> $values
     * @return list<string|int>
     */
    public static function enumValues(array $values): array
    {
        return array_map(self::enumValue(...), $values);
    }

    public static function dateValue(DateTimeInterface $value): string
    {
        return $value->format(DateTimeInterface::RFC3339_EXTENDED);
    }

    /**
     * An empty map written as {} rather than [], which is what a Go map field wants.
     *
     * @param array<string, mixed> $value
     * @return array<string, mixed>|stdClass
     */
    public static function objectValue(array $value): array|stdClass
    {
        return $value === [] ? new stdClass() : $value;
    }
}
