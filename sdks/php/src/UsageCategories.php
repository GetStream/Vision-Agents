<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;

/**
 * The rule the router and the proxy hold teams and usage categories to. A team is a category
 * of its own, team:<name>, which is why its name is shorter and has no ':'.
 *
 * @internal
 */
final class UsageCategories
{
    /**
     * Where a caller the proxy is not in front of names its usage categories, its team among
     * them as team:<name>, joined with commas. Behind the proxy they are claims on the token,
     * and the proxy deletes the header.
     */
    public const string HEADER = 'X-Stream-Usage-Categories';
    public const string TEAM_PREFIX = 'team:';
    public const int MAX = 10;

    private const string TEAM = '/\A[A-Za-z0-9_.-]{1,59}\z/';
    private const string CATEGORY = '/\A[A-Za-z0-9_.:-]{1,64}\z/';

    public static function checkTeam(string $team): string
    {
        if (preg_match(self::TEAM, $team) !== 1) {
            throw new ConfigurationException('team ' . var_export($team, true) . ' is not 1 to 59 characters of A-Z, a-z, 0-9, _ . and -');
        }
        return $team;
    }

    public static function checkCategory(string $category): string
    {
        if (preg_match(self::CATEGORY, $category) !== 1) {
            throw new ConfigurationException('usage category ' . var_export($category, true) . ' is not 1 to 64 characters of A-Z, a-z, 0-9, _ . : and -');
        }
        // In any case, as the proxy reads it: Team:finance would be a second team.
        if (str_starts_with(strtolower($category), self::TEAM_PREFIX)) {
            throw new ConfigurationException('usage category ' . var_export($category, true) . ' names a team: pass it as the team');
        }
        return $category;
    }

    /**
     * The categories each once and sorted, as the proxy keeps them.
     *
     * @param array<string> $categories
     * @return list<string>
     */
    public static function checked(?string $team, array $categories): array
    {
        $kept = array_values(array_unique(array_map(self::checkCategory(...), $categories)));
        sort($kept, SORT_STRING);
        if ($team !== null) {
            self::checkTeam($team);
        }
        $count = count($kept) + ($team !== null ? 1 : 0);
        if ($count > self::MAX) {
            throw new ConfigurationException('at most ' . self::MAX . " usage categories, the team included; these are {$count}");
        }
        return $kept;
    }

    /**
     * The `team` and `usage_categories` claims, each only when there is one.
     *
     * @param list<string> $categories
     * @return array<string, string|list<string>>
     */
    public static function claims(?string $team, array $categories): array
    {
        return [
            ...($team !== null ? ['team' => $team] : []),
            ...($categories !== [] ? ['usage_categories' => $categories] : []),
        ];
    }

    /**
     * @param list<string> $categories
     * @return array<string, string>
     */
    public static function header(?string $team, array $categories): array
    {
        $named = $team !== null ? [...$categories, self::TEAM_PREFIX . $team] : $categories;
        if ($named === []) {
            return [];
        }
        sort($named, SORT_STRING);
        return [self::HEADER => implode(',', $named)];
    }
}
