<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Generated\Limit;
use GetStream\VisionAgents\Generated\LimitChange;
use GetStream\VisionAgents\Generated\LimitInput;
use GetStream\VisionAgents\Generated\LimitInterval;
use GetStream\VisionAgents\Generated\LimitLevel;
use GetStream\VisionAgents\Generated\LimitPage;
use GetStream\VisionAgents\Generated\LimitResetInput;
use GetStream\VisionAgents\Generated\LimitSubject;
use GetStream\VisionAgents\Generated\SubjectUsage;

/**
 * The customer's spend limits: on the organization, its apps, keys, end users, teams and usage
 * categories, each over a day, a week, two weeks or a month.
 *
 *     $client->spendLimits->create(LimitSubject::Organization, LimitInterval::Monthly, Usd::dollars(20_000));
 *
 * Server side only: the router refuses a user's token, 403 server_side_only. Every limit here
 * is the customer's own; Stream's spend limit for the organization is never among them, and
 * nothing here lifts it. Work a limit refuses fails with a `RouterException` of 402
 * `spend_limit_reached`, or 429 `user_limit_reached` for one end user's limit.
 *
 * A team or category within one app is `Team` or `Category` with an app id: it is sent as
 * `app_team` or `app_category`, which are accepted too.
 */
final readonly class SpendLimits
{
    /** The page size lists are read in, the most the router answers. */
    private const int PAGE = 200;

    public function __construct(private Client $client)
    {
    }

    /**
     * Creates a limit: what one subject may spend in each window of the interval.
     *
     * The level is the most specific one the ids name, which the router works out: `User` with
     * a user id is that user's own limit, `User` with only an app id the default for every user
     * of the app, and `User` alone the default for every user of the organization. Keys go the
     * same way. `$level` overrides it.
     *
     * The router refuses with `limit_above_parent` (the limit is above one of a subject that
     * contains it), `limit_exists` (the subject has one at that level and interval: change it
     * with `setAmount`) or `use_team`.
     */
    public function create(
        LimitSubject|string $subject,
        LimitInterval|string $interval,
        Usd $amount,
        ?string $appId = null,
        ?string $apiKey = null,
        ?string $userId = null,
        ?string $team = null,
        ?string $category = null,
        LimitLevel|string|null $level = null,
        ?string $note = null,
    ): Limit {
        $kind = self::subject($subject, $appId, $team, $category);
        if ($interval === '') {
            throw new ConfigurationException('a spend limit needs an interval');
        }
        self::notNegative($amount);
        $body = new LimitInput(
            interval: $interval,
            limitMicros: $amount->micros,
            subject: $kind,
            apiKey: self::set($apiKey),
            appId: self::set($appId),
            category: self::set($category),
            level: $level === '' ? null : $level,
            note: self::set($note),
            team: self::set($team),
            userId: self::set($userId),
        );
        return Limit::fromArray(Json::asObject($this->client->post('/v1/spend/limits', body: $body->toArray())));
    }

    /**
     * Every limit the filters match, read page by page. No filter is every limit.
     *
     * @return list<Limit>
     */
    public function list(
        LimitSubject|string|null $subject = null,
        LimitLevel|string|null $level = null,
        LimitInterval|string|null $interval = null,
        ?string $appId = null,
        ?string $apiKey = null,
        ?string $userId = null,
        ?string $team = null,
        ?string $category = null,
    ): array {
        if ($subject !== null && $subject !== '') {
            $subject = self::subject($subject, $appId, $team, $category);
        } else {
            self::checkIds($team, $category);
        }
        $query = [
            'subject' => self::value($subject),
            'level' => self::value($level),
            'interval' => self::value($interval),
            'app_id' => self::set($appId),
            'api_key_id' => self::set($apiKey),
            'user_id' => self::set($userId),
            'team' => self::set($team),
            'category' => self::set($category),
            'limit' => self::PAGE,
        ];
        $limits = [];
        do {
            $page = LimitPage::fromArray(Json::asObject($this->client->get('/v1/spend/limits', query: $query)));
            array_push($limits, ...$page->items);
            $query['cursor'] = $page->nextCursor;
        } while ($page->hasMore && $page->nextCursor !== null && $page->nextCursor !== '');
        return $limits;
    }

    public function get(int $id): Limit
    {
        return Limit::fromArray(Json::asObject($this->client->get('/v1/spend/limits/{limit_id}', ['limit_id' => (string) $id])));
    }

    /**
     * Changes what a limit allows. What it is on never changes.
     */
    public function setAmount(int $id, Usd $amount): Limit
    {
        return $this->update($id, amount: $amount);
    }

    /**
     * Changes a limit's note. An empty one clears it.
     */
    public function setNote(int $id, string $note): Limit
    {
        return $this->update($id, note: $note);
    }

    /**
     * Changes a limit's amount and its note at once; null leaves either as it is.
     */
    public function update(int $id, ?Usd $amount = null, ?string $note = null): Limit
    {
        if ($amount !== null) {
            self::notNegative($amount);
        }
        $change = new LimitChange(limitMicros: $amount?->micros, note: $note);
        return Limit::fromArray(Json::asObject($this->client->patch('/v1/spend/limits/{limit_id}', ['limit_id' => (string) $id], $change->toArray())));
    }

    /**
     * Deletes a limit. Its subject's spend from then on is held only to its parents'.
     */
    public function delete(int $id): void
    {
        $this->client->delete('/v1/spend/limits/{limit_id}', ['limit_id' => (string) $id]);
    }

    /**
     * What one subject spent in its current windows, against its limits and grants.
     */
    public function usage(
        LimitSubject|string $subject,
        ?string $appId = null,
        ?string $apiKey = null,
        ?string $userId = null,
        ?string $team = null,
        ?string $category = null,
    ): SubjectUsage {
        $kind = self::subject($subject, $appId, $team, $category);
        $query = [
            'subject' => self::value($kind),
            'app_id' => self::set($appId),
            'api_key_id' => self::set($apiKey),
            'user_id' => self::set($userId),
            'team' => self::set($team),
            'category' => self::set($category),
        ];
        $answered = Json::objects(['rows' => $this->client->get('/v1/spend/limits/usage', query: $query)], 'rows');
        if (count($answered) !== 1) {
            throw new ConfigurationException('reading the spend of one ' . self::value($kind) . ': the router answered ' . count($answered));
        }
        return SubjectUsage::fromArray($answered[0]);
    }

    /**
     * Counts a subject's limits from now until each window ends: every interval's, or only
     * those in `$intervals`. Nothing is refunded: what it spent stays spent. The router refuses
     * with `no_limit` a subject with no limit of the customer's.
     *
     * @param list<LimitInterval|string> $intervals
     */
    public function reset(
        LimitSubject|string $subject,
        string $reason,
        array $intervals = [],
        ?string $appId = null,
        ?string $apiKey = null,
        ?string $userId = null,
        ?string $team = null,
        ?string $category = null,
    ): SubjectUsage {
        if ($reason === '') {
            throw new ConfigurationException('a reset needs a reason');
        }
        $body = new LimitResetInput(
            reason: $reason,
            subject: self::subject($subject, $appId, $team, $category),
            apiKey: self::set($apiKey),
            appId: self::set($appId),
            category: self::set($category),
            intervals: $intervals === [] ? null : $intervals,
            team: self::set($team),
            userId: self::set($userId),
        );
        return SubjectUsage::fromArray(Json::asObject($this->client->post('/v1/spend/limits/usage/reset', body: $body->toArray())));
    }

    /**
     * The subject as the router names it: a team or category within an app is `AppTeam` or
     * `AppCategory`. It refuses what the router would, before anything is sent.
     */
    private static function subject(LimitSubject|string $subject, ?string $appId, ?string $team, ?string $category): LimitSubject|string
    {
        if ($subject === '') {
            throw new ConfigurationException('name the subject: organization, app, api_key, user, team or category');
        }
        $subject = is_string($subject) ? (LimitSubject::tryFrom($subject) ?? $subject) : $subject;
        $team = self::set($team);
        $category = self::set($category);
        $inApp = self::set($appId) !== null;
        self::checkIds($team, $category);
        $named = self::value($subject);
        if ($subject === LimitSubject::Team || $subject === LimitSubject::AppTeam) {
            if ($team === null) {
                throw new ConfigurationException("a {$named} subject names its team in team");
            }
            if ($subject === LimitSubject::Team && $inApp) {
                $subject = LimitSubject::AppTeam;
            }
        }
        if ($subject === LimitSubject::Category || $subject === LimitSubject::AppCategory) {
            if ($category === null) {
                throw new ConfigurationException("a {$named} subject names its category in category");
            }
            if ($subject === LimitSubject::Category && $inApp) {
                $subject = LimitSubject::AppCategory;
            }
        }
        if ($team !== null && $subject !== LimitSubject::Team && $subject !== LimitSubject::AppTeam) {
            throw new ConfigurationException("team is for the team subject, not {$named}");
        }
        if ($category !== null && $subject !== LimitSubject::Category && $subject !== LimitSubject::AppCategory) {
            throw new ConfigurationException("category is for the category subject, not {$named}");
        }
        if (($subject === LimitSubject::AppTeam || $subject === LimitSubject::AppCategory) && !$inApp) {
            throw new ConfigurationException("an {$named} subject names its app in appId");
        }
        return $subject;
    }

    private static function checkIds(?string $team, ?string $category): void
    {
        if (self::set($team) !== null) {
            UsageCategories::checkTeam((string) $team);
        }
        if (self::set($category) !== null) {
            UsageCategories::checkCategory((string) $category);
        }
    }

    private static function notNegative(Usd $amount): void
    {
        if ($amount->micros < 0) {
            throw new ConfigurationException("a spend limit of {$amount} is below zero");
        }
    }

    private static function set(?string $value): ?string
    {
        return $value === '' ? null : $value;
    }

    private static function value(\BackedEnum|string|null $value): ?string
    {
        return $value === null || $value === '' ? null : (string) Json::enumValue($value);
    }
}
