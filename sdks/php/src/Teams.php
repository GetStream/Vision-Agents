<?php

declare(strict_types=1);

namespace GetStream\VisionAgents;

use GetStream\VisionAgents\Exception\ConfigurationException;
use GetStream\VisionAgents\Exception\RouterException;
use GetStream\VisionAgents\Generated\Limit;
use GetStream\VisionAgents\Generated\LimitInterval;
use GetStream\VisionAgents\Generated\LimitSubject;
use GetStream\VisionAgents\Generated\SubjectUsage;
use GetStream\VisionAgents\Generated\Team;

/**
 * The organization's teams and their spend limits. Server side only.
 *
 * A team has no members list: a user is in the team their token names. A team's spend is the
 * spend of the work those tokens ran, across the organization or within one app.
 */
final readonly class Teams
{
    public function __construct(private Client $client, private SpendLimits $limits)
    {
    }

    /**
     * Every team with a limit or with spend this month, each with its apps, its spend in the
     * current windows and its limits. With an app id, only within that app.
     *
     * @return list<Team>
     */
    public function list(?string $appId = null): array
    {
        $listed = $this->client->get('/v1/teams', query: ['app_id' => $appId === '' ? null : $appId]);
        return array_map(Team::fromArray(...), Json::objects(['rows' => $listed], 'rows'));
    }

    /**
     * Makes the team's limit for the interval the amount given, across the organization or
     * within `$appId`: created when it has none, changed when it has one.
     */
    public function setLimit(string $team, LimitInterval|string $interval, Usd $amount, ?string $appId = null, ?string $note = null): Limit
    {
        if ($amount->micros < 0) {
            throw new ConfigurationException("a spend limit of {$amount} is below zero");
        }
        $existing = $this->limit($team, $interval, $appId);
        if ($existing === null) {
            try {
                return $this->limits->create(LimitSubject::Team, $interval, $amount, appId: $appId, team: $team, note: $note);
            } catch (RouterException $refused) {
                // Somebody else created it since it was looked up.
                if ($refused->errorCode !== 'limit_exists') {
                    throw $refused;
                }
            }
            $existing = $this->limit($team, $interval, $appId)
                ?? throw new ConfigurationException('the ' . Json::enumValue($interval) . " limit on team {$team} exists and could not be found");
        }
        return $this->limits->update($existing->id, $amount, $note);
    }

    /**
     * Deletes the team's limit for the interval, across the organization or within `$appId`.
     * Nothing happens when it has none.
     */
    public function removeLimit(string $team, LimitInterval|string $interval, ?string $appId = null): void
    {
        $existing = $this->limit($team, $interval, $appId);
        if ($existing === null) {
            return;
        }
        try {
            $this->limits->delete($existing->id);
        } catch (RouterException $refused) {
            // Somebody else deleted it since it was looked up.
            if ($refused->status !== 404) {
                throw $refused;
            }
        }
    }

    /**
     * What the team spent in its current windows, against its limits and grants, across the
     * organization or within `$appId`.
     */
    public function usage(string $team, ?string $appId = null): SubjectUsage
    {
        return $this->limits->usage(LimitSubject::Team, appId: $appId, team: $team);
    }

    private function limit(string $team, LimitInterval|string $interval, ?string $appId): ?Limit
    {
        UsageCategories::checkTeam($team);
        if ($interval === '') {
            throw new ConfigurationException('a team\'s limit needs an interval');
        }
        $appId = $appId === '' ? null : $appId;
        $wanted = Json::enumValue($interval);
        foreach ($this->limits->list(LimitSubject::Team, interval: $interval, appId: $appId, team: $team) as $limit) {
            if ($limit->team === $team && Json::enumValue($limit->interval) === $wanted && $limit->appId === $appId) {
                return $limit;
            }
        }
        return null;
    }
}
