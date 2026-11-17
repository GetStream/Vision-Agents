package client

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// Teams is the organization's teams. Server side only.
//
// A team has no members list: a user is in the team their token names. A team's spend is the
// spend of the work those tokens ran, across the organization or within one app.
type Teams struct {
	limits *SpendLimits
}

// Teams is the organization's teams and their spend limits.
func (c *Client) Teams() *Teams { return &Teams{limits: c.SpendLimits()} }

// TeamLimit is what one team may spend in each window of Interval: across the organization,
// or within AppID.
type TeamLimit struct {
	Team     string
	AppID    string
	Interval acceleration.LimitInterval
	Amount   USD
	Note     string
}

// List is every team with a limit or with spend this month, each with its apps, its spend in
// the current windows and its limits. With an appID, only within that app.
func (t *Teams) List(ctx context.Context, appID string) ([]acceleration.Team, error) {
	api, err := t.limits.client.api()
	if err != nil {
		return nil, err
	}
	listed, err := api.ListTeamsWithResponse(ctx, &acceleration.ListTeamsParams{AppId: pointer(appID)})
	if err != nil {
		return nil, fmt.Errorf("client: listing teams: %w", err)
	}
	if listed.JSON200 == nil {
		return nil, failure("listing teams", listed.HTTPResponse, listed.Body)
	}
	return *listed.JSON200, nil
}

// SetLimit makes the team's limit for the interval the amount given: created when it has
// none, changed when it has one.
func (t *Teams) SetLimit(ctx context.Context, limit TeamLimit) (acceleration.Limit, error) {
	if limit.Amount < 0 {
		return acceleration.Limit{}, fmt.Errorf("client: a spend limit of %s is below zero", limit.Amount)
	}
	existing, err := t.limit(ctx, limit.Team, limit.AppID, limit.Interval)
	if err != nil {
		return acceleration.Limit{}, err
	}
	if existing == nil {
		created, err := t.limits.Create(ctx, Limit{
			Subject: Team, Team: limit.Team, AppID: limit.AppID,
			Interval: limit.Interval, Amount: limit.Amount, Note: limit.Note,
		})
		var refused *stream.RouterError
		// Somebody else created it since it was looked up.
		if !errors.As(err, &refused) || refused.Code != "limit_exists" {
			return created, err
		}
		if existing, err = t.limit(ctx, limit.Team, limit.AppID, limit.Interval); err != nil {
			return acceleration.Limit{}, err
		}
		if existing == nil {
			return acceleration.Limit{}, fmt.Errorf("client: the %s limit on team %s exists and could not be found", limit.Interval, limit.Team)
		}
	}
	micros := limit.Amount.Micros()
	return t.limits.update(ctx, existing.Id, acceleration.LimitChange{LimitMicros: &micros, Note: pointer(limit.Note)})
}

// RemoveLimit deletes the team's limit for the interval, across the organization or within
// appID. Nothing happens when it has none.
func (t *Teams) RemoveLimit(ctx context.Context, team, appID string, interval acceleration.LimitInterval) error {
	existing, err := t.limit(ctx, team, appID, interval)
	if err != nil || existing == nil {
		return err
	}
	err = t.limits.Delete(ctx, existing.Id)
	var refused *stream.RouterError
	// Somebody else deleted it since it was looked up.
	if errors.As(err, &refused) && refused.Status == http.StatusNotFound {
		return nil
	}
	return err
}

// Usage is what the team spent in its current windows, against its limits and grants, across
// the organization or within appID.
func (t *Teams) Usage(ctx context.Context, team, appID string) (acceleration.SubjectUsage, error) {
	return t.limits.Usage(ctx, Target{Subject: Team, Team: team, AppID: appID})
}

// Grant gives the team a one-time amount past its spent limits, as SpendLimits.Grant does.
// Its Subject is Team whatever grant says.
func (t *Teams) Grant(ctx context.Context, grant Grant) (acceleration.LimitGrant, error) {
	grant.Subject = Team
	return t.limits.Grant(ctx, grant)
}

// limit is the team's own limit in the interval, across the organization or within the app.
func (t *Teams) limit(ctx context.Context, team, appID string, interval acceleration.LimitInterval) (*acceleration.Limit, error) {
	if err := checkTeam(team); err != nil {
		return nil, err
	}
	if interval == "" {
		return nil, errors.New("client: a team's limit needs an interval")
	}
	limits, err := t.limits.List(ctx, LimitFilter{Subject: Team, Team: team, AppID: appID, Interval: interval})
	if err != nil {
		return nil, err
	}
	for _, limit := range limits {
		inApp := ""
		if limit.AppId != nil {
			inApp = *limit.AppId
		}
		if limit.Team != nil && *limit.Team == team && limit.Interval == interval && inApp == appID {
			return &limit, nil
		}
	}
	return nil, nil
}
