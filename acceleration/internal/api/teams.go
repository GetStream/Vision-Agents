package api

import (
	"context"
	"errors"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errTeamsUnreadable refuses a config to an end user whose teams Stream did not tell, in an
// app that keeps each team's configs to its own users. Opening it anyway would let anybody in
// while Stream is down, and a 404 would say the config is gone when asking again may work.
var errTeamsUnreadable = unavailable("could not read the user's teams from Stream; try again shortly")

// errSharedStream is a customer acting in the deployment's shared Stream app.
var errSharedStream = errors.New("api: the customer has no Stream app of its own to read teams from")

// refuseOtherTeam answers unknown, what the caller is told of a config that does not exist,
// when config is of a team the end user asking is not in. Only an app with AgentTeams on is
// checked, and never its backend, which Chat does not limit by teams either.
func (s *Server) refuseOtherTeam(ctx context.Context, customerID string, config store.AgentConfig, unknown error) error {
	if ServerSideFrom(ctx) {
		return nil
	}
	settings, err := s.configs.AppSettings(ctx, customerID)
	if err != nil {
		return err
	}
	if !settings.AgentTeamsChecked() {
		return nil
	}
	teams, err := s.callerTeams(ctx, customerID)
	if err != nil {
		return err
	}
	if !hasCommonTeam(teams, strings.TrimSpace(config.Tags[session.TeamTag])) {
		return unknown
	}
	return nil
}

// callerTeams is the teams of the end user asking, as their user in the customer's own
// Stream app holds them. In the deployment's shared app, user ids are not the customer's and
// whoever holds that app writes their teams, so a customer acting there reads none. A caller
// with no token is in none: the user id it goes by is a claim nobody checked,
// and reading that user's teams would let anybody borrow them.
func (s *Server) callerTeams(ctx context.Context, customerID string) ([]string, error) {
	caller := CallerFrom(ctx)
	if !caller.Verified {
		return nil, nil
	}
	bound, ok, err := s.streamFor(ctx, customerID)
	switch {
	case err == nil && !ok:
		err = errNoStream
	case err == nil && !bound.Identity.Registered:
		err = errSharedStream
	}
	if err == nil {
		var teams []string
		if teams, err = conversation.UserTeams(ctx, bound.Client, caller.UserID); err == nil {
			return teams, nil
		}
	}
	s.logger.Error("could not read a user's teams from Stream", "user", caller.UserID, "error", err)
	return nil, errTeamsUnreadable
}

// hasCommonTeam is Chat's shared-team rule for a config, which is in one team or none: a
// config of no team is open to a user in no team, and one of a team to the users in it.
func hasCommonTeam(teams []string, team string) bool {
	if team == "" {
		return len(teams) == 0
	}
	return slices.Contains(teams, team)
}
