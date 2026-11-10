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

// teamsGate is what keeps an app's configs of one team to its users: the teams of the end
// user asking, when the app has AgentTeams on and the caller is not its backend, which Chat
// does not limit by teams either. The zero gate refuses nothing.
type teamsGate struct {
	checked bool
	teams   []string
}

// teamsGateFor reads the settings and the caller's teams before any config is looked up, so
// that a failed read answers alike for a config that exists and one that does not.
func (s *Server) teamsGateFor(ctx context.Context, customerID string) (teamsGate, error) {
	if ServerSideFrom(ctx) {
		return teamsGate{}, nil
	}
	settings, err := s.configs.AppSettings(ctx, customerID)
	if err != nil {
		return teamsGate{}, err
	}
	if !settings.AgentTeamsChecked() {
		return teamsGate{}, nil
	}
	teams, err := s.callerTeams(ctx, customerID)
	if err != nil {
		return teamsGate{}, err
	}
	return teamsGate{checked: true, teams: teams}, nil
}

// refuses reports whether config is of a team the end user asking is not in, which the caller
// is told as it is of a config that does not exist.
func (g teamsGate) refuses(config store.AgentConfig) bool {
	return g.checked && !hasCommonTeam(g.teams, strings.TrimSpace(config.Tags[session.TeamTag]))
}

// configOpenedWith is the config a session was opened with, as it is now, for a fork or a
// reopen that names none: nil when it was deleted since. To an end user the gate checks, a
// config of another team is unknown, as configFor answers, and so is a deleted one, which
// leaves no team to check.
func (s *Server) configOpenedWith(ctx context.Context, customerID, id string, teams teamsGate) (*store.AgentConfig, error) {
	read, err := s.store.AgentConfig(ctx, customerID, id)
	switch {
	case errors.Is(err, store.ErrNoAgentConfig) && !teams.checked:
		return nil, nil
	case errors.Is(err, store.ErrNoAgentConfig):
		return nil, errUnknownConfig
	case err != nil:
		return nil, err
	case teams.refuses(read):
		return nil, errUnknownConfig
	}
	return &read, nil
}

// refuseOutsideTeam refuses a session opened with a config to an end user the gate checks
// and who is not in that config's team now. A session opened with no config has no team to
// check, and nobody else is limited, so for them the config is not read at all.
func (s *Server) refuseOutsideTeam(ctx context.Context, customerID, configID string) error {
	if configID == "" || s.store == nil {
		return nil
	}
	teams, err := s.teamsGateFor(ctx, customerID)
	if err != nil || !teams.checked {
		return err
	}
	_, err = s.configOpenedWith(ctx, customerID, configID, teams)
	return err
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
