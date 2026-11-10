//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AgentTeamsSuite covers an app that keeps each team's agent configs to its own users
// (AppSettings.AgentTeams): an end user opens a config only if they share a team with it,
// by Chat's shared-team rule, their teams read from their user in the app's Stream.
type AgentTeamsSuite struct {
	RouterSuite
	// teamsChat is the customer's own Stream app, where its users and their teams live.
	teamsChat *chattest.Server
	logged    *lockedLog
}

func (s *AgentTeamsSuite) SetupSuite() {
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

const teamsAppKey = "own-key"

func TestAgentTeamsSuite(t *testing.T) {
	runSuite(t, new(AgentTeamsSuite))
}

func (s *AgentTeamsSuite) SetupTest() {
	s.useTeamsApp(store.AppSettings{AgentTeams: new(true)})
}

// useTeamsApp is an app with settings whose customer acts in a Stream app of its own, the
// only one whose users' teams the customer's backend sets.
func (s *AgentTeamsSuite) useTeamsApp(settings store.AppSettings) {
	s.useApp(s.data.createAppAdmitting(settings))
	s.teamsChat = s.giveApp(s.customerID(), 4242, teamsAppKey)
}

func (s *AgentTeamsSuite) TestAUserOpensAConfigOfTheirTeamAskingStreamEachTime() {
	anya := s.inTeams("red")
	config := s.configOf("red")
	asked := s.teamReads()

	s.Equal(http.StatusCreated, anya.do(http.MethodPost, "/v1/agents/sessions", s.naming(config), nil))
	s.Equal(http.StatusCreated, anya.do(http.MethodPost, "/v1/agents/sessions", s.naming(config), nil))
	s.Equal(asked+2, s.teamReads(), "no teams are kept between sessions")
}

func (s *AgentTeamsSuite) TestAConfigOfAnotherTeamIsAnsweredAsAnUnknownOne() {
	anya := s.inTeams("red")

	status, payload := anya.call(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("blue")))
	unknownStatus, unknown := anya.call(http.MethodPost, "/v1/agents/sessions", s.naming(AgentConfig{Id: s.utils.uuid()}))

	s.Equal(http.StatusNotFound, status, string(payload))
	s.Equal(unknownStatus, status)
	var refused, missing ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &refused))
	s.Require().NoError(json.Unmarshal(unknown, &missing))
	s.Equal(missing, refused, "the same answer as a config that does not exist")
}

func (s *AgentTeamsSuite) TestAConfigOfAnotherTeamNamedByNameIsAnsweredAsAnUnknownName() {
	anya := s.inTeams("red")
	config := s.configOf("blue")

	status, message := anya.failure(http.MethodPost, "/v1/agents/sessions", CreateSessionRequest{Agent: &config.Name})

	s.Equal(http.StatusNotFound, status, message)
	s.Equal("there is no agent called "+config.Name, message)
}

func (s *AgentTeamsSuite) TestAUserInSeveralTeamsOpensAConfigOfAnyOfThem() {
	carl := s.inTeams("red", "blue")

	s.Equal(http.StatusCreated, carl.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("blue")), nil))
	s.Equal(http.StatusCreated, carl.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("red")), nil))
	s.Equal(http.StatusNotFound, carl.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("green")), nil))
}

func (s *AgentTeamsSuite) TestAConfigsTeamIsReadAsItsUsageTeamIsWithoutTheSpacesAround() {
	anya := s.inTeams("red")

	s.Equal(http.StatusCreated, anya.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf(" red ")), nil))
}

func (s *AgentTeamsSuite) TestAConfigOfNoTeamIsClosedToAUserInATeam() {
	anya := s.inTeams("red")

	s.Equal(http.StatusNotFound, anya.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("")), nil))
}

func (s *AgentTeamsSuite) TestAUserInNoTeamOpensOnlyConfigsOfNoTeam() {
	dana := s.inTeams()

	s.Equal(http.StatusCreated, dana.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("")), nil))
	s.Equal(http.StatusNotFound, dana.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("red")), nil))
}

func (s *AgentTeamsSuite) TestAUserStreamDoesNotHaveIsInNoTeam() {
	stranger := s.data.createUser()

	s.Equal(http.StatusCreated, stranger.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("")), nil))
	s.Equal(http.StatusNotFound, stranger.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("red")), nil))
}

func (s *AgentTeamsSuite) TestAGuestIsCheckedByTheTeamsOfTheirStreamUser() {
	guest := s.data.createGuest()
	s.teamsChat.PutUser(map[string]any{"id": guest.userID, "teams": []any{"red"}})

	s.Equal(http.StatusCreated, guest.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("red")), nil))
	s.Equal(http.StatusNotFound, guest.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("")), nil))
}

// An anonymous caller goes by any user id it likes, so it cannot take that user's teams.
func (s *AgentTeamsSuite) TestACallerWithNoTokenIsInNoTeamWhoeverItClaimsToBe() {
	anya := s.inTeams("red")
	claiming := s.data.claiming(anya.userID)
	asked := s.teamReads()

	s.Equal(http.StatusNotFound, claiming.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("red")), nil))
	s.Equal(http.StatusCreated, claiming.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("")), nil))
	s.Equal(asked, s.teamReads(), "Stream is not asked about the user it claims to be")
}

func (s *AgentTeamsSuite) TestAFailedTeamsReadOpensNothing() {
	dana := s.inTeams()
	config := s.configOf("")
	s.teamsChat.Fail(http.MethodGet, "/users")
	s.T().Cleanup(func() { s.teamsChat.Heal(http.MethodGet, "/users") })

	status, payload := dana.call(http.MethodPost, "/v1/agents/sessions", s.naming(config))

	s.Equal(http.StatusServiceUnavailable, status, string(payload))
	s.Contains(string(payload), errTeamsUnreadable.Message)
}

// What Stream failing tells must not depend on whether the name or id exists, or the failure
// would let anybody list the configs of the app.
func (s *AgentTeamsSuite) TestAFailedReadAnswersAlikeForAConfigThatExistsAndOneThatDoesNot() {
	dana := s.inTeams()
	config := s.configOf("")
	missing := AgentConfig{Id: s.utils.uuid(), Name: "nope-" + s.utils.uuid()}
	s.teamsChat.Fail(http.MethodGet, "/users")
	s.T().Cleanup(func() { s.teamsChat.Heal(http.MethodGet, "/users") })

	for name, pair := range map[string][2]CreateSessionRequest{
		"by id":   {s.naming(config), s.naming(missing)},
		"by name": {{Agent: &config.Name}, {Agent: &missing.Name}},
	} {
		existsStatus, exists := dana.failure(http.MethodPost, "/v1/agents/sessions", pair[0])
		missingStatus, absent := dana.failure(http.MethodPost, "/v1/agents/sessions", pair[1])
		s.Equal(http.StatusServiceUnavailable, existsStatus, name)
		s.Equal(existsStatus, missingStatus, name)
		s.Equal(exists, absent, name)
	}
}

func (s *AgentTeamsSuite) TestAnAppWhoseSettingsCannotBeReadAnswersAlikeForAnUnknownConfig() {
	dana := s.inTeams()
	config := s.configOf("")
	_, err := s.store.DB().NewRaw(`UPDATE apps SET settings = '{"agent_teams": "unreadable"}'::jsonb WHERE id = ?`,
		s.customerID()).Exec(s.T().Context())
	s.Require().NoError(err)

	existsStatus, exists := dana.failure(http.MethodPost, "/v1/agents/sessions", s.naming(config))
	missingStatus, absent := dana.failure(http.MethodPost, "/v1/agents/sessions", s.naming(AgentConfig{Id: s.utils.uuid()}))

	s.Equal(http.StatusInternalServerError, existsStatus)
	s.Equal(existsStatus, missingStatus)
	s.Equal(exists, absent)
}

func (s *AgentTeamsSuite) TestAnAppWithNoStreamToReadTeamsFromOpensNothingToAUser() {
	dana := s.inTeams()
	config := s.configOf("")
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.nowhere[s.customerID()] = true })

	status, message := dana.failure(http.MethodPost, "/v1/agents/sessions", s.naming(config))

	s.Equal(http.StatusServiceUnavailable, status, message)
	s.Equal(errTeamsUnreadable.Message, message)
}

// In the deployment's shared app the user ids are nobody's own and its teams are written by
// whoever holds that app, so a customer acting there cannot tell its users' teams from it.
func (s *AgentTeamsSuite) TestACustomerInTheDeploymentsSharedAppOpensNothingToAUser() {
	anya := s.inTeams("red")
	config := s.configOf("red")
	s.chat.PutUser(map[string]any{"id": anya.userID, "teams": []any{"red"}})
	s.setApps(s.customerID(), func(apps *suiteApps) { delete(apps.own, s.customerID()) })

	status, message := anya.failure(http.MethodPost, "/v1/agents/sessions", s.naming(config))

	s.Equal(http.StatusServiceUnavailable, status, message)
	s.Equal(errTeamsUnreadable.Message, message)
}

func (s *AgentTeamsSuite) TestAnAppWhoseSettingsCannotBeReadOpensNothingToAUser() {
	dana := s.inTeams()
	config := s.configOf("")
	_, err := s.store.DB().NewRaw(`UPDATE apps SET settings = '{"agent_teams": "unreadable"}'::jsonb WHERE id = ?`,
		s.customerID()).Exec(s.T().Context())
	s.Require().NoError(err)

	s.Equal(http.StatusInternalServerError, dana.do(http.MethodPost, "/v1/agents/sessions", s.naming(config), nil))
	s.Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions", s.naming(config), nil),
		"a backend is not limited by teams, so it needs no settings")
}

// Chat compares team names exactly.
func (s *AgentTeamsSuite) TestATeamIsMatchedExactlyNotByCase() {
	anya := s.inTeams("red")

	s.Equal(http.StatusNotFound, anya.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("Red")), nil))
}

func (s *AgentTeamsSuite) TestABackendOpensAConfigOfAnyTeamWithoutAskingStream() {
	anya := s.inTeams("red")
	asked := s.teamReads()

	s.Equal(http.StatusCreated, s.serverClient.actingFor(anya).do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("blue")), nil))
	s.Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions", s.naming(s.configOf("")), nil))
	s.Equal(asked, s.teamReads())
}

func (s *AgentTeamsSuite) TestWithTheSettingOffAUserOpensAnotherTeamsConfigAndStreamIsNotAsked() {
	for _, settings := range []store.AppSettings{{}, {AgentTeams: new(false)}} {
		s.useTeamsApp(settings)
		anya := s.inTeams("red")
		config := s.configOf("blue")
		asked := s.teamReads()

		s.Equal(http.StatusCreated, anya.do(http.MethodPost, "/v1/agents/sessions", s.naming(config), nil), "settings %+v", settings)
		s.Equal(asked, s.teamReads(), "settings %+v", settings)
	}
}

func (s *AgentTeamsSuite) TestAForkNamingAConfigOfAnotherTeamIsAnsweredAsAnUnknownOne() {
	anya := s.inTeams("red")
	parent := anya.createSession(s.recorded(s.configOf("red")))
	blue := s.configOf("blue")

	status, message := anya.failure(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork", ForkSessionRequest{ConfigId: &blue.Id})

	s.Equal(http.StatusNotFound, status, message)
	s.Equal(errUnknownConfig.Message, message)
}

func (s *AgentTeamsSuite) TestASocketSessionWithAConfigOfAnotherTeamIsRefused() {
	anya := s.inTeams("red")
	config := s.configOf("blue")
	connection := anya.opens("/v1/agents/socket")
	s.Require().NoError(connection.WriteJSON(map[string]any{
		"type": "start", "sample_rate": 24_000, "session": map[string]any{"config_id": config.Id},
	}))

	var refused map[string]any
	s.Require().NoError(connection.ReadJSON(&refused))

	s.Equal("error", refused["type"], "%v", refused)
	s.Equal(errUnknownConfig.Message, refused["error"])
}

// A conversation carried on in another process is checked again: its user moved to another
// team since it was opened, so its config is no longer theirs.
func (s *AgentTeamsSuite) TestAChatCarriedOnFromItsRowIsCheckedAgainstTheUsersTeamsNow() {
	anya := s.inTeams("red")
	opened := anya.createSession(s.recorded(s.configOf("red")))
	anya.stopSession(opened.Id)
	_, other := s.otherRouter(nil)
	ctx := context.WithValue(context.Background(), customerContextKey{}, s.customerID())
	ctx = context.WithValue(ctx, kindContextKey{}, auth.KindAuthenticated)
	ctx = routing.WithCaller(ctx, routing.Caller{UserID: anya.userID, Verified: true})

	_, failure := other.reopenedFromRow(ctx, opened.Id)
	s.Require().NoError(failure)

	s.teamsChat.PutUser(map[string]any{"id": anya.userID, "teams": []any{"blue"}})
	_, failure = other.reopenedFromRow(ctx, opened.Id)
	s.Equal(errUnknownConfig, failure)
}

// A fork that names no config goes on with its parent's, which is checked as if it had been
// named: Anya moved to team blue since she opened her red session, live or ended.
func (s *AgentTeamsSuite) TestAForkOfASessionWhoseConfigIsOfAnotherTeamNowIsRefused() {
	for _, ended := range []bool{false, true} {
		anya := s.inTeams("red")
		parent := anya.createSession(s.recorded(s.configOf("red")))
		s.Equal(http.StatusCreated, anya.do(http.MethodPost, s.forking(parent.Id), s.offRecord(), nil),
			"ended %v: still in its team", ended)
		if ended {
			anya.stopSession(parent.Id)
		}
		s.teamsChat.PutUser(map[string]any{"id": anya.userID, "teams": []any{"blue"}})

		status, message := anya.failure(http.MethodPost, s.forking(parent.Id), s.offRecord())

		s.Equal(http.StatusNotFound, status, "ended %v: %s", ended, message)
		s.Equal(errUnknownConfig.Message, message, "ended %v", ended)
	}
}

// A config deleted since leaves no team to check the fork against.
func (s *AgentTeamsSuite) TestAForkOfASessionWhoseConfigWasDeletedIsRefused() {
	for _, ended := range []bool{false, true} {
		anya := s.inTeams("red")
		parent := anya.createSession(s.recorded(s.configOf("red")))
		if ended {
			anya.stopSession(parent.Id)
		}
		s.deleteConfig(value(parent.ConfigId))

		status, message := anya.failure(http.MethodPost, s.forking(parent.Id), s.offRecord())

		s.Equal(http.StatusNotFound, status, "ended %v: %s", ended, message)
		s.Equal(errUnknownConfig.Message, message, "ended %v", ended)
	}
}

// What Stream failing tells must not say whether the parent's config is still there.
func (s *AgentTeamsSuite) TestAFailedTeamsReadAnswersAForkAlikeWhetherItsConfigExistsOrNot() {
	anya := s.inTeams("red")
	kept := anya.createSession(s.recorded(s.configOf("red")))
	deleted := anya.createSession(s.recorded(s.configOf("red")))
	s.deleteConfig(value(deleted.ConfigId))
	s.teamsChat.Fail(http.MethodGet, "/users")
	s.T().Cleanup(func() { s.teamsChat.Heal(http.MethodGet, "/users") })

	keptStatus, keptMessage := anya.failure(http.MethodPost, s.forking(kept.Id), s.offRecord())
	deletedStatus, deletedMessage := anya.failure(http.MethodPost, s.forking(deleted.Id), s.offRecord())

	s.Equal(http.StatusServiceUnavailable, keptStatus, keptMessage)
	s.Equal(errTeamsUnreadable.Message, keptMessage)
	s.Equal(keptStatus, deletedStatus)
	s.Equal(keptMessage, deletedMessage)
}

func (s *AgentTeamsSuite) TestAForkOfASessionOpenedWithNoConfigIsNotChecked() {
	anya := s.inTeams("red")
	parent := anya.createSession(textSession(nil))
	asked := s.teamReads()

	s.Equal(http.StatusCreated, anya.do(http.MethodPost, s.forking(parent.Id), s.offRecord(), nil))
	s.Equal(asked, s.teamReads())
}

// A chat whose session ended in this node is carried on with the spec it ran on, which is
// checked against the config it was opened with as it is now, on every way of writing to it.
func (s *AgentTeamsSuite) TestAChatThatEndedHereIsCarriedOnOnlyForAUserStillInItsConfigsTeam() {
	for path, body := range map[string]any{"/responses": s.asking(), "/voice": nil} {
		anya := s.inTeams("red")
		opened := anya.createSession(s.recorded(s.configOf("red")))
		s.endHere(opened.Id)
		s.Equal(http.StatusAccepted, s.status(anya.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses", s.asking())),
			"%s: still in its team", path)
		s.endHere(opened.Id)
		s.teamsChat.PutUser(map[string]any{"id": anya.userID, "teams": []any{"blue"}})

		status, message := anya.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+path, body)

		s.Equal(http.StatusNotFound, status, "%s: %s", path, message)
		s.Equal(errUnknownConfig.Message, message, path)
	}
}

func (s *AgentTeamsSuite) TestAChatThatEndedHereWhoseConfigWasDeletedIsNotCarriedOn() {
	anya := s.inTeams("red")
	opened := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(opened.Id)
	s.deleteConfig(value(opened.ConfigId))

	status, message := anya.failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses", s.asking())

	s.Equal(http.StatusNotFound, status, message)
	s.Equal(errUnknownConfig.Message, message)
}

func (s *AgentTeamsSuite) TestAFailedTeamsReadAnswersAReopenAlikeWhetherItsConfigExistsOrNot() {
	anya := s.inTeams("red")
	kept := anya.createSession(s.recorded(s.configOf("red")))
	deleted := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(kept.Id)
	s.endHere(deleted.Id)
	s.deleteConfig(value(deleted.ConfigId))
	s.teamsChat.Fail(http.MethodGet, "/users")
	s.T().Cleanup(func() { s.teamsChat.Heal(http.MethodGet, "/users") })

	keptStatus, keptMessage := anya.failure(http.MethodPost, "/v1/agents/sessions/"+kept.Id+"/responses", s.asking())
	deletedStatus, deletedMessage := anya.failure(http.MethodPost, "/v1/agents/sessions/"+deleted.Id+"/responses", s.asking())

	s.Equal(http.StatusServiceUnavailable, keptStatus, keptMessage)
	s.Equal(errTeamsUnreadable.Message, keptMessage)
	s.Equal(keptStatus, deletedStatus)
	s.Equal(keptMessage, deletedMessage)
}

func (s *AgentTeamsSuite) TestAChatThatEndedHereOpenedWithNoConfigIsNotChecked() {
	anya := s.inTeams("red")
	opened := anya.createSession(textSession(nil))
	s.endHere(opened.Id)
	asked := s.teamReads()

	s.Equal(http.StatusAccepted, s.status(anya.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses", s.asking())))
	s.Equal(asked, s.teamReads())
}

// A config that cannot be read is not taken for a deleted one: the fork and the reopen fail
// rather than open on what the user is not checked against. With the setting off a reopen does
// not read it: a fork answered 500 and a reopen 202 on base b4d31ea60, as they do here.
func (s *AgentTeamsSuite) TestAConfigThatCannotBeReadFailsAReopenOnlyWhenChecked() {
	for settings, reopened := range map[*bool]int{new(true): http.StatusInternalServerError, nil: http.StatusAccepted} {
		s.useTeamsApp(store.AppSettings{AgentTeams: settings})
		anya := s.inTeams("red")
		parent := anya.createSession(s.recorded(s.configOf("red")))
		ended := anya.createSession(s.recorded(s.configOf("red")))
		s.endHere(ended.Id)
		for _, opened := range []Session{parent, ended} {
			_, err := s.store.DB().NewRaw(`UPDATE agent_configs SET skills = '"unreadable"'::jsonb WHERE id = ?`,
				value(opened.ConfigId)).Exec(s.T().Context())
			s.Require().NoError(err)
		}

		s.Equal(http.StatusInternalServerError, s.status(anya.call(http.MethodPost, s.forking(parent.Id), s.offRecord())), "flag %v", settings != nil)
		s.Equal(reopened, s.status(anya.call(http.MethodPost, "/v1/agents/sessions/"+ended.Id+"/responses", s.asking())), "flag %v", settings != nil)
	}
}

// The socket reopens a chat that ended here as the REST writes do (sessionws.go:123), so
// a user who moved to another team is answered as for an unknown config.
func (s *AgentTeamsSuite) TestASocketReopenOfAChatThatEndedHereIsCheckedAgainstTheUsersTeamsNow() {
	anya := s.inTeams("red")
	opened := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(opened.Id)
	_, status := anya.watch("/v1/agents/sessions/" + opened.Id + "/events")
	s.Equal(http.StatusSwitchingProtocols, status, "still in its team")
	s.endHere(opened.Id)
	s.teamsChat.PutUser(map[string]any{"id": anya.userID, "teams": []any{"blue"}})

	status, message := anya.refusal("/v1/agents/sessions/" + opened.Id + "/events")

	s.Equal(http.StatusNotFound, status, message)
	s.Equal(errUnknownConfig.Message, message)
}

func (s *AgentTeamsSuite) TestAFailedTeamsReadAnswersASocketReopenAlikeWhetherItsConfigExistsOrNot() {
	anya := s.inTeams("red")
	kept := anya.createSession(s.recorded(s.configOf("red")))
	deleted := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(kept.Id)
	s.endHere(deleted.Id)
	s.deleteConfig(value(deleted.ConfigId))
	s.teamsChat.Fail(http.MethodGet, "/users")
	s.T().Cleanup(func() { s.teamsChat.Heal(http.MethodGet, "/users") })

	keptStatus, keptMessage := anya.refusal("/v1/agents/sessions/" + kept.Id + "/events")
	deletedStatus, deletedMessage := anya.refusal("/v1/agents/sessions/" + deleted.Id + "/events")

	s.Equal(http.StatusServiceUnavailable, keptStatus, keptMessage)
	s.Equal(errTeamsUnreadable.Message, keptMessage)
	s.Equal(keptStatus, deletedStatus)
	s.Equal(keptMessage, deletedMessage)
}

func (s *AgentTeamsSuite) TestASocketReopenOfAChatWhoseConfigWasDeletedIsAnsweredAsAnUnknownConfig() {
	anya := s.inTeams("red")
	gone := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(gone.Id)
	s.deleteConfig(value(gone.ConfigId))

	status, message := anya.refusal("/v1/agents/sessions/" + gone.Id + "/events")

	s.Equal(http.StatusNotFound, status, message)
	s.Equal(errUnknownConfig.Message, message)
}

func (s *AgentTeamsSuite) TestWithTheSettingOffASocketReopenIsNotCheckedAndStreamIsNotAsked() {
	s.useTeamsApp(store.AppSettings{})
	anya := s.inTeams("blue")
	opened := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(opened.Id)
	gone := anya.createSession(s.recorded(s.configOf("red")))
	s.endHere(gone.Id)
	s.deleteConfig(value(gone.ConfigId))
	asked := s.teamReads()

	_, status := anya.watch("/v1/agents/sessions/" + opened.Id + "/events")
	s.Equal(http.StatusSwitchingProtocols, status)
	_, status = anya.watch("/v1/agents/sessions/" + gone.Id + "/events")
	s.Equal(http.StatusSwitchingProtocols, status)
	s.Equal(asked, s.teamReads())
}

// A backend forks and carries on any session of another team or a deleted config, without
// asking Stream, as it opens any config.
func (s *AgentTeamsSuite) TestABackendForksAndCarriesOnASessionOfAnyTeamWithoutAskingStream() {
	backend := s.serverClient.actingFor(s.inTeams("blue"))
	opened := backend.createSession(s.recorded(s.configOf("red")))
	other := backend.createSession(s.recorded(s.configOf("red")))
	s.deleteConfig(value(other.ConfigId))
	asked := s.teamReads()

	for _, parent := range []Session{opened, other} {
		status, payload := backend.call(http.MethodPost, s.forking(parent.Id), s.offRecord())
		s.Equal(http.StatusCreated, status, string(payload))
		s.endHere(parent.Id)
		status, payload = backend.call(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/responses", s.asking())
		s.Equal(http.StatusAccepted, status, string(payload))
	}
	s.Equal(asked, s.teamReads())
}

// With the setting off, forks and reopens of another team's or a deleted config's session go
// on as on base b4d31ea60, which answered these requests 201 and 202 asking Stream nothing.
func (s *AgentTeamsSuite) TestWithTheSettingOffForksAndReopensAreNotCheckedAndStreamIsNotAsked() {
	for _, settings := range []store.AppSettings{{}, {AgentTeams: new(false)}} {
		s.useTeamsApp(settings)
		anya := s.inTeams("blue")
		live := anya.createSession(s.recorded(s.configOf("red")))
		stopped := anya.createSession(s.recorded(s.configOf("red")))
		anya.stopSession(stopped.Id)
		deleted := anya.createSession(s.recorded(s.configOf("red")))
		s.deleteConfig(value(deleted.ConfigId))
		ended := anya.createSession(s.recorded(s.configOf("red")))
		s.endHere(ended.Id)
		gone := anya.createSession(s.recorded(s.configOf("red")))
		s.endHere(gone.Id)
		s.deleteConfig(value(gone.ConfigId))
		asked := s.teamReads()

		for _, parent := range []Session{live, stopped, deleted} {
			s.Equal(http.StatusCreated, anya.do(http.MethodPost, s.forking(parent.Id), s.offRecord(), nil), "settings %+v", settings)
		}
		for _, opened := range []Session{ended, gone} {
			s.Equal(http.StatusAccepted, s.status(anya.call(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses", s.asking())),
				"settings %+v", settings)
		}
		s.Equal(asked, s.teamReads(), "settings %+v", settings)
	}
}

// endHere ends a session in this node the way one nobody watches ends, which leaves it held
// under its id, ended, with the spec it ran on.
func (s *AgentTeamsSuite) endHere(id string) {
	held, ok := s.manager.Get(id, session.Owner{CustomerID: s.customerID(), Kind: auth.KindServer})
	s.Require().True(ok, "session %s is held here", id)
	s.Require().NoError(held.Close())
	s.Require().Equal(session.Ended, held.State())
}

// deleteConfig deletes an agent config of the app, as its backend does.
func (s *AgentTeamsSuite) deleteConfig(id string) {
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+id, nil, nil))
}

// offRecord is a fork kept off the record and started from nothing, which opens no Stream
// channel and so asks Stream nothing about its users: the only user query it makes is the
// teams read.
func (s *AgentTeamsSuite) offRecord() ForkSessionRequest {
	return ForkSessionRequest{Incognito: pointerTo(true), Messages: pointerTo(false)}
}

// asking is a question from an end user's device, which names its request.
func (s *AgentTeamsSuite) asking() CreateResponseRequest {
	return CreateResponseRequest{Text: "Still there?", RequestId: pointerTo(s.utils.uuid())}
}

// forking is the path that forks the session id.
func (s *AgentTeamsSuite) forking(id string) string {
	return "/v1/agents/sessions/" + id + "/fork"
}

// status is the status of a call, for an assertion that reads only that.
func (s *AgentTeamsSuite) status(status int, _ []byte) int {
	return status
}

// inTeams is a new end user of the app whose Stream user is in teams, none for no teams.
func (s *AgentTeamsSuite) inTeams(teams ...string) *testClient {
	person := s.data.createUser()
	user := map[string]any{"id": person.userID}
	if len(teams) > 0 {
		listed := make([]any, 0, len(teams))
		for _, team := range teams {
			listed = append(listed, team)
		}
		user["teams"] = listed
	}
	s.teamsChat.PutUser(user)
	return person
}

// configOf is a new agent config of the app tagged with team, untagged for "".
func (s *AgentTeamsSuite) configOf(team string) AgentConfig {
	tags := map[string]string{}
	if team != "" {
		tags["team"] = team
	}
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name: "agent-" + s.utils.uuid(), Llm: pointerTo("en-low-latency"), Tags: &tags,
	}, &created))
	return created
}

// naming is an incognito text session with config, which opens no Stream channel and so
// asks Stream nothing about its users: the only user query it makes is the teams read.
func (s *AgentTeamsSuite) naming(config AgentConfig) CreateSessionRequest {
	request := s.recorded(config)
	request.Incognito = pointerTo(true)
	return request
}

// recorded is a text session with config that is kept, for a test that needs its row.
func (s *AgentTeamsSuite) recorded(config AgentConfig) CreateSessionRequest {
	request := textSession(nil)
	request.ConfigId = &config.Id
	return request
}

// teamReads is how many user queries the suite's Stream has been sent.
func (s *AgentTeamsSuite) teamReads() int {
	read := 0
	for _, request := range s.teamsChat.Requests(teamsAppKey) {
		if request.Method == http.MethodGet && strings.HasSuffix(request.Path, "/users") {
			read++
		}
	}
	return read
}
