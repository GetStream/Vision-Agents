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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// AgentTeamsSuite covers an app that keeps each team's agent configs to its own users
// (AppSettings.AgentTeams): an end user opens a config only if they share a team with it,
// by Chat's shared-team rule, their teams read from their user in the app's Stream.
type AgentTeamsSuite struct {
	RouterSuite
	// teamsChat is the customer's own Stream app, where its users and their teams live.
	teamsChat *chattest.Server
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
