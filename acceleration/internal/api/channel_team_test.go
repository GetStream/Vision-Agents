//go:build integration

package api

import (
	"maps"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv/redisready"
)

// What a kept session's new channel is made with in an app with the setting off, read from
// the stored channel data of the router before agent channels had a team: no team field, and
// a caller's custom team left out.
var (
	channelFieldsBefore = []string{"created_by_id", "custom", "members"}
	channelCustomBefore = []string{
		"plan", "support_agent_id", "support_customer_id", "support_memory_scope", "support_owner_id", "support_trigger",
	}
)

func (s *AgentTeamsSuite) TestAChannelOfATeamsConfigIsMadeInThatTeam() {
	anya := s.inTeams("red")

	s.Equal("red", s.channelOf(anya.createSession(s.recorded(s.configOf("red"))))["team"])
	s.Equal("blue", s.channelOf(s.serverClient.createSession(s.recorded(s.configOf("blue"))))["team"],
		"a backend's session is not limited by teams, and its channel is in the config's team still")
}

func (s *AgentTeamsSuite) TestAChannelsTeamIsTheConfigsWithoutTheSpacesAround() {
	s.Equal("red", s.channelOf(s.serverClient.createSession(s.recorded(s.configOf(" red "))))["team"])
}

func (s *AgentTeamsSuite) TestAChannelOfAConfigOfNoTeamIsMadeAsBefore() {
	dana := s.inTeams()

	s.madeAsBefore(s.channelOf(dana.createSession(s.withCustom(s.recorded(s.configOf(""))))))
}

func (s *AgentTeamsSuite) TestACallerCannotNameTheTeamOfTheChannel() {
	anya := s.inTeams("red")

	s.Equal("red", s.channelOf(anya.createSession(s.withCustom(s.recorded(s.configOf("red")))))["team"])
}

func (s *AgentTeamsSuite) TestWithTheSettingOffAChannelIsMadeAsBefore() {
	for _, settings := range []store.AppSettings{{}, {AgentTeams: new(false)}} {
		s.useTeamsApp(settings)
		anya := s.inTeams("red")

		s.madeAsBefore(s.channelOf(anya.createSession(s.withCustom(s.recorded(s.configOf("red"))))))
		s.madeAsBefore(s.channelOf(s.serverClient.createSession(s.withCustom(s.recorded(s.configOf("red"))))))
	}
}

// A fork carries its parent's spec over, the channel team included, and the app's setting is
// read again for its own channel.
func (s *AgentTeamsSuite) TestAForkIsMadeInATeamOnlyWhileTheAppSaysSo() {
	parent := s.serverClient.createSession(s.withCustom(s.recorded(s.configOf("red"))))
	s.Require().Equal("red", s.channelOf(parent)["team"])
	s.turnTeamsOff()

	var forked Session
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork",
		ForkSessionRequest{}, &forked))

	s.madeAsBefore(s.channelOf(forked))
}

// Without the setting the channel could not be told apart from one of no team, which every
// user of the app without a team would be let into.
func (s *AgentTeamsSuite) TestASessionOfATeamsConfigIsRefusedWhenTheSettingsCannotBeRead() {
	config := s.configOf("red")
	_, err := s.store.DB().NewRaw(`UPDATE apps SET settings = '{"agent_teams": "unreadable"}'::jsonb WHERE id = ?`,
		s.customerID()).Exec(s.T().Context())
	s.Require().NoError(err)

	sessionID := s.utils.uuid()
	request := s.recorded(config)
	request.Id = &sessionID

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sessions", request)
	s.Equal(http.StatusInternalServerError, status, string(payload))
	s.NotContains(string(payload), "Scan", "the store's error text stays out of the answer")
	s.NotContains(string(payload), "unreadable")
	_, made := s.teamsChat.Channel(sessionID)
	s.False(made, "no channel is made outside the config's team")
}

// A fork reads the app's settings again, and a read that fails is answered as the create is.
func (s *AgentTeamsSuite) TestAForkIsRefusedWhenTheSettingsCannotBeRead() {
	parent := s.serverClient.createSession(s.withCustom(s.recorded(s.configOf("red"))))
	_, err := s.store.DB().NewRaw(`UPDATE apps SET settings = '{"agent_teams": "unreadable"}'::jsonb WHERE id = ?`,
		s.customerID()).Exec(s.T().Context())
	s.Require().NoError(err)
	client, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{os.Getenv("ROUTER_REDIS_ADDR")}, DisableCache: true})
	s.Require().NoError(err)
	defer client.Close()
	s.Require().NoError(client.Do(s.T().Context(), client.B().Del().Key("appcfg:app-settings:"+s.customerID()).Build()).Error())
	s.Require().Eventually(func() bool {
		_, err := s.configs.AppSettings(s.T().Context(), s.customerID())
		return err != nil
	}, 10*time.Second, 20*time.Millisecond)

	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/sessions/"+parent.Id+"/fork", ForkSessionRequest{})
	s.Equal(http.StatusInternalServerError, status, string(payload))
	s.NotContains(string(payload), "Scan", "the store's error text stays out of the answer")
}

// A backend may name a team of its own on the request; the channel is still the config's.
func (s *AgentTeamsSuite) TestABackendsOwnTeamDoesNotMoveTheChannel() {
	request := s.recorded(s.configOf("red"))
	request.Team = pointerTo("blue")

	s.Equal("red", s.channelOf(s.serverClient.createSession(request))["team"])
}

// channelOf is what the channel a kept session holds its conversation in was made with.
func (s *AgentTeamsSuite) channelOf(opened Session) map[string]any {
	data, found := s.teamsChat.Channel(strings.TrimPrefix(value(opened.ConversationId), "agent:"))
	s.Require().True(found, "session %s made no channel in its customer's app", opened.Id)
	return data
}

// withCustom is request with custom data naming a team of its own.
func (s *AgentTeamsSuite) withCustom(request CreateSessionRequest) CreateSessionRequest {
	request.Custom = &map[string]any{"team": "blue", "plan": "pro"}
	return request
}

// madeAsBefore checks that a channel of a withCustom session was made with the fields it was
// before channels had a team.
func (s *AgentTeamsSuite) madeAsBefore(data map[string]any) {
	s.Equal(channelFieldsBefore, slices.Sorted(maps.Keys(data)))
	custom, _ := data["custom"].(map[string]any)
	s.Equal(channelCustomBefore, slices.Sorted(maps.Keys(custom)))
}

// turnTeamsOff turns the setting off for the suite's app, as Stream would, and waits until the
// router's settings cache no longer holds it on.
func (s *AgentTeamsSuite) turnTeamsOff() {
	_, err := s.store.DB().NewRaw(`UPDATE apps SET settings = '{}'::jsonb WHERE id = ?`, s.customerID()).Exec(s.T().Context())
	s.Require().NoError(err)
	address := os.Getenv("ROUTER_REDIS_ADDR")
	client, err := redisready.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	defer client.Close()
	s.Require().NoError(client.Do(s.T().Context(), client.B().Del().Key("appcfg:app-settings:"+s.customerID()).Build()).Error())
	s.Require().Eventually(func() bool {
		settings, err := s.configs.AppSettings(s.T().Context(), s.customerID())
		return err == nil && !settings.AgentTeamsChecked()
	}, 10*time.Second, 20*time.Millisecond)
}
