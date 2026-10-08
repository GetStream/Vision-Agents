//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"testing"
)

type AuditSuite struct {
	RouterSuite
}

func TestAuditSuite(t *testing.T) {
	runSuite(t, new(AuditSuite))
}

// SetupTest gives every test an app of its own: the log is the app's whole history, so a
// test reading it has to be the only one writing it.
func (s *AuditSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *AuditSuite) TestChangingAnAgentRecordsTheFieldThatMovedAndWhoMovedIt() {
	config := s.createConfig(map[string]any{"name": "support", "llm": "llm-flow"})

	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch,
		"/v1/agents/configs/"+config.Id, map[string]any{"llm": "llm-fast"}, nil))

	entry := s.latest(AuditFilter{Action: action(AuditAction("updated"))})
	s.Equal(AuditResourceType("agent_config"), entry.ResourceType)
	s.Equal(config.Id, entry.ResourceID)
	s.Equal("support", entry.ResourceName)
	s.Equal(config.Id, entry.AgentID)
	s.Equal(AuditSource("dashboard"), entry.Source)
	s.Equal("volt-7", entry.ActorID)
	s.Equal("Ada Lovelace", entry.ActorName)
	s.Require().Len(entry.Changes, 1)
	s.Equal("llm", entry.Changes[0].Field)
	s.Equal("llm-flow", entry.Changes[0].Before)
	s.Equal("llm-fast", entry.Changes[0].After)
}

func (s *AuditSuite) TestASaveThatChangedNothingIsNotAChange() {
	config := s.createConfig(map[string]any{"name": "support", "llm": "llm-flow"})
	before := s.entries(AuditFilter{AgentID: &config.Id})

	s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch,
		"/v1/agents/configs/"+config.Id, map[string]any{"llm": "llm-flow"}, nil))

	s.Len(s.entries(AuditFilter{AgentID: &config.Id}), len(before),
		"saving a setting onto the value it already held is not somebody changing it")
}

func (s *AuditSuite) TestCreatingAnAgentRecordsWhatItWasCreatedWith() {
	config := s.createConfig(map[string]any{"name": "newcomer", "llm": "llm-flow"})

	entry := s.latest(AuditFilter{AgentID: &config.Id})
	s.Equal(AuditAction("created"), entry.Action)
	s.Equal("llm-flow", s.changed(entry, "llm").After)
	s.Nil(s.changed(entry, "llm").Before, "nothing held the value before it was created")
	s.Empty(s.fields(entry, "id"), "the router's own bookkeeping is not somebody's change")
	s.Empty(s.fields(entry, "updated_at"))
}

func (s *AuditSuite) TestDeletingAnAgentLeavesItsDeletionOnRecord() {
	config := s.createConfig(map[string]any{"name": "goner", "llm": "llm-flow"})

	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config.Id, nil, nil))

	entry := s.latest(AuditFilter{AgentID: &config.Id})
	s.Equal(AuditAction("deleted"), entry.Action)
	s.Equal("goner", entry.ResourceName)
	s.Equal("llm-flow", s.changed(entry, "llm").Before)
	s.Nil(s.changed(entry, "llm").After)
}

func (s *AuditSuite) TestASkillIsRecordedAgainstTheAgentItBelongsTo() {
	config := s.createConfig(map[string]any{"name": "support"})

	var skill Skill
	s.Require().Equal(http.StatusCreated, s.dashboard().do(http.MethodPost, "/v1/agents/skills",
		map[string]any{"config_id": config.Id, "name": "refund",
			"description": "work out a refund", "instructions": "Read the policy."}, &skill))

	entry := s.latest(AuditFilter{ResourceType: resource("skill")})
	s.Equal(skill.Id, entry.ResourceID)
	s.Equal("refund", entry.ResourceName)
	s.Equal(config.Id, entry.AgentID, "a skill's history is its agent's history")
	s.Equal("Read the policy.", s.changed(entry, "instructions").After)
}

func (s *AuditSuite) TestARouterConfigIsRecordedUnderNoAgent() {
	var created RouterConfig
	s.Require().Equal(http.StatusCreated, s.dashboard().do(http.MethodPost,
		"/v1/router/configs", map[string]any{"name": "fast"}, &created))

	entry := s.latest(AuditFilter{ResourceType: resource("router_config")})
	s.Equal(created.Id, entry.ResourceID)
	s.Equal("fast", entry.ResourceName)
	s.Empty(entry.AgentID, "a router belongs to no agent")
}

func (s *AuditSuite) TestACallerThatNamesNoClientIsRecordedAsTheApi() {
	config := s.createConfig(map[string]any{"name": "unsigned"})

	entry := s.latest(AuditFilter{AgentID: &config.Id})
	s.Equal(AuditSource("api"), entry.Source)
	s.Empty(entry.ActorID)
	s.Empty(entry.ActorName)
}

func (s *AuditSuite) TestTheLogIsNarrowedToOneAgent() {
	kept := s.createConfig(map[string]any{"name": "kept"})
	other := s.createConfig(map[string]any{"name": "other"})

	listed := s.entries(AuditFilter{AgentID: &kept.Id})

	s.Require().NotEmpty(listed)
	for _, entry := range listed {
		s.Equal(kept.Id, entry.AgentID)
		s.NotEqual(other.Id, entry.ResourceID)
	}
}

func (s *AuditSuite) TestTheLogIsNarrowedToOneClient() {
	s.createConfig(map[string]any{"name": "by-the-api"})
	var byTheCLI AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.from("cli", "tom", "tom@example.com").
		do(http.MethodPost, "/v1/agents/configs", map[string]any{"name": "by-the-cli"}, &byTheCLI))

	listed := s.entries(AuditFilter{Source: source("cli")})

	s.Require().Len(listed, 1)
	s.Equal(byTheCLI.Id, listed[0].ResourceID)
	s.Equal("tom@example.com", listed[0].ActorName)
}

func (s *AuditSuite) TestTheLogPagesToTheEndWithoutRepeatingOrSkippingAChange() {
	config := s.createConfig(map[string]any{"name": "busy", "llm": "llm-0"})
	for i := range 5 {
		s.Require().Equal(http.StatusOK, s.dashboard().do(http.MethodPatch,
			"/v1/agents/configs/"+config.Id,
			map[string]any{"greeting": "hello " + string(rune('a'+i))}, nil))
	}

	seen := map[string]bool{}
	cursor := ""
	for pages := 0; ; pages++ {
		s.Require().Less(pages, 10, "the cursor is not moving")
		query := map[string]any{"filter": map[string]any{"agent_id": config.Id}, "limit": 2}
		if cursor != "" {
			query["cursor"] = cursor
		}
		var answered AuditPage
		s.Require().Equal(http.StatusOK,
			s.serverClient.do(http.MethodPost, "/v1/audit/query", query, &answered))
		for _, entry := range answered.Items {
			s.False(seen[entry.ID], "a change was listed twice")
			seen[entry.ID] = true
		}
		if !answered.HasMore {
			s.Nil(answered.NextCursor)
			break
		}
		s.Require().NotNil(answered.NextCursor)
		cursor = *answered.NextCursor
	}
	s.Len(seen, 6, "the create and the five edits")
}

func (s *AuditSuite) TestACursorThisListNeverHandedOutIsRefused() {
	status, message := s.serverClient.failure(http.MethodPost, "/v1/audit/query",
		map[string]any{"cursor": "not-a-cursor"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(message, "cursor")
}

func (s *AuditSuite) TestAnotherAppsChangesAreNotListed() {
	s.createConfig(map[string]any{"name": "ours"})

	var answered AuditPage
	s.Require().Equal(http.StatusOK, s.data.backendOfAnotherApp().
		do(http.MethodPost, "/v1/audit/query", map[string]any{}, &answered))

	s.Empty(answered.Items, "the log is the app's own history and nobody else's")
}

func (s *AuditSuite) TestOnlyTheAppsOwnBackendMayReadTheLog() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/audit/query", map[string]any{}, nil)
	})
}

// dashboard is the backend saying it is the dashboard, with the operator who clicked save.
func (s *AuditSuite) dashboard() *testClient {
	return s.serverClient.from("dashboard", "volt-7", "Ada Lovelace")
}

func (s *AuditSuite) createConfig(body map[string]any) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/configs", body, &created))
	return created
}

// entries are the app's changes the filter matches, newest first.
func (s *AuditSuite) entries(filter AuditFilter) []AuditEntry {
	var answered AuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/audit/query",
		map[string]any{"filter": filter, "limit": 200}, &answered))
	return answered.Items
}

// latest is the newest change the filter matches, and fails the test when there is none.
func (s *AuditSuite) latest(filter AuditFilter) AuditEntry {
	listed := s.entries(filter)
	s.Require().NotEmpty(listed, "no change was recorded")
	return listed[0]
}

// changed is the entry's change to one field, and fails the test when it did not move.
func (s *AuditSuite) changed(entry AuditEntry, field string) AuditChange {
	for _, change := range entry.Changes {
		if change.Field == field {
			return change
		}
	}
	s.Require().Fail("no change to " + field, "%s", must(json.Marshal(entry.Changes)))
	return AuditChange{}
}

// fields are the entry's changes to a field, for a test asserting that it has none.
func (s *AuditSuite) fields(entry AuditEntry, field string) []AuditChange {
	var found []AuditChange
	for _, change := range entry.Changes {
		if change.Field == field {
			found = append(found, change)
		}
	}
	return found
}

func resource(name AuditResourceType) *AuditResourceType { return &name }
func action(name AuditAction) *AuditAction               { return &name }
func source(name AuditSource) *AuditSource               { return &name }

func must[T any](value T, err error) T {
	if err != nil {
		panic(err)
	}
	return value
}
