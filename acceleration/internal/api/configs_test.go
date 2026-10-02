//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type ConfigsSuite struct {
	RouterSuite
}

func TestConfigsSuite(t *testing.T) {
	runSuite(t, new(ConfigsSuite))
}

// SetupSuite registers the scheme the built-ins name, so the app can define a custom
// connector of its own, and seeds the built-ins as a router start does. Seeding is
// idempotent, so suites running beside this one see the same rows.
func (s *ConfigsSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{"oauth2_code": namedScheme("oauth2_code")}}
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

// SetupTest gives every test an app of its own, because a config's name has to be free and
// a list of skills is everything an app has.
func (s *ConfigsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConfigsSuite) TestAnAgentConfigSurvivesBeingStoredAndReadBack() {
	created := s.createConfig(map[string]any{
		"name": "support", "llm": "llm-flow", "tts": "en-low-latency", "voice": "aurora",
		"subagent": "llm-flow", "instructions": "be brief", "skills": []string{"think", "refund"},
		"keyterms":            []string{"Vision Agents", "Stream"},
		"knowledge_namespace": "handbook", "sandbox": "daytona",
		"tags": map[string]string{"project": "support"},
	})
	s.Require().NotEmpty(created.Id)
	s.Equal("support", created.Name)

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal("llm-flow", value(read.Llm))
	s.Equal("aurora", value(read.Voice))
	s.Equal([]string{"think", "refund"}, value(read.Skills))
	s.Equal([]string{"Vision Agents", "Stream"}, value(read.Keyterms))
	s.Equal("handbook", value(read.KnowledgeNamespace))
	s.Equal(Daytona, value(read.Sandbox))
	s.Equal("support", value(read.Tags)["project"])
}

func (s *ConfigsSuite) TestAConfigRemembersHowFastItsVoiceSpeaks() {
	created := s.createConfig(map[string]any{"name": "support", "voice": "aurora", "speed": 0.9})

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal(0.9, value(read.Speed))
}

func (s *ConfigsSuite) TestAConfigWithANegativeSpeedIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "speed": -1})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "speed")
}

func (s *ConfigsSuite) TestPatchingASpeedKeepsTheVoice() {
	created := s.createConfig(map[string]any{"name": "support", "voice": "aurora"})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"speed": 0.9}, &patched))

	s.Equal(0.9, value(patched.Speed))
	s.Equal("aurora", value(patched.Voice))
}

func (s *ConfigsSuite) TestAConfigRemembersWhichSearchItRoutesTo() {
	created := s.createConfig(map[string]any{"name": "support", "search": "en-low-latency"})

	s.Equal("en-low-latency", value(created.Search))
}

func (s *ConfigsSuite) TestAConfigRunsTheDefaultHarnessUnlessItSaysOtherwise() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.Equal(Default, value(unnamed.Harness), "a config that names none runs the default")

	named := s.createConfig(map[string]any{"name": "sales", "harness": "default"})
	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+named.Id, nil, &read))
	s.Equal(Default, value(read.Harness))
}

func (s *ConfigsSuite) TestAConfigLeavesNothingToDispatchUnlessItSaysSo() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.Equal(Disabled, value(value(unnamed.Dispatch).Text))
	s.Equal(Disabled, value(value(unnamed.Dispatch).IncomingCall))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+unnamed.Id,
		map[string]any{"dispatch": map[string]any{"text": "enabled"}}, &patched))
	s.Equal(Enabled, value(value(patched.Dispatch).Text))
	s.Equal(Disabled, value(value(patched.Dispatch).IncomingCall), "a setting left out keeps what is stored")
}

func (s *ConfigsSuite) TestADispatchSettingThatIsNeitherOnNorOffIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "dispatch": map[string]any{"text": "sometimes"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "dispatch.text")
}

func (s *ConfigsSuite) TestAConfigNamingAHarnessThatDoesNotExistIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "harness": "fancy"})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "fancy")

	created := s.createConfig(map[string]any{"name": "support"})
	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"harness": "fancy"})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "harness")
}

func (s *ConfigsSuite) TestAConfigNamingASandboxNobodyRunsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "sandbox": "docker"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "docker")
}

func (s *ConfigsSuite) TestAConfigNamingMoreKeytermsThanAnyProviderTakesIsRefused() {
	terms := make([]string, stt.MaxKeyterms+1)
	for i := range terms {
		terms[i] = fmt.Sprintf("term-%d", i)
	}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "keyterms": terms})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "keyterms")
}

func (s *ConfigsSuite) TestAConfigRemembersWhichToolsEndUsersSee() {
	created := s.createConfig(map[string]any{"name": "support", "visible_tools": []string{"athena_*", "search"}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"visible_tools": []string{"web_search"}}, &patched))

	s.Equal([]string{"athena_*", "search"}, value(created.VisibleTools))
	s.Equal([]string{"web_search"}, value(patched.VisibleTools))
}

func (s *ConfigsSuite) TestAVisibleToolPatternThatCannotMatchIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "visible_tools": []string{"athena_[*"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "visible_tools")
}

func (s *ConfigsSuite) TestUpdatingAConfigReplacesWhatItWas() {
	created := s.createConfig(map[string]any{
		"name": "support", "llm": "llm-flow", "instructions": "be brief",
	})

	var updated AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "llm": "en-low-latency"}, &updated))

	s.Equal(created.Id, updated.Id, "an update keeps the id callers already hold")
	s.Equal("en-low-latency", value(updated.Llm))
	s.Nil(updated.Instructions, "a field left out of a replacement is gone from it")
}

func (s *ConfigsSuite) TestPatchingAConfigKeepsWhatWasNotSent() {
	created := s.createConfig(map[string]any{
		"name": "support", "llm": "llm-flow", "instructions": "be brief",
		"skills": []string{"think"},
	})
	policy := "---\ntype: lcm\n---\nOnly answer questions about Acme."

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"guardrail": policy}, &patched))

	s.Equal(policy, value(patched.Guardrail))
	s.Equal("llm-flow", value(patched.Llm))
	s.Equal("be brief", value(patched.Instructions))
	s.Equal([]string{"think"}, value(patched.Skills))
}

func (s *ConfigsSuite) TestAPatchWithAGuardrailThatDoesNotParseChangesNothing() {
	created := s.createConfig(map[string]any{"name": "support", "llm": "llm-flow"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"llm": "en-low-latency", "guardrail": "---\ntype: regex\n---\nNo."})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "regex")
	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal("llm-flow", value(read.Llm))
}

func (s *ConfigsSuite) TestAPatchNamingAFieldConfigsDoNotHaveIsRefused() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"guardrails": "Only answer questions about Acme."}, nil))
}

func (s *ConfigsSuite) TestAnotherAppsConfigIsNotTheirsToPatch() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
			map[string]any{"instructions": "mine now"}, nil)
	})
}

func (s *ConfigsSuite) TestOnlyTheAppsOwnBackendMayPatchAConfig() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
			map[string]any{"instructions": "be brief"}, nil)
	})
}

func (s *ConfigsSuite) TestADeletedConfigCannotBeUsedAgain() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+created.Id, nil, nil))
	s.Equal(http.StatusNotFound,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, nil))

	// The name is free again, which is what makes deleting one usable rather than final.
	s.createConfig(map[string]any{"name": "support"})
}

func (s *ConfigsSuite) TestAnotherAppsConfigIsNotFound() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, nil)
	})
}

func (s *ConfigsSuite) TestOnlyTheAppsOwnBackendMayStoreAConfig() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "config-" + s.utils.uuid()}, nil)
	})
}

func (s *ConfigsSuite) TestASkillIsStoredAndListed() {
	config := s.createConfig(map[string]any{"name": "support"})

	var created Skill
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/skills",
		map[string]any{
			"config_id": config.Id,
			"name":      "refund", "description": "work out what a caller is owed",
			"instructions": "Read the order and the policy, then say what to refund.",
			"deadline_ms":  20_000,
		}, &created))
	s.EqualValues(20_000, value(created.DeadlineMs))

	var listed []Skill
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/skills", nil, &listed))
	s.Require().Len(listed, 1)
	s.Equal("refund", listed[0].Name)
}

func (s *ConfigsSuite) TestASkillWithoutADescriptionIsRefused() {
	// The description is the whole of how the fast model decides when to hand work over,
	// so a skill without one would never be reached for.
	config := s.createConfig(map[string]any{"name": "support"})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/skills", map[string]any{
		"config_id": config.Id, "name": "refund", "description": "", "instructions": "work it out",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "description")
}

func (s *ConfigsSuite) TestASkillBelongingToNoConfigIsRefused() {
	// A skill is not shared: it is one config's, so an unnamed config is a skill nothing
	// would ever reach for.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/skills",
		map[string]any{"name": "refund", "description": "work it out", "instructions": "read the policy"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "config")
}

// createConfig stores a config the router must accept.
func (s *ConfigsSuite) createConfig(body map[string]any) AgentConfig {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/configs", body, &created))
	return created
}

func (s *ConfigsSuite) TestAConfigsBindingsAreReadBackExactlyAsTheyWereWritten() {
	bindings := []map[string]any{fixedSlack("crm", s.connection("")), sessionSlack("inbox")}
	created := s.createConfig(map[string]any{"name": "support", "connectors": bindings})

	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/configs/"+created.Id, nil)
	s.Require().Equal(http.StatusOK, status)
	var read struct {
		Connectors json.RawMessage `json:"connectors"`
	}
	s.Require().NoError(json.Unmarshal(raw, &read))
	written, err := json.Marshal(bindings)
	s.Require().NoError(err)
	s.JSONEq(string(written), string(read.Connectors))
}

func (s *ConfigsSuite) TestAConfigWithoutBindingsShowsNone() {
	created := s.createConfig(map[string]any{"name": "support"})

	s.Nil(created.Connectors)
}

func (s *ConfigsSuite) TestABindingToAConnectorThatDoesNotExistIsRefusedByName() {
	binding := sessionSlack("crm")
	binding["connector_id"] = "custom_nothing_here"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "custom_nothing_here")
}

func (s *ConfigsSuite) TestABindingToTheAppsOwnCustomConnectorIsStored() {
	id := s.customConnector(s.serverClient)
	binding := sessionSlack("crm")
	binding["connector_id"] = id

	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Require().Len(value(created.Connectors), 1)
	s.Equal(id, value(created.Connectors)[0].ConnectorId)
}

func (s *ConfigsSuite) TestABindingToAnotherAppsCustomConnectorIsRefused() {
	id := s.customConnector(s.data.backendOfAnotherApp())
	binding := sessionSlack("crm")
	binding["connector_id"] = id

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, id)
}

func (s *ConfigsSuite) TestAFixedBindingToAUsersConnectionIsRefused() {
	connection := s.connection(s.utils.uuid())

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{fixedSlack("crm", connection)}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"crm"`)
	s.Contains(failure, "app's own")
}

func (s *ConfigsSuite) TestAFixedBindingToAnotherAppsConnectionIsRefused() {
	mine := s.app
	s.useApp(s.data.createApp())
	theirs := s.connection("")
	s.useApp(mine)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{fixedSlack("crm", theirs)}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, theirs)
}

func (s *ConfigsSuite) TestAFixedBindingToADeletedConnectionIsRefused() {
	connection := s.connection("")
	s.Require().NoError(s.store.DeleteConnectorConnection(context.Background(), s.customerID(), connection))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{fixedSlack("crm", connection)}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, connection)
}

func (s *ConfigsSuite) TestAFixedBindingNeedsAConnection() {
	binding := fixedSlack("crm", "")
	binding["connection"] = map[string]any{"type": "fixed"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "connection_id")
}

func (s *ConfigsSuite) TestASessionBindingCannotNameItsConnection() {
	binding := sessionSlack("inbox")
	binding["connection"] = map[string]any{"type": "session", "connection_id": s.connection(s.utils.uuid())}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"inbox"`)
}

func (s *ConfigsSuite) TestABindingChosenNeitherFixedNorPerSessionIsRefused() {
	binding := sessionSlack("inbox")
	binding["connection"] = map[string]any{"type": "shared"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "shared")
}

func (s *ConfigsSuite) TestAnAliasHoldingTheToolSeparatorIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("team__inbox")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "__")
}

func (s *ConfigsSuite) TestAnAliasThatIsNotLowercaseIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("Inbox")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "Inbox")
}

func (s *ConfigsSuite) TestAnAliasOfSixtyThreeCharactersIsTheLongest() {
	longest := "a" + strings.Repeat("b", 62)

	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack(longest)}})
	s.Equal(longest, value(created.Connectors)[0].Name)

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "other", "connectors": []map[string]any{sessionSlack(longest + "c")}})
	s.Equal(http.StatusBadRequest, status)
}

func (s *ConfigsSuite) TestTwoBindingsWithOneAliasAreRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "support", "connectors": []map[string]any{sessionSlack("inbox"), sessionSlack("inbox")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"inbox"`)
}

func (s *ConfigsSuite) TestADigestThatIsNotASHA256IsRefused() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{{"name": "search", "schema_digest": toolDigest[:63]}}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "schema_digest")
}

func (s *ConfigsSuite) TestAToolGrantedTwiceIsRefused() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{
		{"name": "search", "schema_digest": toolDigest}, {"name": "search", "schema_digest": toolDigest},
	}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "twice")
}

func (s *ConfigsSuite) TestThirtySecondsIsTheLongestATimeoutMayBe() {
	binding := sessionSlack("inbox")
	binding["timeout_ms"] = 30000
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})
	s.Equal(30000, value(value(created.Connectors)[0].TimeoutMs))

	binding["timeout_ms"] = 30001
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "other", "connectors": []map[string]any{binding}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "timeout_ms")
}

func (s *ConfigsSuite) TestATimeoutOfNothingIsRefused() {
	binding := sessionSlack("inbox")
	binding["timeout_ms"] = 0

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "timeout_ms")
}

func (s *ConfigsSuite) TestUpdatingAConfigWithoutItsBindingsKeepsThem() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	var updated AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "instructions": "be brief"}, &updated))

	s.Equal(created.Connectors, updated.Connectors)
	s.Equal(created.Connectors, s.read(created.Id).Connectors)
}

func (s *ConfigsSuite) TestUpdatingAConfigWithNoBindingsClearsThem() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	var updated AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "connectors": []map[string]any{}}, &updated))

	s.Nil(updated.Connectors)
	s.Nil(s.read(created.Id).Connectors)
}

func (s *ConfigsSuite) TestPatchingAConfigWithoutItsBindingsKeepsThem() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"instructions": "be brief"}, &patched))

	s.Equal(created.Connectors, patched.Connectors)
	s.Equal(created.Connectors, s.read(created.Id).Connectors)
}

func (s *ConfigsSuite) TestPatchingAConfigWithNoBindingsClearsThem() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{}}, nil))

	s.Nil(s.read(created.Id).Connectors)
}

// A null is read as left out, as it is for every field of a patch: Huma skips a null
// optional property before validating it, and Go decodes it to the nil a missing one is.
func (s *ConfigsSuite) TestPatchingBindingsToNullKeepsThem() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": nil}, nil))

	s.Equal(created.Connectors, s.read(created.Id).Connectors)
}

func (s *ConfigsSuite) TestUpdatingBindingsToNullKeepsThem() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "connectors": nil}, nil))

	s.Equal(created.Connectors, s.read(created.Id).Connectors)
}

func (s *ConfigsSuite) TestPatchingBindingsReplacesTheOnesStored() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})
	connection := s.connection("")

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{fixedSlack("crm", connection)}}, nil))

	bindings := value(s.read(created.Id).Connectors)
	s.Require().Len(bindings, 1)
	s.Equal("crm", bindings[0].Name)
	s.Equal(connection, value(bindings[0].Connection.ConnectionId))
}

func (s *ConfigsSuite) TestPatchingAFixedBindingToAUsersConnectionIsRefused() {
	created := s.createConfig(map[string]any{"name": "support"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{fixedSlack("crm", s.connection(s.utils.uuid()))}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "app's own")
	s.Nil(s.read(created.Id).Connectors)
}

// toolDigest is a schema_digest of the right shape. Nothing in these suites discovers the
// tools it would pin.
var toolDigest = strings.Repeat("ab", 32)

// fixedSlack binds Slack through a connection named in the config.
func fixedSlack(alias, connectionID string) map[string]any {
	return map[string]any{
		"name": alias, "connector_id": "slack",
		"connection": map[string]any{"type": "fixed", "connection_id": connectionID},
		"tools":      []map[string]any{{"name": "search", "schema_digest": toolDigest}},
		"required":   true, "timeout_ms": 5000,
	}
}

// sessionSlack binds Slack through the connection a session's end user picks.
func sessionSlack(alias string) map[string]any {
	return map[string]any{
		"name": alias, "connector_id": "slack",
		"connection": map[string]any{"type": "session"},
		"tools":      []map[string]any{{"name": "search", "schema_digest": toolDigest}},
		"required":   false,
	}
}

// connection stores a Slack connection of the suite's app, the app's own when owner is
// empty and that user's otherwise, and returns its id. It is written through the store,
// since no endpoint makes one yet.
func (s *ConfigsSuite) connection(owner string) string {
	ctx := context.Background()
	slack, err := s.store.LatestConnectorDefinition(ctx, s.customerID(), "slack")
	s.Require().NoError(err)
	connection := &store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: slack.ID, DefinitionRevision: slack.Revision,
		OwnerType: store.OwnerApp, AuthScheme: "oauth2_code",
	}
	if owner != "" {
		connection.OwnerType, connection.OwnerID = store.OwnerUser, owner
	}
	s.Require().NoError(s.store.CreateConnectorConnection(ctx, s.connectors, connection))
	return connection.ID
}

// customConnector defines a custom MCP connector as the app behind backend, and returns
// its id.
func (s *ConfigsSuite) customConnector(backend *testClient) string {
	id := "custom_t" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	s.Require().Equal(http.StatusOK, backend.do(http.MethodPost, "/v1/agents/connectors", map[string]any{
		"id": id, "name": "Our CRM", "endpoint": "https://8.8.8.8/mcp", "schemes": []string{"oauth2_code"},
		"client": map[string]any{"policy": []string{"dcr"}},
	}, nil))
	return id
}

// read is a config as GET returns it.
func (s *ConfigsSuite) read(id string) AgentConfig {
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &read))
	return read
}
