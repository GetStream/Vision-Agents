//go:build integration

package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type ConfigsSuite struct {
	RouterSuite
	// describing runs whenever the router asks a config's MCP server about itself, which a
	// save does between checking the config and writing it. Nil runs nothing.
	describing atomic.Pointer[func()]
	// logged is what the router logged, for the tests of the deprecation it warns of.
	logged *lockedLog
}

func TestConfigsSuite(t *testing.T) {
	runSuite(t, new(ConfigsSuite))
}

// SetupSuite registers the scheme the built-ins name, so the app can define a custom
// connector of its own, and seeds the built-ins as a router start does. Seeding is
// idempotent, so suites running beside this one see the same rows. An MCP server is
// reached by no one, as mcpTransport's default, after running describing.
func (s *ConfigsSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{"oauth2_code": namedScheme("oauth2_code")}}
	s.pluginHTTP = &http.Client{Transport: roundTripper(func(*http.Request) (*http.Response, error) {
		if describing := s.describing.Load(); describing != nil {
			(*describing)()
		}
		return nil, errors.New("tests reach no real MCP server")
	})}
	s.logged = &lockedLog{}
	s.logs = s.logged
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
	s.Equal("llm-flow", value(read.Subagent))
	s.Equal("aurora", value(read.Voice))
	s.Equal([]string{"think", "refund"}, value(read.Skills))
	s.Equal([]string{"Vision Agents", "Stream"}, value(read.Keyterms))
	s.Equal("handbook", value(read.KnowledgeNamespace))
	s.Equal(Daytona, value(read.Sandbox))
	s.Equal("support", value(read.Tags)["project"])
}

func (s *ConfigsSuite) TestAConfigRemembersHowItGreets() {
	created := s.createConfig(map[string]any{"name": "support",
		"greeting": map[string]any{"text": "Hello.", "mode": "variation"}})

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Require().NotNil(read.Greeting)
	s.Equal("Hello.", read.Greeting.Text)
	s.Equal(GreetingModeVariation, value(read.Greeting.Mode))
}

func (s *ConfigsSuite) TestAGreetingWithNoModeIsSaidExactly() {
	created := s.createConfig(map[string]any{"name": "support", "greeting": map[string]any{"text": "Hello."}})

	s.Require().NotNil(created.Greeting)
	s.Equal(GreetingModeExact, value(created.Greeting.Mode))
}

func (s *ConfigsSuite) TestAGreetingWithAnUnknownModeIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "greeting": map[string]any{"text": "Hello.", "mode": "loud"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "mode")
}

func (s *ConfigsSuite) TestATextAgentNamingASubagentIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "analyst", "mode": "text", "subagent": "llm-thinking"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "subagent")
}

func (s *ConfigsSuite) TestSwitchingAnAgentToTextDropsItsSubagent() {
	created := s.createConfig(map[string]any{"name": "support", "subagent": "llm-thinking"})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"mode": "text"}, &patched))
	s.Equal(AgentModeText, patched.Mode)
	s.Nil(patched.Subagent)
}

func (s *ConfigsSuite) TestPatchingAnEmptyGreetingRemovesIt() {
	created := s.createConfig(map[string]any{"name": "support", "greeting": map[string]any{"text": "Hello."}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"greeting": map[string]any{"text": ""}}, &patched))

	s.Nil(patched.Greeting)
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

// Episode cards are off unless a config turns them on, and an update that leaves the setting
// out keeps it, so a client that does not know it cannot turn the cards off by saving.
func (s *ConfigsSuite) TestAConfigWritesNoEpisodeCardsUnlessItSaysSo() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.False(value(unnamed.EpisodeCards))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"episode_cards": true}, &patched))
	s.True(value(patched.EpisodeCards))

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"name": "support"}, &saved))
	s.True(value(saved.EpisodeCards), "a setting left out on an update stays")

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+unnamed.Id, nil, &read))
	s.True(value(read.EpisodeCards))
}

// Tools are offered whole unless a config asks for them progressively, and an update that
// leaves the setting out keeps it.
func (s *ConfigsSuite) TestAConfigOffersToolsWholeUnlessItSaysProgressively() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.False(value(unnamed.Tools.Progressive))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"tools": map[string]any{"progressive": true}}, &patched))
	s.True(value(patched.Tools.Progressive))

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"name": "support", "tools": map[string]any{}}, &saved))
	s.True(value(saved.Tools.Progressive), "a setting left out on an update stays")
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

func (s *ConfigsSuite) TestAConfigRemembersHowItsSandboxIsBuilt() {
	created := s.createConfig(map[string]any{
		"name": "artist", "sandbox": "daytona",
		"sandbox_options": map[string]any{
			"image":      "python:3.13-slim-bookworm",
			"setup":      []string{"pip install bpy==5.2.2"},
			"timeout_ms": 300000, "cpu": 2, "memory_gb": 4,
		},
	})

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	options := value(read.SandboxOptions)
	s.Equal("python:3.13-slim-bookworm", value(options.Image))
	s.Equal([]string{"pip install bpy==5.2.2"}, value(options.Setup))
	s.Equal(300000, value(options.TimeoutMs))
	s.Equal(2, value(options.Cpu))
	s.Equal(4, value(options.MemoryGb))
}

func (s *ConfigsSuite) TestASandboxLeftAloneHasNoOptions() {
	created := s.createConfig(map[string]any{"name": "analyst", "sandbox": "daytona"})

	s.Nil(created.SandboxOptions, "the provider's own sandbox has nothing to say about how it is built")
}

func (s *ConfigsSuite) TestPatchingTheSandboxOptionsReplacesThem() {
	created := s.createConfig(map[string]any{"name": "artist", "sandbox": "daytona",
		"sandbox_options": map[string]any{"setup": []string{"pip install numpy"}, "timeout_ms": 60000}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"sandbox_options": map[string]any{"timeout_ms": 600000}}, &patched))

	s.Equal(600000, value(value(patched.SandboxOptions).TimeoutMs))
	s.Empty(value(value(patched.SandboxOptions).Setup))
	s.Equal(Daytona, value(patched.Sandbox), "the sandbox itself is untouched")
}

func (s *ConfigsSuite) TestARunLongerThanThirtyMinutesIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "artist", "sandbox": "daytona",
			"sandbox_options": map[string]any{"timeout_ms": 3600000}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "timeout_ms")
}

func (s *ConfigsSuite) TestAnEmptySetupCommandIsRefused() {
	created := s.createConfig(map[string]any{"name": "artist", "sandbox": "daytona"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"sandbox_options": map[string]any{"setup": []string{"  "}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "setup")
}

func (s *ConfigsSuite) TestAConfigRemembersTheMCPServersItNamesByURL() {
	created := s.createConfig(map[string]any{"name": "concierge", "mcp_servers": []map[string]any{
		{"name": "tablejourney", "url": "https://tablejourney.com/mcp"},
	}})

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal([]McpServer{{Name: "tablejourney", Url: "https://tablejourney.com/mcp"}}, value(read.McpServers))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"mcp_servers": []map[string]any{}}, &patched))
	s.Nil(patched.McpServers)
}

func (s *ConfigsSuite) TestAConfigRemembersHowItReachesAPlugin() {
	created := s.createConfig(map[string]any{
		"name": "triage",
		"plugins": []any{
			"sentry",
			map[string]any{"name": "linear", "user": true, "readonly": true, "scopes": []string{"read", " "}},
			map[string]any{"name": "google_calendar", "user": true},
		},
	})

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal([]PluginEntry{
		{Name: "sentry"},
		{Name: "linear", Readonly: pointerTo(true), Scopes: &[]string{"read"}},
		{Name: "google_calendar"},
	}, value(read.Plugins), "user is answered only where it differs from the catalog")

	var raw map[string]any
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &raw))
	s.Equal("sentry", raw["plugins"].([]any)[0], "an entry with no options answers as its id")

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"plugins": []any{"linear"}}, &patched))
	s.Equal([]PluginEntry{{Name: "linear"}}, value(patched.Plugins))
}

func (s *ConfigsSuite) TestAPluginLeftToTheCatalogIsConnectedTheWayTheCatalogSays() {
	created := s.createConfig(map[string]any{
		"name":    "triage",
		"plugins": []any{"sentry", "google_calendar", map[string]any{"name": "linear", "user": false}},
	})

	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.Equal([]store.PluginEntry{{Name: "sentry"}, {Name: "google_calendar", User: true}, {Name: "linear"}},
		stored.Plugins, "a company's account is the app's, a person's own each user's, unless the entry says")
	s.Equal([]PluginEntry{{Name: "sentry"}, {Name: "google_calendar"}, {Name: "linear", User: pointerTo(false)}},
		value(created.Plugins))
}

func (s *ConfigsSuite) TestAReadonlyPluginWithNoReadOnlyEndpointIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":    "triage",
		"plugins": []any{map[string]any{"name": "sentry", "readonly": true}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "read-only")
}

func (s *ConfigsSuite) TestAConfigRemembersWhichToolsetsAPluginIsLimitedTo() {
	created := s.createConfig(map[string]any{
		"name":    "scheduler",
		"plugins": []any{map[string]any{"name": "calcom", "user": true, "toolsets": []string{"bookings", "availability"}}},
	})

	s.Equal([]PluginEntry{{Name: "calcom", Toolsets: &[]string{"bookings", "availability"}}},
		value(created.Plugins))
}

func (s *ConfigsSuite) TestAToolsetThePluginDoesNotHaveIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":    "scheduler",
		"plugins": []any{map[string]any{"name": "calcom", "user": true, "toolsets": []string{"invoices"}}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "invoices")
}

func (s *ConfigsSuite) TestAConfigRemembersWhichToolsEachServerOffers() {
	created := s.createConfig(map[string]any{
		"name":    "researcher",
		"plugins": []any{map[string]any{"name": "google_drive", "user": true, "tools": []string{"search_files", "read_*"}}},
		"mcp_servers": []map[string]any{
			{"name": "tablejourney", "url": "https://tablejourney.com/mcp", "tools": []string{"search_restaurants"}},
		},
	})

	s.Equal([]PluginEntry{{Name: "google_drive", Tools: &[]string{"search_files", "read_*"}}},
		value(created.Plugins))
	s.Equal([]McpServer{{Name: "tablejourney", Url: "https://tablejourney.com/mcp", Tools: &[]string{"search_restaurants"}}},
		value(created.McpServers))
}

func (s *ConfigsSuite) TestAToolPatternThatCannotBeReadIsRefused() {
	for _, body := range []map[string]any{
		{"name": "researcher", "plugins": []any{map[string]any{"name": "google_drive", "user": true, "tools": []string{"read_[*"}}}},
		{"name": "researcher", "mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp", "tools": []string{"read_[*"}}}},
	} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", body)

		s.Equal(http.StatusBadRequest, status)
		s.Contains(failure, "read_[*")
	}
}

func (s *ConfigsSuite) TestAScopeThePluginsServerDoesNotAcceptIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":    "researcher",
		"plugins": []any{map[string]any{"name": "google_drive", "scopes": []string{"https://www.googleapis.com/auth/gmail.readonly"}}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "gmail.readonly")
}

func (s *ConfigsSuite) TestAPluginNotInTheCatalogIsRefused() {
	for _, entry := range []any{"jira", map[string]any{"name": "jira", "readonly": true}} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
			"name":    "triage",
			"plugins": []any{entry},
		})

		s.Equal(http.StatusBadRequest, status)
		s.Contains(failure, "jira")
	}
}

func (s *ConfigsSuite) TestAPluginNamedTwiceInAListIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":    "triage",
		"plugins": []any{"linear", map[string]any{"name": "linear", "readonly": true}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "linear is named twice")
}

func (s *ConfigsSuite) TestAPluginEntryThatIsNeitherAnIdNorAnObjectIsRefused() {
	for _, entry := range []any{"", 7, map[string]any{"readonly": true}} {
		status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
			"name":    "triage",
			"plugins": []any{entry},
		})

		s.Equal(http.StatusBadRequest, status, "%v", entry)
	}
}

func (s *ConfigsSuite) TestAnMCPServerThatCannotBeNamedOrReachedSafelyIsRefused() {
	for _, refused := range []struct {
		server  map[string]any
		failure string
	}{
		{map[string]any{"name": "tablejourney", "url": "http://tablejourney.com/mcp"}, "https"},
		{map[string]any{"name": "slack", "url": "https://mcp.slack.example/mcp"}, "catalog"},
		{map[string]any{"name": "table__journey", "url": "https://tablejourney.com/mcp"}, "__"},
		{map[string]any{"name": "TableJourney", "url": "https://tablejourney.com/mcp"}, "name"},
	} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "concierge", "mcp_servers": []map[string]any{refused.server}})

		s.Equal(http.StatusBadRequest, status, refused.server)
		s.Contains(failure, refused.failure)
	}
}

func (s *ConfigsSuite) TestTwoMCPServersWithOneNameAreRefused() {
	server := map[string]any{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "concierge", "mcp_servers": []map[string]any{server, server}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "twice")
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
	policy := "---\ntype: decision_model\n---\nOnly answer questions about Acme."

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

// readConnectors is a config's bindings as GET answers them, and as its row stores them.
func (s *ConfigsSuite) readConnectors(id string) (json.RawMessage, string) {
	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/configs/"+id, nil)
	s.Require().Equal(http.StatusOK, status)
	var read struct {
		Connectors json.RawMessage `json:"connectors"`
	}
	s.Require().NoError(json.Unmarshal(raw, &read))
	var stored string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT connectors::text FROM agent_configs WHERE id = ?", id).Scan(&stored))
	return read.Connectors, stored
}

func (s *ConfigsSuite) TestABindingsPolicyIsReadBackExactlyAsItWasWritten() {
	speaking := sessionSlack("inbox")
	speaking["policy"] = map[string]any{"pre_speech": "Let me look in Slack.", "on_interrupt": "wait", "cancellable": false}
	waiting := fixedSlack("crm", s.connection(""))
	waiting["policy"] = map[string]any{"on_interrupt": "cancel"}
	bindings := []map[string]any{speaking, waiting}

	created := s.createConfig(map[string]any{"name": "support", "connectors": bindings})

	read, _ := s.readConnectors(created.Id)
	written, err := json.Marshal(bindings)
	s.Require().NoError(err)
	s.JSONEq(string(written), string(read))
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	cancellable := false
	s.Equal(&store.BindingPolicy{PreSpeech: "Let me look in Slack.", OnInterrupt: store.InterruptWait, Cancellable: &cancellable},
		stored.Connectors[0].Policy)
	s.Equal(&store.BindingPolicy{OnInterrupt: store.InterruptCancel}, stored.Connectors[1].Policy)
}

func (s *ConfigsSuite) TestAPatchSetsABindingsPolicy() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})
	patched := sessionSlack("inbox")
	patched["policy"] = map[string]any{"on_interrupt": "wait"}

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{patched}}, nil))

	read, _ := s.readConnectors(created.Id)
	written, err := json.Marshal([]map[string]any{patched})
	s.Require().NoError(err)
	s.JSONEq(string(written), string(read))
}

func (s *ConfigsSuite) TestAPolicyTheSchemaCannotHoldIsRefused() {
	for name, policy := range map[string]map[string]any{
		"an on_interrupt that is neither cancel nor wait": {"on_interrupt": "ignore"},
		"an empty pre_speech":                             {"pre_speech": ""},
		"cancellable that is not a boolean":               {"cancellable": "no"},
	} {
		s.Run(name, func() {
			binding := sessionSlack("inbox")
			binding["policy"] = policy

			status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
				map[string]any{"name": "support", "connectors": []map[string]any{binding}})

			s.Equal(http.StatusBadRequest, status)
			s.Contains(failure, "policy")
		})
	}
}

func (s *ConfigsSuite) TestABindingWithoutAPolicyIsStoredAndReadAsBefore() {
	// No policy key on the wire or in the row: what a binding was before policies existed.
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	read, stored := s.readConnectors(created.Id)

	s.NotContains(string(read), "policy")
	s.NotContains(stored, "policy")
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

// The save checks the connection, then asks its MCP servers about themselves, then writes.
// A delete in between is seen by the write, which runs in a transaction of its own (AI-889).
func (s *ConfigsSuite) TestAFixedBindingToAConnectionDeletedDuringTheSaveIsRefused() {
	connection := s.connection("")
	var once sync.Once
	deleted := make(chan error, 1)
	describing := func() {
		once.Do(func() {
			deleted <- s.store.DeleteUnboundConnectorConnection(context.Background(), s.customerID(), connection)
		})
	}
	s.describing.Store(&describing)
	defer s.describing.Store(nil)

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":        "support",
		"mcp_servers": []map[string]any{{"name": "crm", "url": "https://crm.example.com/mcp"}},
		"connectors":  []map[string]any{fixedSlack("inbox", connection)},
	})

	s.Require().Len(deleted, 1, "the connection was deleted while the save described its MCP server")
	s.Require().NoError(<-deleted)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, connection)
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs", nil, &listed))
	s.Empty(listed, "nothing is stored")
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

// TestASessionBindingMayGrantAToolByNameAlone: its connection is each person's own, so the
// digest is pinned per connection on first use, and the config is read back without one.
func (s *ConfigsSuite) TestASessionBindingMayGrantAToolByNameAlone() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{{"name": "search"}, {"name": "fetch", "schema_digest": toolDigest}}

	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal([]ConnectorToolGrant{{Name: "search"}, {Name: "fetch", SchemaDigest: toolDigest}},
		value(s.read(created.Id).Connectors)[0].Tools)
}

// TestAFixedBindingCannotGrantAToolByNameAlone: its connection is the app's own, whose tools
// the developer lists and grants at their digests.
func (s *ConfigsSuite) TestAFixedBindingCannotGrantAToolByNameAlone() {
	binding := fixedSlack("inbox", s.connection(""))
	binding["tools"] = []map[string]any{{"name": "search"}}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `grants "search" with no schema_digest`)
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
		"client": map[string]any{"registration": []string{"dcr"}},
	}, nil))
	return id
}

// read is a config as GET returns it.
func (s *ConfigsSuite) read(id string) AgentConfig {
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &read))
	return read
}

func (s *ConfigsSuite) TestAnAliasEndingInAnUnderscoreIsRefused() {
	// a_ and search would be offered as a___search, which splits back as a and _search.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("a_")}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "connectors[0].name: a_")
}

func (s *ConfigsSuite) TestAFixedBindingThroughAnotherConnectorsConnectionIsRefused() {
	slack := s.connection("")
	binding := fixedSlack("crm", slack)
	binding["connector_id"] = "linear"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "linear")
	s.Contains(failure, slack)
}

// A binding called what a plugin is, to another connector: the session keeps the plugin
// (Spec.withoutBoundPlugins drops only a plugin whose connector a binding names), and both
// would offer linear__<tool>.
func (s *ConfigsSuite) TestABindingCalledWhatAPluginOfTheConfigIsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "support", "plugins": []string{"linear"}, "connectors": []map[string]any{sessionSlack("linear")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "plugin")
}

// The plugin entry the binding shares its name with must be the binding's own connector: a
// binding called linear to the connector slack leaves both offering linear__<tool>.
func (s *ConfigsSuite) TestABindingCalledWhatAPluginIsToAnotherConnectorIsRefusedWhenThePluginIsNotItsConnector() {
	binding := sessionSlack("linear")
	binding["connector_id"] = "slack"

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "support", "plugins": []string{"linear", "slack"}, "connectors": []map[string]any{binding},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `plugin "linear"`)
}

func (s *ConfigsSuite) TestABindingCalledWhatAUserPluginOfTheConfigIsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":       "support",
		"plugins":    []any{map[string]any{"name": "linear", "user": true}},
		"connectors": []map[string]any{sessionSlack("linear")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `plugin "linear"`)
}

// TestABindingToThePluginsOwnConnectorIsCalledWhatThePluginIs: router plugins migrate binds
// a plugin entry under the plugin's id, to the connector of that id, and keeps the entry
// (AI-994 F42). The session drops the entry for the binding (Spec.withoutBoundPlugins), so the
// two never offer the same name, and the config stays editable.
func (s *ConfigsSuite) TestABindingToThePluginsOwnConnectorIsCalledWhatThePluginIs() {
	for _, user := range []bool{false, true} {
		created := s.createConfig(map[string]any{
			"name":       "support-" + s.utils.uuid(),
			"plugins":    []any{map[string]any{"name": "slack", "user": user}},
			"connectors": []map[string]any{sessionSlack("slack")},
		})

		s.Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
			map[string]any{"instructions": "Answer briefly."}, nil), "user: %v", user)

		read := s.read(created.Id)
		s.Equal("Answer briefly.", value(read.Instructions), "user: %v", user)
		s.Len(value(read.Connectors), 1, "user: %v", user)
	}
}

// TestASavedPluginEntryABindingReplacesIsNotCountedAsUnbound: the deprecation warning is the
// evidence for removing plugins, so an entry the session drops for a binding (a migrated
// config's, with connectors on) is not real use, and an entry no binding replaces (linear, beside a slack binding) is.
func (s *ConfigsSuite) TestASavedPluginEntryABindingReplacesIsNotCountedAsUnbound() {
	migrated := s.createConfig(map[string]any{
		"name":       "migrated-" + s.utils.uuid(),
		"plugins":    []any{map[string]any{"name": "slack"}},
		"connectors": []map[string]any{sessionSlack("slack")},
	})
	live := s.createConfig(map[string]any{
		"name":       "live-" + s.utils.uuid(),
		"plugins":    []any{map[string]any{"name": "slack"}, map[string]any{"name": "linear"}},
		"connectors": []map[string]any{sessionSlack("inbox")},
	})

	for id, unbound := range map[string]string{migrated.Id: "unbound=0", live.Id: "unbound=1"} {
		lines := deprecations(s.logged, plugins.PathConfigSave, id)
		s.Require().Len(lines, 1, id)
		s.Contains(lines[0], " "+unbound+" ", id)
	}
}

func (s *ConfigsSuite) TestABindingCalledWhatAnMCPServerOfTheConfigIsIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name":        "support",
		"mcp_servers": []map[string]any{{"name": "crm", "url": "https://crm.example.com/mcp"}},
		"connectors":  []map[string]any{sessionSlack("crm")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `MCP server "crm"`)
}

func (s *ConfigsSuite) TestABindingWithANullToolsListIsRefused() {
	binding := sessionSlack("inbox")
	binding["tools"] = nil

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
}

func (s *ConfigsSuite) TestPatchingInAPluginABindingIsCalledIsRefused() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("linear")}})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"plugins": []string{"linear"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "plugin")
	s.Nil(s.read(created.Id).Plugins)
}

func (s *ConfigsSuite) TestUpdatingInAPluginAKeptBindingIsCalledIsRefused() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("linear")}})

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "plugins": []string{"linear"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "plugin")
}

func (s *ConfigsSuite) TestMoreBindingsThanAConfigMayHoldAreRefused() {
	bindings := make([]map[string]any, 0, 65)
	for index := range 65 {
		bindings = append(bindings, sessionSlack(fmt.Sprintf("inbox-%d", index)))
	}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": bindings})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "64")
}

func (s *ConfigsSuite) TestMoreToolsThanABindingMayGrantAreRefused() {
	tools := make([]map[string]any, 0, 129)
	for index := range 129 {
		tools = append(tools, map[string]any{"name": fmt.Sprintf("tool-%d", index), "schema_digest": toolDigest})
	}
	binding := sessionSlack("inbox")
	binding["tools"] = tools

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "128")
}

func (s *ConfigsSuite) TestABindingWithoutAToolsListIsRefusedOnCreateAsOnPatch() {
	binding := sessionSlack("inbox")
	delete(binding, "tools")
	created := s.createConfig(map[string]any{"name": "support"})

	created400, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "other", "connectors": []map[string]any{binding}})
	patched400, _ := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, created400)
	s.Equal(http.StatusBadRequest, patched400)
}
