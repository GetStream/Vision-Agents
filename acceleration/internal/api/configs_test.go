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

func (s *ConfigsSuite) TestAConfigRemembersItsScheduling() {
	created := s.createConfig(map[string]any{"name": "support", "scheduling": map[string]any{
		"enabled": true, "max_tasks_per_user": 5, "default_timezone": "Europe/Amsterdam",
	}})
	s.Require().NotNil(created.Scheduling)
	s.True(value(created.Scheduling.Enabled))
	s.Equal(5, value(created.Scheduling.MaxTasksPerUser))
	s.Equal("Europe/Amsterdam", value(created.Scheduling.DefaultTimezone))

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.True(value(read.Scheduling.Enabled))
	s.Equal(5, value(read.Scheduling.MaxTasksPerUser))
	s.Equal("Europe/Amsterdam", value(read.Scheduling.DefaultTimezone))
}

func (s *ConfigsSuite) TestAConfigSchedulingStaysWhenAPatchLeavesItOut() {
	created := s.createConfig(map[string]any{"name": "support", "scheduling": map[string]any{
		"enabled": true, "default_timezone": "Europe/Amsterdam",
	}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+created.Id, map[string]any{"name": "help"}, &patched))
	s.True(value(patched.Scheduling.Enabled))
	s.Equal("Europe/Amsterdam", value(patched.Scheduling.DefaultTimezone))
}

// The dashboard saves an agent by sending back what it read, less the fields the router
// writes. A config that never set scheduling, or never set its task limit, takes that back.
func (s *ConfigsSuite) TestAConfigReadBackIsSavedAsItWas() {
	for name, scheduling := range map[string]map[string]any{
		"no scheduling":            nil,
		"scheduling with no limit": {"enabled": true},
	} {
		s.Run(name, func() {
			body := map[string]any{"name": "support-" + s.utils.uuid(), "instructions": "Be brief."}
			if scheduling != nil {
				body["scheduling"] = scheduling
			}
			created := s.createConfig(body)

			var read map[string]any
			s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
			for _, written := range []string{"id", "created_at", "updated_at", "sync_hash"} {
				delete(read, written)
			}

			var saved AgentConfig
			s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id, read, &saved))
			s.Equal("Be brief.", value(saved.Instructions))
			if scheduling == nil {
				s.NotContains(read, "scheduling", "a config that never set scheduling renders none")
				s.Nil(saved.Scheduling)
				return
			}
			s.Require().NotNil(saved.Scheduling)
			s.True(value(saved.Scheduling.Enabled))
			s.Nil(saved.Scheduling.MaxTasksPerUser, "a limit never set is left out, not rendered as 0")
		})
	}
}

// A scheduling block on an update keeps what it leaves out, as its doc says.
func (s *ConfigsSuite) TestAnUpdateNamingOnlyTheTaskLimitKeepsSchedulingOn() {
	created := s.createConfig(map[string]any{"name": "support", "scheduling": map[string]any{
		"enabled": true, "default_timezone": "Europe/Amsterdam",
	}})

	var updated AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "scheduling": map[string]any{"max_tasks_per_user": 7}}, &updated))

	s.Require().NotNil(updated.Scheduling)
	s.True(value(updated.Scheduling.Enabled))
	s.Equal(7, value(updated.Scheduling.MaxTasksPerUser))
	s.Equal("Europe/Amsterdam", value(updated.Scheduling.DefaultTimezone))
}

func (s *ConfigsSuite) TestAPaddedDefaultTimezoneIsTrimmedAndTheRoutersOwnIsRefused() {
	created := s.createConfig(map[string]any{"name": "support", "scheduling": map[string]any{
		"enabled": true, "default_timezone": " Europe/Amsterdam ",
	}})
	s.Equal("Europe/Amsterdam", value(created.Scheduling.DefaultTimezone))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"scheduling": map[string]any{"default_timezone": "Asia/Tokyo "}}, &patched))
	s.Equal("Asia/Tokyo", value(patched.Scheduling.DefaultTimezone))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "scheduling": map[string]any{"default_timezone": "Local"}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "scheduling.default_timezone")
}

func (s *ConfigsSuite) TestAConfigNamingAnUnknownDefaultTimezoneIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "scheduling": map[string]any{"default_timezone": "Nowhere/Special"}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "scheduling.default_timezone")

	created := s.createConfig(map[string]any{"name": "support"})
	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"scheduling": map[string]any{"default_timezone": "Nowhere/Special"}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "scheduling.default_timezone")
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

func (s *ConfigsSuite) TestAConfigRemembersWhatItsSandboxMayReach() {
	created := s.createConfig(map[string]any{"name": "researcher", "sandbox": "daytona",
		"sandbox_options": map[string]any{
			"network":         true,
			"allowed_domains": []string{"api.example.com"},
		}})

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	options := value(read.SandboxOptions)
	s.True(value(options.Network))
	s.Equal([]string{"api.example.com"}, value(options.AllowedDomains))
	s.Empty(value(options.NetworkAllowList))
}

func (s *ConfigsSuite) TestDomainsAndCIDRRangesTogetherAreRefused() {
	// Daytona takes one allow list per sandbox and refuses to create one given both, which
	// would fail every run of code rather than this one request.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "researcher", "sandbox": "daytona", "sandbox_options": map[string]any{
			"allowed_domains":    []string{"api.example.com"},
			"network_allow_list": []string{"10.0.0.0/8"},
		}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not both")
}

func (s *ConfigsSuite) TestANetworkAllowListEntryThatIsNoCIDRIsRefused() {
	created := s.createConfig(map[string]any{"name": "researcher", "sandbox": "daytona"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"sandbox_options": map[string]any{"network_allow_list": []string{"10.0.0.0"}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "network_allow_list")
}

func (s *ConfigsSuite) TestAnAllowedDomainWithACommaIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "researcher", "sandbox": "daytona",
			"sandbox_options": map[string]any{"allowed_domains": []string{"a.com,b.com"}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "allowed_domains")
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

// privateMCPURLs are servers a session could never reach, since it dials only public
// addresses: one on a private network, a loopback or link-local address, or a name that
// does not resolve.
var privateMCPURLs = []string{
	"https://10.1.2.3/mcp",            // Private-Use, RFC 1918
	"https://100.101.102.103/mcp",     // a Tailscale node: 100.64.0.0/10, https://tailscale.com/kb/1015/100.x-addresses
	"https://[fd7a:115c:a1e0::1]/mcp", // a Tailscale node over IPv6: fd7a:115c:a1e0::/48, same page
	"https://127.0.0.1:8443/mcp",      // Loopback, RFC 1122 section 3.2.1.3
	"https://169.254.169.254/mcp",     // the cloud metadata server, Link Local, RFC 3927
	"https://localhost/mcp",           // resolves to loopback, RFC 6761 section 6.3
	"https://mcp.wwsd.invalid/mcp",    // never resolves, RFC 6761 section 6.4
}

func (s *ConfigsSuite) TestAnMCPServerOnAPrivateAddressIsRefusedWithAPlainError() {
	for _, url := range privateMCPURLs {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "concierge", "mcp_servers": []map[string]any{{"name": "wwsd", "url": url}}})

		s.Equal(http.StatusBadRequest, status, url)
		s.Contains(failure, "mcp_servers: wwsd: only public https URLs are reachable", url)
	}
}

func (s *ConfigsSuite) TestARefusalNamesThePrivateMCPServerAfterAPublicOne() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "concierge",
		"mcp_servers": []map[string]any{
			{"name": "docs", "url": "https://8.8.8.8/mcp"}, // public, Google DNS
			{"name": "wwsd", "url": privateMCPURLs[1]},
		}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "mcp_servers: wwsd: only public https URLs are reachable")
	s.NotContains(failure, "docs")
}

func (s *ConfigsSuite) TestReplacingOrPatchingAConfigRefusesAPrivateMCPServer() {
	created := s.createConfig(map[string]any{"name": "concierge"})
	servers := []map[string]any{{"name": "wwsd", "url": privateMCPURLs[1]}}

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "concierge", "mcp_servers": servers})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "wwsd: only public https URLs are reachable")

	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"mcp_servers": servers})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "wwsd: only public https URLs are reachable")
}

func (s *ConfigsSuite) TestAPublicMCPServerKeepsThePathAndQueryItWasGiven() {
	created := s.createConfig(map[string]any{"name": "concierge",
		"mcp_servers": []map[string]any{{"name": "wwsd", "url": "https://8.8.8.8/mcp?workspace=wwsd"}}})

	s.Equal([]McpServer{{Name: "wwsd", Url: "https://8.8.8.8/mcp?workspace=wwsd"}}, value(created.McpServers))
}

// webhookGuardrail is a guardrail.md asking the server at url.
func webhookGuardrail(url string) string {
	return "---\ntype: webhook\nurl: " + url + "\n---\nMy server decides."
}

// privateWebhookURLs are guardrail webhooks a session could never post to, since it dials
// only public addresses, with what refuses each: egress's check, or the https rule of a
// policy. A path and a query, the endpoint's own, are kept on each.
var privateWebhookURLs = map[string]string{
	"https://10.1.2.3/check":            "must be a public https URL", // Private-Use, RFC 1918
	"https://100.101.102.103/check":     "must be a public https URL", // Shared Address Space, RFC 6598
	"https://[fd7a:115c:a1e0::1]/check": "must be a public https URL", // Unique-Local, RFC 4193
	"https://127.0.0.1:8443/check?a=1":  "must be a public https URL", // Loopback, RFC 1122 section 3.2.1.3
	"https://169.254.169.254/check":     "must be a public https URL", // Link Local, RFC 3927: the cloud metadata server
	"https://[fe80::1]/check":           "must be a public https URL", // Link-Local Unicast, RFC 4291
	"https://localhost/check":           "must be a public https URL", // resolves to loopback, RFC 6761 section 6.3
	"https://2130706433/check":          "must be a public https URL", // 127.0.0.1 written as one number
	"https://hooks.acme.invalid/check":  "must be a public https URL", // never resolves, RFC 6761 section 6.4
	"https://user:pass@8.8.8.8/check":   "must be a public https URL", // userinfo, which egress never sends
	"https:///check":                    "must be a public https URL", // no host
	"http://8.8.8.8/check":              "not an https url",           // plain http, which egress never sends
}

func (s *ConfigsSuite) TestAGuardrailWebhookOnAPrivateAddressIsRefused() {
	for url, complaint := range privateWebhookURLs {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "support", "guardrail": webhookGuardrail(url)})

		s.Equal(http.StatusBadRequest, status, url)
		s.Contains(failure, complaint, url)
	}
}

func (s *ConfigsSuite) TestReplacingOrPatchingAConfigRefusesAPrivateGuardrailWebhook() {
	created := s.createConfig(map[string]any{"name": "support"})
	policy := webhookGuardrail("https://169.254.169.254/check")

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "guardrail": policy})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "guardrail: url must be a public https URL")

	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"guardrail": policy})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "guardrail: url must be a public https URL")
}

func (s *ConfigsSuite) TestAPublicGuardrailWebhookIsKeptAsWritten() {
	// Unchanged from before public addresses were required: the path and the query are the
	// endpoint's own.
	policy := webhookGuardrail("https://8.8.8.8/check?tenant=acme") // public, Google DNS
	created := s.createConfig(map[string]any{"name": "support", "guardrail": policy})
	s.Equal(policy, value(created.Guardrail))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"guardrail": webhookGuardrail("https://[2001:4860:4860::8888]/check")}, &patched))
	s.Equal(webhookGuardrail("https://[2001:4860:4860::8888]/check"), value(patched.Guardrail))
}

func (s *ConfigsSuite) TestAConfigStoredWithAPrivateMCPServerStillReads() {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "concierge", Mode: "text",
		MCPServers: []store.MCPServer{{Name: "wwsd", URL: privateMCPURLs[1]}}}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+config.ID, nil, &read))
	s.Equal([]McpServer{{Name: "wwsd", Url: privateMCPURLs[1]}}, value(read.McpServers))
}

func (s *ConfigsSuite) TestAVoiceConfigPinnedToABatchTranscriberIsRefused() {
	pinned := "batch/" + batchModel

	status, raw := s.serverClient.call(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "stt": pinned})

	s.Equal(http.StatusBadRequest, status)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(raw, &answered), "the error envelope: %s", raw)
	s.Equal(ErrorTypeInvalidRequest, answered.Error.Type)
	s.Equal("routing: "+pinned+" is a batch model for recordings, not live calls; for a live call name en-low-latency",
		answered.Error.Message, "the refusal a session start gives, said at the save")
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs", nil, &listed))
	s.Empty(listed, "nothing was stored")
}

func (s *ConfigsSuite) TestAVoiceConfigWithALiveTranscriberIsSaved() {
	created := s.createConfig(map[string]any{"name": "support", "stt": "en-low-latency"})

	s.Equal("en-low-latency", value(created.Stt))
}

func (s *ConfigsSuite) TestAConfigThatOpensNoTranscriberMayNameABatchOne() {
	pinned := "batch/" + batchModel

	written := s.createConfig(map[string]any{"name": "analyst", "mode": "text", "stt": pinned})
	native := s.createConfig(map[string]any{"name": "concierge", "sts": "sts-fast", "stt": pinned})

	s.Equal(pinned, value(written.Stt), "a config written to has no transcriber")
	s.Equal(pinned, value(native.Stt), "an STS model hears for itself")
}

func (s *ConfigsSuite) TestUpdatingAConfigOntoABatchTranscriberIsRefused() {
	created := s.createConfig(map[string]any{"name": "support", "stt": "en-low-latency"})

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "stt": "batch/" + batchModel})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "is a batch model for recordings, not live calls")
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal("en-low-latency", value(read.Stt), "the stored config is as it was")
}

func (s *ConfigsSuite) TestPatchingAConfigOntoABatchTranscriberIsRefused() {
	created := s.createConfig(map[string]any{"name": "support", "stt": "en-low-latency"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"stt": "batch/" + batchModel})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "is a batch model for recordings, not live calls")
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal("en-low-latency", value(read.Stt), "the stored config is as it was")
}

func (s *ConfigsSuite) TestAConfigStoredWithABatchTranscriberCanStillBeEdited() {
	// Stored before saves were checked; its sessions are refused as they start instead.
	pinned := "batch/" + batchModel
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "support", Mode: store.AgentModeVoice, STT: pinned}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))

	var updated, patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+config.ID,
		map[string]any{"name": "support", "stt": pinned, "instructions": "Be brief."}, &updated))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+config.ID,
		map[string]any{"instructions": "Be briefer."}, &patched))

	s.Equal("Be brief.", value(updated.Instructions))
	s.Equal("Be briefer.", value(patched.Instructions))
	s.Equal(pinned, value(patched.Stt))
}

func (s *ConfigsSuite) TestADeploymentWithoutATranscriberRouterRefusesNoConfig() {
	config := store.AgentConfig{Mode: store.AgentModeVoice, STT: "batch/" + batchModel}

	for name, server := range map[string]*Server{"no streams": {}, "no stt router": {streams: &Streams{}}} {
		_, ok := server.liveSTTComplaint(config, "")

		s.True(ok, name)
	}
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
		"mcp_servers": []map[string]any{{"name": "crm", "url": "https://8.8.8.8/mcp"}},
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

// leftOutSQL is the refusal of a Metabase binding granting execute_sql: revision 2 of
// providers/metabase.yaml says it needs agent:sql:run, which its scopes.list never asks for.
const leftOutSQL = `connector binding "metabase" grants "execute_sql", which connector "metabase" leaves out: ` +
	`it needs agent:sql:run, which the connector never asks for`

func (s *ConfigsSuite) TestASessionBindingGrantingAToolItsConnectorLeavesOutIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "support", "connectors": []map[string]any{sessionMetabase("search", "execute_sql")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, leftOutSQL)
}

func (s *ConfigsSuite) TestAFixedBindingGrantingAToolItsConnectorLeavesOutIsRefused() {
	binding := fixedSlack("metabase", s.connectionTo("metabase", ""))
	binding["connector_id"] = "metabase"
	binding["tools"] = []map[string]any{{"name": "execute_sql", "schema_digest": toolDigest}}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, leftOutSQL)
}

// TestAGrantIsJudgedByTheMCPSourceAlone: a source of another kind carrying a rule for the
// tool does not leave it out, as the mcp source's filter reads the mcp source only.
func (s *ConfigsSuite) TestAGrantIsJudgedByTheMCPSourceAlone() {
	manifest := core.Manifest{Sources: []core.SourceRule{
		{Kind: "rest", Tools: []core.ToolRule{{Name: "execute_sql", NeedsScopes: []string{"agent:sql:run"}}}},
		{Kind: core.SourceKindMCP},
	}}
	binding := AgentConnectorBinding{Name: "metabase", ConnectorId: "metabase", Tools: []ConnectorToolGrant{{Name: "execute_sql"}}}

	complaint, ok := leftOutGrant(binding, manifest)

	s.True(ok, complaint)
}

// TestAGrantOfAToolWhoseScopeAnAskedScopeCoversIsStored: scopes.list asks for agent:sql:all,
// which the manifest says covers the agent:sql:run execute_sql needs, so a consent grants it.
func (s *ConfigsSuite) TestAGrantOfAToolWhoseScopeAnAskedScopeCoversIsStored() {
	manifest := core.Manifest{
		Scopes:  core.ScopePolicy{List: []string{"agent:sql:all"}, CoveredBy: map[string][]string{"agent:sql:run": {"agent:sql:all"}}},
		Sources: []core.SourceRule{{Kind: core.SourceKindMCP, Tools: []core.ToolRule{{Name: "execute_sql", NeedsScopes: []string{"agent:sql:run"}}}}},
	}
	binding := AgentConnectorBinding{Name: "metabase", ConnectorId: "metabase", Tools: []ConnectorToolGrant{{Name: "execute_sql"}}}

	complaint, ok := leftOutGrant(binding, manifest)
	s.True(ok, complaint)

	manifest.Scopes.CoveredBy = nil
	complaint, ok = leftOutGrant(binding, manifest)
	s.False(ok)
	s.Contains(complaint, "it needs agent:sql:run, which the connector never asks for")
}

// TestAGrantOnAConnectorWithNoMCPSourceIsStored: linq, slack_bot, whatsapp and telnyx have
// no mcp source, so no filter leaves a tool out and the grant is not refused.
func (s *ConfigsSuite) TestAGrantOnAConnectorWithNoMCPSourceIsStored() {
	manifest := core.Manifest{Sources: []core.SourceRule{{Kind: "rest"}}}
	binding := AgentConnectorBinding{Name: "chat", ConnectorId: "linq", Tools: []ConnectorToolGrant{{Name: "send"}}}

	complaint, ok := leftOutGrant(binding, manifest)

	s.True(ok, complaint)
}

func (s *ConfigsSuite) TestReplacingOrPatchingInAToolTheConnectorLeavesOutIsRefused() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "connectors": []map[string]any{sessionMetabase("execute_sql")}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, leftOutSQL)

	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{sessionMetabase("execute_sql")}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, leftOutSQL)

	s.Equal(created.Connectors, s.read(created.Id).Connectors)
}

// TestABindingGrantingOnlyToolsItsConnectorOffersIsStored is the control: the same bindings
// without execute_sql are stored, as on the base b7cea98f.
func (s *ConfigsSuite) TestABindingGrantingOnlyToolsItsConnectorOffersIsStored() {
	fixed := fixedSlack("analytics", s.connectionTo("metabase", ""))
	fixed["connector_id"] = "metabase"

	created := s.createConfig(map[string]any{
		"name": "support", "connectors": []map[string]any{fixed, sessionMetabase("search")},
	})

	got := value(s.read(created.Id).Connectors)
	s.Require().Len(got, 2)
	s.Equal([]ConnectorToolGrant{{Name: "search", SchemaDigest: toolDigest}}, got[0].Tools)
	s.Equal([]ConnectorToolGrant{{Name: "search"}}, got[1].Tools)
}

// TestAPatternBindingToAConnectorThatLeavesToolsOutIsStored: a pattern matches only the tools
// the connector lists, so it never grants one left out.
func (s *ConfigsSuite) TestAPatternBindingToAConnectorThatLeavesToolsOutIsStored() {
	binding := sessionMetabase()
	binding["tool_patterns"] = []string{"*"}

	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal([]string{"*"}, value(value(s.read(created.Id).Connectors)[0].ToolPatterns))
}

// TestAnEmptyToolPatternsListIsRefused: tool_patterns [] is not a grant of every tool.
// The field requires at least one pattern, on create, replace, patch and sync.
func (s *ConfigsSuite) TestAnEmptyToolPatternsListIsRefused() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = []string{}
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "patterned", "connectors": []map[string]any{binding}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "expected array length >= 1")

	status, failure = s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "expected array length >= 1")

	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{binding}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "expected array length >= 1")

	status, failure = s.serverClient.failure(http.MethodPost, "/v1/agents/sync",
		map[string]any{"name": "synced", "hash": "v1", "connectors": []map[string]any{binding}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "expected array length >= 1")

	kept := value(s.read(created.Id).Connectors)[0]
	s.Nil(kept.ToolPatterns)
	s.Equal([]ConnectorToolGrant{{Name: "search", SchemaDigest: toolDigest}}, kept.Tools)
}

func (s *ConfigsSuite) TestPatternsAndToolsTogetherAreRefused() {
	binding := sessionSlack("inbox")
	binding["tool_patterns"] = []string{"*"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "one or the other")
}

func (s *ConfigsSuite) TestAToolPatternThatCannotMatchIsRefused() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = []string{"["}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not a tool name or pattern")
}

func (s *ConfigsSuite) TestMoreThan128ToolPatternsAreRefused() {
	patterns := make([]string, 129)
	for index := range patterns {
		patterns[index] = "search"
	}
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = patterns

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "128")
}

func (s *ConfigsSuite) TestAFixedBindingWithPatternsAndNoDigestsIsAccepted() {
	binding := fixedSlack("inbox", s.connection(""))
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = []string{"search", "linear_*"}

	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	got := value(s.read(created.Id).Connectors)[0]
	s.Empty(got.Tools)
	s.Equal([]string{"search", "linear_*"}, value(got.ToolPatterns))
}

func (s *ConfigsSuite) TestGetAndListAnswerToolPatterns() {
	binding := sessionSlack("inbox")
	binding["tools"] = []map[string]any{}
	binding["tool_patterns"] = []string{"*"}
	created := s.createConfig(map[string]any{"name": "patterned", "connectors": []map[string]any{binding}})

	s.Equal([]string{"*"}, value(value(s.read(created.Id).Connectors)[0].ToolPatterns))
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name=patterned", nil, &listed))
	s.Require().NotEmpty(listed)
	s.Equal([]string{"*"}, value(value(listed[0].Connectors)[0].ToolPatterns))
}

func (s *ConfigsSuite) TestPutAndPatchStoreToolPatterns() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})
	put := sessionSlack("inbox")
	put["tools"] = []map[string]any{}
	put["tool_patterns"] = []string{"search_*"}

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "connectors": []map[string]any{put}}, nil))
	s.Equal([]string{"search_*"}, value(value(s.read(created.Id).Connectors)[0].ToolPatterns))

	patched := sessionSlack("crm")
	patched["tools"] = []map[string]any{}
	patched["tool_patterns"] = []string{"*"}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{patched}}, nil))

	got := value(s.read(created.Id).Connectors)[0]
	s.Equal("crm", got.Name)
	s.Equal([]string{"*"}, value(got.ToolPatterns))
}

// TestABindingWithOnlyToolsIsAnsweredWithoutToolPatterns is the control for the answer.
// On e70351d4 a tools-only binding's GET body has no tool_patterns key: the field did not
// exist. A binding that still grants tools by name is answered the same way.
func (s *ConfigsSuite) TestABindingWithOnlyToolsIsAnsweredWithoutToolPatterns() {
	created := s.createConfig(map[string]any{"name": "support", "connectors": []map[string]any{sessionSlack("inbox")}})

	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/configs/"+created.Id, nil)

	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(raw), "tool_patterns")
	s.Equal([]ConnectorToolGrant{{Name: "search", SchemaDigest: toolDigest}}, value(s.read(created.Id).Connectors)[0].Tools)
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

// sessionMetabase binds Metabase through the connection a session's end user picks,
// granting tools by name.
func sessionMetabase(tools ...string) map[string]any {
	binding := sessionSlack("metabase")
	binding["connector_id"] = "metabase"
	grants := []map[string]any{}
	for _, tool := range tools {
		grants = append(grants, map[string]any{"name": tool})
	}
	binding["tools"] = grants
	return binding
}

// connection stores a Slack connection of the suite's app, the app's own when owner is
// empty and that user's otherwise, and returns its id. It is written through the store,
// since no endpoint makes one yet.
func (s *ConfigsSuite) connection(owner string) string {
	return s.connectionTo("slack", owner)
}

// connectionTo is connection to another built-in connector.
func (s *ConfigsSuite) connectionTo(connectorID, owner string) string {
	ctx := context.Background()
	definition, err := s.store.LatestConnectorDefinition(ctx, s.customerID(), connectorID)
	s.Require().NoError(err)
	connection := &store.ConnectorConnection{
		CustomerID: s.customerID(), ConnectorID: definition.ID, DefinitionRevision: definition.Revision,
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
		"mcp_servers": []map[string]any{{"name": "crm", "url": "https://8.8.8.8/mcp"}},
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

// sessionGitHub binds GitHub, whose manifest has a sandbox block (GITHUB_TOKEN), through the
// connection a session's end user picks, granting no tools by name. sandbox sets the binding's
// sandbox key; false leaves it out, as a client written before it existed does.
func sessionGitHub(alias string, sandbox bool) map[string]any {
	binding := map[string]any{
		"name": alias, "connector_id": "github",
		"connection": map[string]any{"type": "session"},
		"tools":      []map[string]any{},
	}
	if sandbox {
		binding["sandbox"] = true
	}
	return binding
}

func (s *ConfigsSuite) TestAGitHubSandboxBindingIsStoredAndReadBack() {
	created := s.createConfig(map[string]any{"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionGitHub("code", true)}})

	s.Require().Len(value(created.Connectors), 1)
	s.True(value(created.Connectors)[0].Sandbox)
	read, stored := s.readConnectors(created.Id)
	s.JSONEq(`[{"name":"code","connector_id":"github","connection":{"type":"session"},"tools":[],"required":false,"sandbox":true}]`, string(read))
	s.JSONEq(`[{"name":"code","connector_id":"github","connection":{"type":"session"},"tools":[],"required":false,"sandbox":true}]`, stored)
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name=builder", nil, &listed))
	s.Require().Len(listed, 1)
	s.True(value(listed[0].Connectors)[0].Sandbox)
}

// TestABindingWithoutSandboxIsReadAsOnTheBase is the control: the GET answer and the row are
// what the base, a8181c43b, answered and stored for the same request (pr-w13-bind/author-1.md,
// «Control probe»).
func (s *ConfigsSuite) TestABindingWithoutSandboxIsReadAsOnTheBase() {
	created := s.createConfig(map[string]any{"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionGitHub("code", false)}})

	read, stored := s.readConnectors(created.Id)

	s.Equal(`[{"name":"code","connector_id":"github","connection":{"type":"session"},"tools":[],"required":false}]`, string(read))
	s.Equal(`[{"name": "code", "tools": [], "required": false, "connection": {"type": "session"}, "connector_id": "github"}]`, stored)
}

func (s *ConfigsSuite) TestASandboxBindingToAConnectorWithoutASandboxBlockIsRefused() {
	slack := sessionSlack("inbox")
	slack["sandbox"] = true

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "builder", "sandbox": "daytona", "connectors": []map[string]any{slack}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector "slack" offers no access from a sandbox`)
}

func (s *ConfigsSuite) TestASandboxBindingInAConfigWithoutASandboxIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "builder", "connectors": []map[string]any{sessionGitHub("code", true)}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "code" sets sandbox, and the config has no sandbox`)
}

func (s *ConfigsSuite) TestTwoSandboxBindingsGivingTheSandboxOneEnvVarAreRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionGitHub("code", true), sessionGitHub("docs", true)},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector bindings "code" and "docs" both set sandbox`)
	s.Contains(failure, "GITHUB_TOKEN")
}

// TestTwoSandboxBindingsToDifferentConnectorsGivingOneEnvVarAreRefused: the clash is on the env
// var the sandbox holds, not on the connector, so a custom connector naming GITHUB_TOKEN clashes
// with github.
func (s *ConfigsSuite) TestTwoSandboxBindingsToDifferentConnectorsGivingOneEnvVarAreRefused() {
	source, err := providers.FS.ReadFile("github.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(source)
	s.Require().NoError(err)
	manifest.ID = "custom_" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	custom := sessionGitHub("docs", true)
	custom["connector_id"] = manifest.ID

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionGitHub("code", true), custom},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector bindings "code" and "docs" both set sandbox`)
	s.Contains(failure, "GITHUB_TOKEN")
}

func (s *ConfigsSuite) TestTwoBindingsToOneConnectorMayHaveSandboxOnOne() {
	created := s.createConfig(map[string]any{"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionGitHub("code", true), sessionGitHub("docs", false)}})

	s.Len(value(created.Connectors), 2)
}

// TestReplacingAConfigWithoutItsSandboxRefusesTheSandboxBindingItKeeps: a replace that leaves
// connectors out keeps the bindings stored, and a sandbox binding needs the sandbox it drops.
func (s *ConfigsSuite) TestReplacingAConfigWithoutItsSandboxRefusesTheSandboxBindingItKeeps() {
	created := s.createConfig(map[string]any{"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionGitHub("code", true)}})

	status, failure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "builder"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "code" sets sandbox, and the config has no sandbox`)
	s.Equal(Daytona, value(s.read(created.Id).Sandbox))
}

func (s *ConfigsSuite) TestPatchingASandboxBindingOntoAConfigWithoutASandboxIsRefused() {
	created := s.createConfig(map[string]any{"name": "builder"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{sessionGitHub("code", true)}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "code" sets sandbox, and the config has no sandbox`)
	s.Nil(s.read(created.Id).Connectors)
}

// sessionNessy binds nessy, which nessy.yaml marks sensitive, through the connection a
// session's end user picks, granting org_lookup, the first of Nessy's read-only tools
// (GetStream/Nessy backend/agenttools/api.go:35-36 at 7e238ffa).
func sessionNessy(alias string) map[string]any {
	return map[string]any{
		"name": alias, "connector_id": "nessy",
		"connection": map[string]any{"type": "session"},
		"tools":      []map[string]any{{"name": "org_lookup"}},
	}
}

// sessionGitHubGranting is sessionGitHub granting tools by name, which github.yaml's source
// does not say only read.
func sessionGitHubGranting(alias string, tools ...string) map[string]any {
	binding := sessionGitHub(alias, false)
	granted := []map[string]any{}
	for _, tool := range tools {
		granted = append(granted, map[string]any{"name": tool})
	}
	binding["tools"] = granted
	return binding
}

// useInternalApp gives the test an app Stream marks internal, the only kind that sees nessy.
func (s *ConfigsSuite) useInternalApp() {
	s.useApp(s.data.createAppAdmitting(store.AppSettings{InternalConnectors: new(true)}))
}

// TestNessyBesideAnotherConnectorsWriteToolIsRefused is AI-1545's «Done when»: a config
// binding nessy and a tool that may send data out is a 400 naming both, and stores nothing.
func (s *ConfigsSuite) TestNessyBesideAnotherConnectorsWriteToolIsRefused() {
	s.useInternalApp()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{sessionNessy("customers"), sessionGitHubGranting("code", "create_issue")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "customers" binds "nessy", whose tools read private data, `+
		`and connector binding "code" can send data out with no person's approval`)
	var listed []AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs?name=athena", nil, &listed))
	s.Empty(listed)
}

// A tool pattern grants whatever matches, so it is as outbound as a tool named.
func (s *ConfigsSuite) TestNessyBesideAToolPatternBindingIsRefused() {
	s.useInternalApp()
	patterned := sessionGitHub("code", false)
	patterned["tool_patterns"] = []string{"list_*"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{patterned, sessionNessy("customers")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "code" can send data out`)
}

// A binding granting no tool offers the model nothing, so it sends nothing out, unless its
// sandbox key hands code the connector's whole API.
func (s *ConfigsSuite) TestNessyBesideABindingGrantingNoToolIsAcceptedUnlessItsCodeCallsTheAPI() {
	s.useInternalApp()
	s.createConfig(map[string]any{"name": "athena", "sandbox": "daytona",
		"connectors": []map[string]any{sessionNessy("customers"), sessionGitHub("code", false)}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "builder", "sandbox": "daytona",
		"connectors": []map[string]any{sessionNessy("customers"), sessionGitHub("code", true)},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "code" can send data out`)
}

// slack_bot.yaml lists no source, so a binding to it offers no tool, whatever it grants.
func (s *ConfigsSuite) TestNessyBesideAConnectorWithNoSourceIsAccepted() {
	s.useInternalApp()
	bot := sessionSlack("bot")
	bot["connector_id"] = "slack_bot"

	created := s.createConfig(map[string]any{"name": "athena",
		"connectors": []map[string]any{sessionNessy("customers"), bot}})

	s.Len(value(created.Connectors), 2)
}

// Every tool that sends data out is named, so one answer says all that has to go.
func (s *ConfigsSuite) TestNessyBesideSeveralOutboundToolsNamesThemAll() {
	s.useInternalApp()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "sandbox": "daytona", "sandbox_options": map[string]any{"network": true},
		"connectors":  []map[string]any{sessionGitHubGranting("code", "create_issue"), sessionNessy("customers")},
		"mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}},
		"plugins":     []map[string]any{{"name": "github"}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "customers" binds "nessy"`)
	s.Contains(failure, `and connector binding "code", MCP server "tablejourney", plugin "github", `+
		`the sandbox's network access can send data out with no person's approval`)
}

// TestNessyBesideAConnectorWithOneOutboundSourceAmongSeveralIsRefused: a connector sends data
// out if any of its sources may, not only the first. Manifest.Validate accepts several sources
// and CreateConnectorDefinition takes any manifest, so a custom connector can put a source that
// says outbound: false before one that does not.
func (s *ConfigsSuite) TestNessyBesideAConnectorWithOneOutboundSourceAmongSeveralIsRefused() {
	s.useInternalApp()
	source, err := providers.FS.ReadFile("github.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(source)
	s.Require().NoError(err)
	manifest.ID = "custom_" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	readOnly := manifest.Sources[0]
	readOnly.Outbound = new(false)
	manifest.Sources = []core.SourceRule{readOnly, manifest.Sources[0]}
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	custom := sessionGitHubGranting("code", "create_issue")
	custom["connector_id"] = manifest.ID

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{sessionNessy("customers"), custom},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "code" can send data out with no person's approval`)
}

// sessionGranting binds connectorID through the connection a session's end user picks,
// granting tools by name.
func sessionGranting(connectorID, alias string, tools ...string) map[string]any {
	binding := sessionGitHubGranting(alias, tools...)
	binding["connector_id"] = connectorID
	return binding
}

// Every tool google_drive.yaml, google_docs.yaml and bigquery.yaml list in read_only_tools
// only reads, so a binding granting only those sits beside nessy.
func (s *ConfigsSuite) TestNessyBesideOnlyTheReadOnlyToolsOfDriveDocsAndBigQueryIsAccepted() {
	s.useInternalApp()

	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{
		sessionNessy("customers"),
		sessionGranting("google_drive", "files", "download_file_content", "get_file_metadata",
			"get_file_permissions", "list_recent_files", "read_file_content", "search_files"),
		sessionGranting("google_docs", "docs", "read_doc"),
		sessionGranting("bigquery", "warehouse", "list_dataset_ids", "get_dataset_info", "list_table_ids",
			"get_table_info", "get_query_results", "get_job"),
	}})

	s.Len(value(created.Connectors), 4)
}

// bigquery.yaml revision 3 takes execute_sql_readonly out of read_only_tools: a SELECT can
// call a remote function, whose code may send the rows anywhere. So it is refused beside
// nessy, and accepted once the binding's require_approval holds it. The base, 8e132bd37,
// answered 201 to the refused request (ai1538-followups/pr2/author-1.md, «Control probe»).
func (s *ConfigsSuite) TestNessyBesideBigQuerysExecuteSQLReadonlyNeedsItsApproval() {
	s.useInternalApp()
	warehouse := sessionGranting("bigquery", "warehouse", "get_table_info", "execute_sql_readonly")

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{sessionNessy("customers"), warehouse}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "warehouse" can send data out with no person's approval`)
	s.Contains(failure, `"warehouse" execute_sql_readonly`)
	warehouse["require_approval"] = []string{"execute_sql_readonly"}
	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{sessionNessy("customers"), warehouse}})
	s.Len(value(created.Connectors), 2)
}

// A tool the source leaves out of read_only_tools may write, so a binding granting one beside
// tools that only read is refused, each such binding named.
func (s *ConfigsSuite) TestNessyBesideAWriteToolOfDriveDocsOrBigQueryNamesEachBindingGrantingOne() {
	s.useInternalApp()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{
			sessionNessy("customers"),
			sessionGranting("google_drive", "drive_copy", "search_files", "copy_file"),
			sessionGranting("google_drive", "drive_create", "create_file"),
			sessionGranting("google_drive", "drive_read", "read_file_content"),
			sessionGranting("google_docs", "docs_update", "read_doc", "update_doc"),
			sessionGranting("bigquery", "bq_execute", "execute_sql"),
			sessionGranting("bigquery", "bq_cancel", "get_job", "cancel_job"),
		},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "customers" binds "nessy", whose tools read private data, `+
		`and connector binding "drive_copy", connector binding "drive_create", connector binding "docs_update", `+
		`connector binding "bq_execute", connector binding "bq_cancel" can send data out with no person's approval`)
}

// nessy.yaml's source says outbound: false, so nessy alone, or twice, is accepted.
func (s *ConfigsSuite) TestNessyAloneIsAccepted() {
	s.useInternalApp()

	created := s.createConfig(map[string]any{"name": "athena",
		"connectors": []map[string]any{sessionNessy("customers"), sessionNessy("billing")}})

	s.Len(value(created.Connectors), 2)
}

func (s *ConfigsSuite) TestNessyBesideAnMCPServerIsRefused() {
	s.useInternalApp()

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{sessionNessy("customers")},
		"mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `and MCP server "tablejourney" can send data out`)
}

// A plugin's read-only endpoint offers no tool that writes (PluginEntry.Readonly), so only the
// plugin reached at its full one is refused.
func (s *ConfigsSuite) TestNessyBesideAPluginIsRefusedUnlessItIsReadOnly() {
	s.useInternalApp()
	s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{sessionNessy("customers")},
		"plugins": []map[string]any{{"name": "linear", "readonly": true}}})

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "builder", "connectors": []map[string]any{sessionNessy("customers")},
		"plugins": []map[string]any{{"name": "linear", "readonly": true}, {"name": "github"}},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `and plugin "github" can send data out`)
}

// A sandbox reaches no network unless its options open some (sandbox.Config.Isolated), and
// options with no sandbox run no code.
func (s *ConfigsSuite) TestNessyBesideASandboxIsRefusedOnlyWhenItReachesTheNetwork() {
	s.useInternalApp()
	s.createConfig(map[string]any{"name": "athena", "sandbox": "daytona",
		"connectors": []map[string]any{sessionNessy("customers")}})
	s.createConfig(map[string]any{"name": "researcher", "sandbox_options": map[string]any{"network": true},
		"connectors": []map[string]any{sessionNessy("customers")}})

	for name, options := range map[string]map[string]any{
		"network": {"network": true},
		"domains": {"allowed_domains": []string{"example.com"}},
		"cidrs":   {"network_allow_list": []string{"10.0.0.0/8"}},
	} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
			"name": name, "sandbox": "daytona", "sandbox_options": options,
			"connectors": []map[string]any{sessionNessy("customers")},
		})

		s.Equal(http.StatusBadRequest, status, name)
		s.Contains(failure, "and the sandbox's network access can send data out", name)
	}
}

// An update and a patch read the config as it is about to be stored: a binding added to a
// config with an MCP server is refused, and so is the MCP server added to a config binding nessy.
func (s *ConfigsSuite) TestAnUpdateOrAPatchBringingNessyBesideAnOutboundToolIsRefused() {
	s.useInternalApp()
	created := s.createConfig(map[string]any{"name": "athena",
		"connectors": []map[string]any{sessionNessy("customers")}})

	updated, updateFailure := s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "athena", "mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}}})
	patched, patchFailure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{sessionNessy("customers"), sessionGitHubGranting("code", "create_issue")}})

	s.Equal(http.StatusBadRequest, updated)
	s.Contains(updateFailure, `and MCP server "tablejourney" can send data out`)
	s.Equal(http.StatusBadRequest, patched)
	s.Contains(patchFailure, `and connector binding "code" can send data out`)
	s.Len(value(s.read(created.Id).Connectors), 1)
	s.Nil(s.read(created.Id).McpServers)
}

// TestAConfigWithoutASensitiveConnectorKeepsItsOutboundTools is the control: the base,
// a103c19ab, answered 201 to the same request (ai1545/author-1.md, «Control probe»).
func (s *ConfigsSuite) TestAConfigWithoutASensitiveConnectorKeepsItsOutboundTools() {
	created := s.createConfig(map[string]any{
		"name": "builder", "sandbox": "daytona", "sandbox_options": map[string]any{"network": true},
		"connectors":  []map[string]any{sessionGitHubGranting("code", "create_issue")},
		"mcp_servers": []map[string]any{{"name": "tablejourney", "url": "https://tablejourney.com/mcp"}},
		"plugins":     []map[string]any{{"name": "github"}},
	})

	s.Len(value(created.Connectors), 1)
}

// TestAConfigWithoutASensitiveConnectorKeepsTheWriteToolsOfDriveDocsAndBigQuery is the
// control: the base, 29091a594, answered 201 to the same request (ai-1844/author-1.md,
// «Control probe»), and 8e132bd37 with execute_sql_readonly granted too
// (ai1538-followups/pr2/author-1.md, «Control probe»).
func (s *ConfigsSuite) TestAConfigWithoutASensitiveConnectorKeepsTheWriteToolsOfDriveDocsAndBigQuery() {
	created := s.createConfig(map[string]any{"name": "builder", "connectors": []map[string]any{
		sessionGranting("google_drive", "files", "create_file", "copy_file"),
		sessionGranting("google_docs", "docs", "update_doc"),
		sessionGranting("bigquery", "warehouse", "execute_sql", "cancel_job", "execute_sql_readonly"),
	}})

	s.Len(value(created.Connectors), 3)
}

// TestAnUpdateKeepingABindingToAForceDeletedConnectorIsAccepted: a forced connector delete
// leaves its bindings behind (store.refuseUndefined), and a binding to a connector with no
// definition opens nothing. The base, a103c19ab, answered 200 (ai1545/author-1.md, «Control
// probe»).
func (s *ConfigsSuite) TestAnUpdateKeepingABindingToAForceDeletedConnectorIsAccepted() {
	source, err := providers.FS.ReadFile("github.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(source)
	s.Require().NoError(err)
	manifest.ID = "custom_" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	custom := sessionGitHubGranting("code", "create_issue")
	custom["connector_id"] = manifest.ID
	created := s.createConfig(map[string]any{"name": "builder", "connectors": []map[string]any{custom}})
	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/connectors/"+manifest.ID+"?force=true", nil, nil))

	status, _ := s.serverClient.call(http.MethodPut, "/v1/agents/configs/"+created.Id, map[string]any{"name": "builder"})

	s.Equal(http.StatusOK, status)
}

// Tool output is kept as long as the session unless a config says otherwise, an update
// that leaves the setting out keeps it, and an empty one removes it.
func (s *ConfigsSuite) TestAConfigKeepsToolOutputForAsLongAsItSays() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.Nil(unnamed.Tools.OutputRetention)

	created := s.createConfig(map[string]any{"name": "billing", "tools": map[string]any{"output_retention": "7d"}})
	s.Equal("7d", value(created.Tools.OutputRetention))

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+created.Id, map[string]any{"name": "billing", "tools": map[string]any{"progressive": true}}, &saved))
	s.Equal("7d", value(saved.Tools.OutputRetention), "a setting left out on an update stays")

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+created.Id, map[string]any{"tools": map[string]any{"output_retention": "none"}}, &patched))
	s.Equal("none", value(patched.Tools.OutputRetention))

	var cleared AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+created.Id, map[string]any{"tools": map[string]any{"output_retention": ""}}, &cleared))
	s.Nil(cleared.Tools.OutputRetention)
}

func (s *ConfigsSuite) TestAToolOutputRetentionOutsideItsBoundsIsRefusedOnEveryWrite() {
	created := s.createConfig(map[string]any{"name": "support"})
	tooLong := map[string]any{"output_retention": "3651d"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "billing", "tools": tooLong})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "tools.output_retention")
	status, failure = s.serverClient.failure(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "tools": tooLong})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "tools.output_retention")
	status, failure = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"tools": tooLong})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "tools.output_retention")
}
