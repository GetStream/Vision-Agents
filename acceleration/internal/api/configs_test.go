//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/stretchr/testify/require"
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

func (s *ConfigsSuite) TestAnImageTargetIsStoredOnCreateAndOnEveryUpdate() {
	// An update writes a named list of columns; a field missing from it is kept on create and
	// silently dropped by every save after.
	created := s.createConfig(map[string]any{"name": "illustrator", "mode": "text", "image": "image-fast"})
	s.Equal("image-fast", value(created.Image))

	var updated AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "illustrator", "mode": "text", "image": "image-quality"}, &updated))
	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal("image-quality", value(read.Image))

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"instructions": "draw flat illustrations"}, &patched))
	s.Equal("image-quality", value(patched.Image), "a patch that leaves it out keeps it")
}

func (s *ConfigsSuite) TestAConfigWithoutAnImageTargetHasNone() {
	created := s.createConfig(map[string]any{"name": "support", "mode": "text"})

	s.Nil(created.Image, "unlike search there is no default: a picture costs money")
}

func (s *ConfigsSuite) TestPatchingAnEmptyImageTargetRemovesIt() {
	created := s.createConfig(map[string]any{"name": "illustrator", "mode": "text", "image": "image-fast"})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"image": ""}, &patched))

	s.Nil(patched.Image)
}

func (s *ConfigsSuite) TestAConfigKeepsItsLLMFallbacksInOrder() {
	created := s.createConfig(map[string]any{"name": "nora", "llm": "echo/echo-model",
		"llm_fallbacks": []string{"recites/recites-model", "checking/checking-model"}})

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal([]string{"recites/recites-model", "checking/checking-model"}, value(read.LlmFallbacks))
}

func (s *ConfigsSuite) TestAnUpdateThatLeavesOutTheLLMFallbacksKeepsThemAndAnEmptyListRemovesThem() {
	created := s.createConfig(map[string]any{"name": "nora", "llm": "echo/echo-model",
		"llm_fallbacks": []string{"recites/recites-model"}})

	var kept AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "nora", "llm": "checking/checking-model"}, &kept))
	s.Equal([]string{"recites/recites-model"}, value(kept.LlmFallbacks), "a client that does not know the list keeps it")

	var removed AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "nora", "llm": "checking/checking-model", "llm_fallbacks": []string{}}, &removed))
	s.Nil(removed.LlmFallbacks)
}

func (s *ConfigsSuite) TestPatchingTheLLMFallbacksReplacesTheListWhole() {
	created := s.createConfig(map[string]any{"name": "nora", "llm": "echo/echo-model",
		"llm_fallbacks": []string{"recites/recites-model", "checking/checking-model"}})

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"llm_fallbacks": []string{"checking/checking-model"}}, &patched))

	s.Equal([]string{"checking/checking-model"}, value(patched.LlmFallbacks))
	s.Equal("echo/echo-model", value(patched.Llm), "llm stays the first choice")
}

func (s *ConfigsSuite) TestAnLLMFallbackThisDeploymentDoesNotRouteIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "nora", "llm": "echo/echo-model", "llm_fallbacks": []string{"xai/grok-typo"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "llm_fallbacks")
	s.Contains(failure, "xai/grok-typo")
}

func (s *ConfigsSuite) TestPatchingInAnLLMFallbackThisDeploymentDoesNotRouteIsRefused() {
	created := s.createConfig(map[string]any{"name": "nora", "llm": "echo/echo-model"})

	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"llm_fallbacks": []string{"xai/grok-typo"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "xai/grok-typo")
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

// Episode cards are off unless a config says otherwise, each mode is read back as it was sent,
// and an update that leaves the setting out keeps it, so a client that does not know it cannot
// turn the cards off by saving. Catches a mode stored as another: record as off or on, which
// the two columns tell apart (store.AgentConfig.WritesCards), on stored without the column the
// previous release reads.
func (s *ConfigsSuite) TestAConfigWritesNoEpisodeCardsUnlessItSaysSo() {
	unnamed := s.createConfig(map[string]any{"name": "support"})
	s.Equal(EpisodeCardsOff, value(unnamed.EpisodeCards))
	s.storedCards(unnamed.Id, false, false)

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"episode_cards": "record"}, &patched))
	s.Equal(EpisodeCardsRecord, value(patched.EpisodeCards))
	s.storedCards(unnamed.Id, false, true)

	var saved AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"name": "support"}, &saved))
	s.Equal(EpisodeCardsRecord, value(saved.EpisodeCards), "a setting left out on an update stays")
	s.storedCards(unnamed.Id, false, true)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"name": "support", "episode_cards": "on"}, &saved))
	s.Equal(EpisodeCardsOn, value(saved.EpisodeCards))
	s.storedCards(unnamed.Id, true, false)

	var read AgentConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+unnamed.Id, nil, &read))
	s.Equal(EpisodeCardsOn, value(read.EpisodeCards))

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch,
		"/v1/agents/configs/"+unnamed.Id, map[string]any{"episode_cards": "off"}, &patched))
	s.Equal(EpisodeCardsOff, value(patched.EpisodeCards))
	s.storedCards(unnamed.Id, false, false)

	created := s.createConfig(map[string]any{"name": "recorder", "episode_cards": "record"})
	s.Equal(EpisodeCardsRecord, value(created.EpisodeCards))
	s.storedCards(created.Id, false, true)
}

// A stored config with both columns on, which no write here makes, reads as on: the column the
// previous release reads decides, so both releases take it the same way.
func (s *ConfigsSuite) TestAConfigWithBothCardColumnsOnReadsAsOn() {
	created := s.createConfig(map[string]any{"name": "support"})
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE agent_configs SET episode_cards = true, episode_cards_record = true WHERE id = ?", created.Id)
	s.Require().NoError(err)

	var read AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+created.Id, nil, &read))
	s.Equal(EpisodeCardsOn, value(read.EpisodeCards))
}

// episode_cards takes off, record and on alone, on create, update and patch: the boolean it was
// is refused (400), so a client written for it learns of the change rather than saving a mode
// it did not mean, and so is any other value, cased otherwise or empty. Catches a value
// accepted that the enum does not name, and stored as some mode.
func (s *ConfigsSuite) TestEpisodeCardsTakesOnlyOffRecordOrOn() {
	created := s.createConfig(map[string]any{"name": "support", "episode_cards": "record"})
	for _, sent := range []any{true, false, "true", "On", "", "recorded", 1} {
		for _, write := range []struct{ method, path string }{
			{http.MethodPost, "/v1/agents/configs"},
			{http.MethodPut, "/v1/agents/configs/" + created.Id},
			{http.MethodPatch, "/v1/agents/configs/" + created.Id},
		} {
			status, body := s.serverClient.call(write.method, write.path, map[string]any{"name": "support-" + s.utils.uuid(), "episode_cards": sent})
			s.Equal(http.StatusBadRequest, status, "%s %v", write.method, sent)
			s.Contains(string(body), `"code":"validation_failed"`, "%s %v", write.method, sent)
		}
	}
	s.storedCards(created.Id, false, true)
}

// storedCards checks the two columns a config's mode is stored in.
func (s *ConfigsSuite) storedCards(id string, on, record bool) {
	s.T().Helper()
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), id)
	s.Require().NoError(err)
	s.Equal(on, stored.EpisodeCards, "episode_cards")
	s.Equal(record, stored.EpisodeCardsRecord, "episode_cards_record")
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
			s.Contains(read, "calls", "every config renders its calls")
			savedCalls, err := json.Marshal(saved.Calls)
			s.Require().NoError(err)
			readCalls, err := json.Marshal(read["calls"])
			s.Require().NoError(err)
			s.JSONEq(string(readCalls), string(savedCalls), "a config saved as read keeps its calls")
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

// The sandbox's egress proxy refuses every address host, so an address among the allowed
// domains would never be reached. A name that only starts like an address is a name.
func (s *ConfigsSuite) TestAnAllowedDomainThatIsAnIPAddressIsRefused() {
	for _, address := range []string{"203.0.113.7", "203.0.113.7:443", "203.0.113.7.", "2001:db8::7", "[2001:db8::7]",
		"[2001:db8::7]:443", "[2001:db8::7].", "::ffff:203.0.113.7", "fe80::1%eth0"} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "researcher", "sandbox": "daytona",
				"sandbox_options": map[string]any{"allowed_domains": []string{"api.example.com", address}}})

		s.Equal(http.StatusBadRequest, status, address)
		s.Contains(failure, fmt.Sprintf("sandbox_options.allowed_domains: %q is not a domain", address), address)
	}

	created := s.createConfig(map[string]any{"name": "researcher", "sandbox": "daytona",
		"sandbox_options": map[string]any{"allowed_domains": []string{"203.0.113.7.example.com", "1-2-3-4.example.com", "example.com."}}})
	s.Equal([]string{"203.0.113.7.example.com", "1-2-3-4.example.com", "example.com."},
		value(value(created.SandboxOptions).AllowedDomains))
	status, failure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"sandbox_options": map[string]any{"allowed_domains": []string{"203.0.113.7"}}})
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `"203.0.113.7" is not a domain`)
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
	"https://hooks acme.com/check":      "must be a public https URL", // a host url.Parse refuses
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

func (s *ConfigsSuite) TestAGuardrailWebhookRefusalDoesNotTellAnUnresolvableNameFromAPrivateOne() {
	refusals := map[string]bool{}
	for _, url := range []string{"https://hooks.acme.invalid/check", "https://localhost/check", "https://10.1.2.3/check"} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
			map[string]any{"name": "support", "guardrail": webhookGuardrail(url)})
		s.Equal(http.StatusBadRequest, status, url)
		refusals[failure] = true
	}

	s.Len(refusals, 1, "every refusal reads the same")
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

// TestAConfigStoredNamingPluginsAndMCPServersReadsWithoutThem: a row written before they were
// refused still reads, in the config and in the list, and shows none of them, so a client that
// saves what it read back is not refused for what it never set.
func (s *ConfigsSuite) TestAConfigStoredNamingPluginsAndMCPServersReadsWithoutThem() {
	config := s.storedWithPlugins()

	status, raw := s.serverClient.call(http.MethodGet, "/v1/agents/configs/"+config.ID, nil)
	s.Require().Equal(http.StatusOK, status)
	var read map[string]any
	s.Require().NoError(json.Unmarshal(raw, &read))
	s.Equal("Be brief.", read["instructions"])
	for _, key := range []string{"plugins", "plugin_events", "mcp_servers"} {
		s.NotContains(read, key)
	}
	status, raw = s.serverClient.call(http.MethodGet, "/v1/agents/configs", nil)
	s.Require().Equal(http.StatusOK, status)
	s.Contains(string(raw), config.ID)
	for _, key := range []string{`"plugins"`, `"plugin_events"`, `"mcp_servers"`} {
		s.NotContains(string(raw), key)
	}

	for _, key := range []string{"id", "created_at", "updated_at", "sync_hash", "duration"} {
		delete(read, key)
	}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+config.ID, read, nil))
}

// storedWithPlugins is a config whose row names a plugin, a plugin event and an MCP server, as
// one saved before they were refused does: written into the columns, as no request can now.
func (s *ConfigsSuite) storedWithPlugins() store.AgentConfig {
	config := store.AgentConfig{CustomerID: s.customerID(), Name: "legacy-" + s.utils.uuid(), Mode: "text",
		Instructions: "Be brief."}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	nameRetired(s.T(), s.store, config.ID)
	return config
}

// nameRetired writes a plugin, a plugin event and an MCP server into the columns of config's
// row, which the model no longer has, as a save before they were refused left them.
func nameRetired(t *testing.T, stored *store.Store, config string) {
	t.Helper()
	result, err := stored.DB().ExecContext(context.Background(), `UPDATE agent_configs
		SET plugins = '["linear"]', plugin_events = '[{"plugin":"linear","event":"issue.created"}]',
		    mcp_servers = '[{"name":"wwsd","url":"https://mcp.wwsd.invalid/mcp"}]'
		WHERE id = ?`, config)
	require.NoError(t, err)
	written, err := result.RowsAffected()
	require.NoError(t, err)
	require.EqualValues(t, 1, written)
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
	s.Equal(`connector binding "inbox" has a tool_patterns entry that is not a tool name or pattern`, failure)
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

func (s *ConfigsSuite) TestABindingWithANullToolsListIsRefused() {
	binding := sessionSlack("inbox")
	binding["tools"] = nil

	status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
		map[string]any{"name": "support", "connectors": []map[string]any{binding}})

	s.Equal(http.StatusBadRequest, status)
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
		"connectors": []map[string]any{sessionGitHubGranting("code", "create_issue"), sessionNessy("customers")},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "customers" binds "nessy"`)
	s.Contains(failure, `and connector binding "code", `+
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

// Every tool gmail.yaml, google_calendar.yaml, linear.yaml and slack.yaml list in
// read_only_tools only reads, so a binding granting only those sits beside nessy (AI-2756).
func (s *ConfigsSuite) TestNessyBesideOnlyTheReadOnlyToolsOfGmailCalendarLinearAndSlackIsAccepted() {
	s.useInternalApp()

	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{
		sessionNessy("customers"),
		sessionGranting("gmail", "mail", "list_drafts", "get_draft", "get_thread", "get_message", "search_threads",
			"list_labels"),
		sessionGranting("google_calendar", "calendar", "list_events", "get_event", "list_calendars",
			"suggest_time", "search_events"),
		sessionGranting("linear", "tracker", "get_attachment", "list_agent_skills", "get_agent_skill",
			"list_comments", "list_cycles", "list_custom_views", "get_document", "list_documents", "get_issue",
			"list_issues", "list_issue_statuses", "get_issue_status", "list_issue_labels", "list_projects",
			"get_project", "list_project_labels", "list_release_pipelines", "list_releases", "get_release",
			"list_release_notes", "get_release_note", "list_milestones", "get_milestone", "get_notifications",
			"list_teams", "get_team", "get_triage_responsibility", "list_templates", "get_template", "list_users",
			"get_user", "get_workspace", "get_diff", "list_diffs", "get_diff_threads",
			"whoami", "list_customers", "list_customer_needs", "list_initiatives", "get_initiative",
			"list_initiative_labels", "get_status_updates"),
		sessionGranting("slack", "chat", "slack_search_public", "slack_search_public_and_private",
			"slack_search_channels", "slack_search_users", "slack_read_channel", "slack_read_thread",
			"slack_read_canvas", "slack_read_user_profile", "slack_read_list", "slack_read_file",
			"slack_list_channel_members", "slack_list_user_channels", "slack_search_emojis"),
	}})

	s.Len(value(created.Connectors), 5)
}

// A tool the four sources leave out of read_only_tools may send data out: one that writes, and
// linear's extract_images, which fetches URLs from markdown the model passes. A binding granting one
// is refused and named; a binding granting only listed tools is not, and the same write tool
// held by require_approval is accepted.
func (s *ConfigsSuite) TestNessyBesideAToolGmailCalendarLinearOrSlackLeaveOutNamesEachBindingGrantingOne() {
	s.useInternalApp()
	trash := sessionGranting("gmail", "mail_trash", "search_threads", "trash_thread")

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{
			sessionNessy("customers"),
			sessionGranting("gmail", "mail_read", "search_threads", "get_thread"),
			trash,
			sessionGranting("gmail", "mail_draft", "create_draft"),
			sessionGranting("google_calendar", "calendar_create", "list_events", "create_event"),
			sessionGranting("google_calendar", "calendar_respond", "respond_to_event"),
			sessionGranting("linear", "linear_save", "get_issue", "save_issue"),
			sessionGranting("linear", "linear_images", "extract_images"),
			sessionGranting("slack", "slack_send", "slack_read_channel", "slack_send_message"),
			sessionGranting("slack", "slack_schedule", "slack_schedule_message"),
		},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector binding "customers" binds "nessy", whose tools read private data, `+
		`and connector binding "mail_trash", connector binding "mail_draft", connector binding "calendar_create", `+
		`connector binding "calendar_respond", connector binding "linear_save", connector binding "linear_images", `+
		`connector binding "slack_send", connector binding "slack_schedule" can send data out with no person's approval`)
	trash["require_approval"] = []string{"trash_thread"}
	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{sessionNessy("customers"), trash}})
	s.Len(value(created.Connectors), 2)
}

// Gmail and Google Calendar have no golden, so this pins the tools their read_only_tools leave out
// (AI-2756): a binding granting every other tool of each is refused, and the 400 names exactly these.
// A tool added to either list drops out of the message and fails the test.
func (s *ConfigsSuite) TestNessyBesideEveryToolGmailAndCalendarLeaveOutNamesExactlyThoseTools() {
	s.useInternalApp()
	gmail := []string{"create_draft", "label_thread", "unlabel_thread", "apply_sensitive_thread_label",
		"trash_thread", "untrash_thread", "mark_thread_spam", "unmark_thread_spam", "label_message",
		"update_message_labels", "unlabel_message", "apply_sensitive_message_label", "trash_message",
		"untrash_message", "mark_message_spam", "unmark_message_spam", "create_label"}
	calendar := []string{"create_event", "update_event", "delete_event", "respond_to_event"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{
			sessionNessy("customers"),
			sessionGranting("gmail", "mail", gmail...),
			sessionGranting("google_calendar", "calendar", calendar...),
		},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `or add to each binding's require_approval its tools that do not only read: `+
		`"mail" `+strings.Join(gmail, ", ")+`; "calendar" `+strings.Join(calendar, ", "))
}

// GET .../tools says which tools only read by the rule a config save checks beside nessy
// (AI-2788): linear.yaml lists list_issues in read_only_tools and leaves out
// search_documentation, which Athena took for a read on 2026-11-05, and save_issue. A binding
// granting by name exactly the tools flagged read_only is accepted beside nessy; one granting a
// tool flagged otherwise as well is refused, naming that tool.
func (s *ConfigsSuite) TestABindingBesideNessyMayGrantExactlyTheToolsFlaggedReadOnly() {
	s.useInternalApp()
	id := s.connectionTo("linear", "")
	s.listTools(id, "list_issues", "search_documentation", "save_issue")

	readOnly := s.readOnlyTools(id)

	s.Equal(map[string]bool{"list_issues": true, "search_documentation": false, "save_issue": false}, readOnly)
	var reads, others []string
	for name, only := range readOnly {
		if only {
			reads = append(reads, name)
		} else {
			others = append(others, name)
		}
	}
	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{
		sessionNessy("customers"), sessionGranting("linear", "tracker", reads...)}})
	s.Len(value(created.Connectors), 2)
	for _, other := range others {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
			"name": "triage", "connectors": []map[string]any{
				sessionNessy("customers"), sessionGranting("linear", "tracker", append(slices.Clone(reads), other)...)}})

		s.Equal(http.StatusBadRequest, status, other)
		s.Contains(failure, `"tracker" `+other)
	}
}

// read_only is read from the connector's latest revision, as a config save beside nessy reads
// it (SensitiveBindingsComplaint), not from the revision the connection pins: a connection
// pinned to revision 1, which leaves search_documentation out of read_only_tools, flags it
// read_only once revision 2 lists it, and a binding granting it sits beside nessy.
func (s *ConfigsSuite) TestReadOnlyIsTheConnectorsLatestRevisionsAsAConfigSaveChecks() {
	s.useInternalApp()
	ctx := context.Background()
	linear, err := s.store.LatestConnectorDefinition(ctx, s.customerID(), "linear")
	s.Require().NoError(err)
	manifest := linear.Manifest
	manifest.ID = "custom_t" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest.Sources = []core.SourceRule{{Kind: "mcp", Endpoint: "mcp", ReadOnlyTools: []string{"list_issues"}}}
	_, err = s.store.CreateConnectorDefinition(ctx, s.customerID(), manifest)
	s.Require().NoError(err)
	id := s.connectionTo(manifest.ID, "")
	manifest.Sources = []core.SourceRule{{Kind: "mcp", Endpoint: "mcp",
		ReadOnlyTools: []string{"list_issues", "search_documentation"}}}
	revised, err := s.store.CreateConnectorDefinition(ctx, s.customerID(), manifest)
	s.Require().NoError(err)
	connection, err := s.store.ConnectorConnection(ctx, s.customerID(), id)
	s.Require().NoError(err)
	s.Require().Equal([]int{1, 2}, []int{connection.DefinitionRevision, revised.Revision})
	s.listTools(id, "list_issues", "search_documentation", "save_issue")

	readOnly := s.readOnlyTools(id)

	s.Equal(map[string]bool{"list_issues": true, "search_documentation": true, "save_issue": false}, readOnly)
	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{
		sessionNessy("customers"), sessionGranting(manifest.ID, "tracker", "list_issues", "search_documentation")}})
	s.Len(value(created.Connectors), 2)
}

// A grant by name does not say which source of a connector a tool comes from, so a tool is
// read_only only when every source lists it in read_only_tools or says outbound: false, and the
// save beside nessy agrees. The first source lists search_documentation too, the second does not:
// it is not read_only and a binding granting it is refused, while list_issues and get_issue are
// listed by both and accepted. The third source says outbound: false and changes neither.
func (s *ConfigsSuite) TestReadOnlyOfAConnectorWithSeveralSourcesNeedsEverySourceToSayIt() {
	s.useInternalApp()
	ctx := context.Background()
	linear, err := s.store.LatestConnectorDefinition(ctx, s.customerID(), "linear")
	s.Require().NoError(err)
	manifest := linear.Manifest
	manifest.ID = "custom_t" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest.Sources = []core.SourceRule{
		{Kind: "mcp", Endpoint: "mcp", ReadOnlyTools: []string{"list_issues", "get_issue", "search_documentation"}},
		{Kind: "mcp", Endpoint: "mcp", ReadOnlyTools: []string{"list_issues", "get_issue"}},
		{Kind: "mcp", Endpoint: "mcp", Outbound: new(false)}}
	_, err = s.store.CreateConnectorDefinition(ctx, s.customerID(), manifest)
	s.Require().NoError(err)
	id := s.connectionTo(manifest.ID, "")
	s.listTools(id, "list_issues", "get_issue", "search_documentation", "save_issue")

	readOnly := s.readOnlyTools(id)

	s.Equal(map[string]bool{"list_issues": true, "get_issue": true, "search_documentation": false, "save_issue": false}, readOnly)
	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{
		sessionNessy("customers"), sessionGranting(manifest.ID, "tracker", "list_issues", "get_issue")}})
	s.Len(value(created.Connectors), 2)
	for _, other := range []string{"search_documentation", "save_issue"} {
		status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
			"name": "triage", "connectors": []map[string]any{
				sessionNessy("customers"), sessionGranting(manifest.ID, "tracker", "list_issues", other)}})

		s.Equal(http.StatusBadRequest, status, other)
		s.Contains(failure, `"tracker" `+other)
	}
}

// A connection whose connector the app can no longer see (nessy, an internal connector, once the app
// loses internal_connectors) still lists its tools, none of them read_only: a config save skips
// a binding with no definition, so this read does not fail where it answered before (AI-2788).
func (s *ConfigsSuite) TestAConnectionWhoseConnectorTheAppCanNoLongerSeeListsItsToolsAsNotReadOnly() {
	s.useInternalApp()
	id := s.connectionTo("nessy", "")
	s.listTools(id, "org_lookup")
	_, err := s.store.DB().NewUpdate().Model(&store.App{ID: s.customerID(), Settings: store.AppSettings{}}).
		Column("settings").WherePK().Exec(context.Background())
	s.Require().NoError(err)

	readOnly := s.readOnlyTools(id)

	s.Equal(map[string]bool{"org_lookup": false}, readOnly)
}

// listTools stores names as the tools a validate of the connection listed.
func (s *ConfigsSuite) listTools(id string, names ...string) {
	tools := make([]store.ConnectorTool, 0, len(names))
	for _, name := range names {
		tools = append(tools, store.ConnectorTool{Name: name, InputSchema: map[string]any{"type": "object"}})
	}
	s.Require().NoError(s.store.SetConnectorConnectionTools(context.Background(), s.customerID(), id, tools,
		"digest", time.Now()))
}

// readOnlyTools is GET .../tools as each tool's name and read_only.
func (s *ConfigsSuite) readOnlyTools(id string) map[string]bool {
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id+"/tools", nil, &tools))
	readOnly := make(map[string]bool, len(tools.Tools))
	for _, tool := range tools.Tools {
		readOnly[tool.Name] = tool.ReadOnly
	}
	return readOnly
}

// Every tool calcom.yaml, calendly.yaml, github.yaml, gong.yaml, hubspot.yaml, linear_app.yaml,
// metabase.yaml, notion.yaml and sentry.yaml list in read_only_tools only reads, so a binding
// granting only those sits beside nessy (AI-2777). linear_app grants a few of linear.yaml's,
// which it lists too (TestToolsetSuite/TestLinearAppListsTheReadOnlyToolsLinearDoes).
func (s *ConfigsSuite) TestNessyBesideOnlyTheReadOnlyToolsOfTheOtherMCPConnectorsIsAccepted() {
	s.useInternalApp()

	created := s.createConfig(map[string]any{"name": "athena", "connectors": []map[string]any{
		sessionNessy("customers"),
		sessionGranting("calcom", "scheduling", "get_app_link", "get_me", "get_event_types",
			"get_event_type", "get_event_type_settings", "get_event_type_history",
			"get_scheduling_config", "get_crm_sync_errors", "get_events", "get_event",
			"get_bookings", "get_booking", "get_booking_attendees", "get_booking_attendee",
			"get_booking_routing_trace", "get_org_team_bookings", "get_org_user_bookings",
			"get_schedules", "get_schedule", "get_default_schedule", "get_availability",
			"get_busy_times", "get_connected_calendars", "get_conferencing_apps", "get_my_teams",
			"get_org_teams", "get_team_memberships", "get_team_membership", "get_org_memberships",
			"get_org_membership", "get_org_attributes", "get_org_attribute",
			"get_attribute_options", "get_user_attributes", "get_user_attribute_history",
			"get_org_routing_forms", "get_org_routing_form_responses", "get_team_routing_forms",
			"get_routing_form", "get_routing_form_history", "describe_api_operation"),
		sessionGranting("calendly", "booking", "event_types-list_event_types",
			"event_types-get_event_type", "event_types-list_event_type_available_times",
			"event_types-list_event_type_availability_schedule",
			"locations-list_user_meeting_locations", "meetings-list_events", "meetings-get_event",
			"meetings-list_event_invitees", "meetings-get_event_invitee",
			"meetings-get_invitee_no_show", "availability-list_user_availability_schedules",
			"availability-get_user_availability_schedule", "availability-list_user_busy_times",
			"routing_forms-list_routing_forms", "routing_forms-get_routing_form",
			"routing_forms-list_routing_form_submissions",
			"routing_forms-get_routing_form_submission", "users-get_current_user",
			"users-get_user", "organizations-get_organization",
			"organizations-list_organization_memberships",
			"organizations-get_organization_membership",
			"organizations-list_organization_invitations", "notetaker-list_recaps",
			"notetaker-get_recap", "notetaker-get_recap_transcript", "list_calendly_skills"),
		sessionGranting("github", "code", "get_me", "get_team_members", "get_teams", "ui_get",
			"get_commit", "get_file_contents", "get_latest_release", "get_release_by_tag",
			"get_tag", "list_branches", "list_commits", "list_releases",
			"list_repository_collaborators", "list_tags", "search_code", "search_commits",
			"search_repositories", "get_label", "issue_read", "list_issue_fields",
			"list_issue_types", "list_issues", "search_issues", "list_pull_requests",
			"pull_request_read", "search_pull_requests", "search_users", "actions_get",
			"actions_list", "get_job_logs"),
		sessionGranting("gong", "calls", "generate_brief"),
		sessionGranting("hubspot", "crm", "get_user_details", "get_organization_details",
			"discover_hubspot_schema", "search_crm_objects", "get_crm_objects",
			"search_properties", "get_properties", "search_owners",
			"get_campaign_attribution_reports", "read_campaign_data", "search_conversations",
			"get_conversation_channel_metadata", "search_intent_signals",
			"get_marketing_email_analytics", "get_content_analytics_report", "get_aeo_metrics",
			"render_asset", "tool_guidance"),
		sessionGranting("metabase", "analytics", "browse_collection", "browse_data",
			"execute_query", "get_content", "get_parameter_values", "glossary", "learn",
			"render_drill_through", "run_saved_question", "search", "visualize_query"),
		sessionGranting("notion", "docs", "notion-download-attachment", "notion-download-skill",
			"notion-fetch", "notion-get-async-task", "notion-get-comments",
			"notion-get-enabled-skills", "notion-get-file-download-urls", "notion-get-groups",
			"notion-get-self", "notion-get-session-status", "notion-get-suggested-edit",
			"notion-get-teams", "notion-get-tool-access", "notion-get-users",
			"notion-get-version-changes", "notion-list-favorite-pages",
			"notion-list-page-backlinks", "notion-list-private-pages", "notion-list-recent-pages",
			"notion-list-session-events", "notion-list-shared-pages", "notion-memory-search",
			"notion-query-data-sources", "notion-query-meeting-notes",
			"notion-query-multiple-data-sources", "notion-query-sessions",
			"notion-read-session-event", "notion-search-agents",
			"notion-search-sessions", "notion-view-version", "notion-wait-session"),
		sessionGranting("sentry", "errors", "find_alert_rules", "find_dashboards",
			"find_dropped_events", "find_dsns", "find_metric_monitors", "find_monitors",
			"find_organizations", "find_projects", "find_releases", "find_teams",
			"find_uptime_monitors", "get_agent_conversation_details",
			"get_ai_conversation_details", "get_alert_options", "get_alert_rule",
			"get_dashboard_details", "get_event_attachment", "get_event_stacktrace",
			"get_issue_activity", "get_issue_breadcrumbs", "get_issue_details",
			"get_issue_tag_values", "get_issue_user_reports", "get_latest_base_snapshot",
			"get_metric_monitor_details", "get_monitor_details", "get_profile",
			"get_profile_details", "get_release_details", "get_replay_details",
			"get_sentry_mcp_info", "get_sentry_resource", "get_snapshot", "get_snapshot_image",
			"get_span_details", "get_trace_details", "get_uptime_monitor_details",
			"search_agent_conversations", "search_ai_conversations", "whoami"),
		sessionGranting("linear_app", "tracker", "get_issue", "list_issues", "get_attachment"),
	}})

	s.Len(value(created.Connectors), 10)
}

// HubSpot, Notion and Sentry have no golden, so this pins every tool their vendors list that
// their read_only_tools leave out (AI-2777): a binding granting all of them is refused, and the
// 400 names exactly these. Cal.com, Calendly, GitHub and Gong have goldens; their bindings grant the
// tools left out for a reason other than a write: a search of public docs or an API catalog, a
// call of any operation, a routing form response, a booking link anyone may open, a workflow
// run, a question the model sends to Gong's AI. A tool added to any list drops out of the message
// and fails the test.
func (s *ConfigsSuite) TestNessyBesideAToolTheOtherMCPConnectorsLeaveOutNamesExactlyThoseTools() {
	s.useInternalApp()
	calcom := []string{"search_docs", "find_api_operation", "call_api_operation",
		"calculate_routing_form_slots"}
	calendly := []string{"scheduling_links-create_single_use_scheduling_link", "shares-create_share"}
	github := []string{"actions_run_trigger", "issue_write"}
	hubspot := []string{"manage_aeo_prompts", "manage_aeo_recommendations", "manage_blog_post",
		"manage_campaign_objects", "manage_crm_objects", "manage_custom_pipelines",
		"manage_custom_properties", "manage_landing_page", "manage_marketing_email",
		"manage_onboarding", "manage_segment", "manage_website_page", "query_crm_data",
		"submit_feedback"}
	gong := []string{"ask_account", "ask_deal"}
	notion := []string{"notion-ai-search", "notion-calendar-cancel-events",
		"notion-calendar-create-events", "notion-calendar-list-calendar-resources",
		"notion-calendar-list-calendars", "notion-calendar-list-contacts",
		"notion-calendar-list-coworkers-events", "notion-calendar-list-events",
		"notion-calendar-update-events", "notion-connector-search", "notion-convert-page-to-skill",
		"notion-create-attachment", "notion-create-comment", "notion-create-database",
		"notion-create-file-upload", "notion-create-folder", "notion-create-pages",
		"notion-create-suggested-edit", "notion-create-view", "notion-delete-comment",
		"notion-delete-databases", "notion-delete-pages", "notion-duplicate-page",
		"notion-move-pages", "notion-resolve-suggested-edit", "notion-restore-pages",
		"notion-restore-version", "notion-search", "notion-send-message-to-session", "notion-set-comment-reaction",
		"notion-set-discussion-resolved", "notion-spawn-session", "notion-stop-session",
		"notion-update-comment", "notion-update-data-source", "notion-update-enabled-skills",
		"notion-update-folder", "notion-update-page", "notion-update-view", "notion-upload-skill"}
	sentry := []string{"add_issue_note", "add_team_to_project", "analyze_issue_with_seer",
		"create_alert_rule", "create_dsn", "create_metric_monitor", "create_monitor",
		"create_project", "create_team", "create_uptime_monitor", "delete_alert_rule",
		"delete_metric_monitor", "delete_monitor_environment", "delete_uptime_monitor", "get_doc",
		"link_issue", "onboarding_status_update", "remove_team_from_project", "search_docs",
		"search_errors", "search_events", "search_issue_events", "search_issues", "search_logs",
		"search_metrics", "search_profiles", "search_replays", "search_traces",
		"submit_feedback_to_sentry_team", "unlink_issue", "update_alert_rule", "update_dsn",
		"update_issue", "update_metric_monitor", "update_project", "update_uptime_monitor"}

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "athena", "connectors": []map[string]any{
			sessionNessy("customers"),
			sessionGranting("calcom", "scheduling", calcom...),
			sessionGranting("calendly", "booking", calendly...),
			sessionGranting("github", "code", github...),
			sessionGranting("gong", "calls", gong...),
			sessionGranting("hubspot", "crm", hubspot...),
			sessionGranting("notion", "docs", notion...),
			sessionGranting("sentry", "errors", sentry...),
		},
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `or add to each binding's require_approval its tools that do not only read: `+
		`"scheduling" `+strings.Join(calcom, ", ")+`; "booking" `+strings.Join(calendly, ", ")+
		`; "code" `+strings.Join(github, ", ")+`; "calls" `+strings.Join(gong, ", ")+`; "crm" `+strings.Join(hubspot, ", ")+
		`; "docs" `+strings.Join(notion, ", ")+`; "errors" `+strings.Join(sentry, ", "))
}

// nessy.yaml's source says outbound: false, so nessy alone, or twice, is accepted.
func (s *ConfigsSuite) TestNessyAloneIsAccepted() {
	s.useInternalApp()

	created := s.createConfig(map[string]any{"name": "athena",
		"connectors": []map[string]any{sessionNessy("customers"), sessionNessy("billing")}})

	s.Len(value(created.Connectors), 2)
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
		map[string]any{"name": "athena", "sandbox": "daytona", "sandbox_options": map[string]any{"network": true}})
	patched, patchFailure := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"connectors": []map[string]any{sessionNessy("customers"), sessionGitHubGranting("code", "create_issue")}})

	s.Equal(http.StatusBadRequest, updated)
	s.Contains(updateFailure, `and the sandbox's network access can send data out`)
	s.Equal(http.StatusBadRequest, patched)
	s.Contains(patchFailure, `and connector binding "code" can send data out`)
	s.Len(value(s.read(created.Id).Connectors), 1)
	s.Nil(s.read(created.Id).Sandbox)
}

// TestAConfigWithoutASensitiveConnectorKeepsItsOutboundTools is the control: the base,
// a103c19ab, answered 201 to the same request (ai1545/author-1.md, «Control probe»).
func (s *ConfigsSuite) TestAConfigWithoutASensitiveConnectorKeepsItsOutboundTools() {
	created := s.createConfig(map[string]any{
		"name": "builder", "sandbox": "daytona", "sandbox_options": map[string]any{"network": true},
		"connectors": []map[string]any{sessionGitHubGranting("code", "create_issue")},
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

// TestAConfigWithoutASensitiveConnectorKeepsTheWriteToolsOfGmailCalendarLinearAndSlack is the
// control: the base, 90167d722, answered 201 to the same request (checked by running this
// test on that commit).
func (s *ConfigsSuite) TestAConfigWithoutASensitiveConnectorKeepsTheWriteToolsOfGmailCalendarLinearAndSlack() {
	created := s.createConfig(map[string]any{"name": "builder", "connectors": []map[string]any{
		sessionGranting("gmail", "mail", "trash_thread", "create_draft"),
		sessionGranting("google_calendar", "calendar", "create_event", "delete_event"),
		sessionGranting("linear", "tracker", "save_issue", "extract_images"),
		sessionGranting("slack", "chat", "slack_send_message", "slack_add_reaction"),
	}})

	s.Len(value(created.Connectors), 4)
}

// TestAConfigWithoutASensitiveConnectorKeepsTheWriteToolsOfTheOtherMCPConnectors is the control:
// the base, f82b5987a, answered 201 to the same request (checked by running this test on that
// commit).
func (s *ConfigsSuite) TestAConfigWithoutASensitiveConnectorKeepsTheWriteToolsOfTheOtherMCPConnectors() {
	created := s.createConfig(map[string]any{"name": "builder", "connectors": []map[string]any{
		sessionGranting("calcom", "scheduling", "create_booking", "search_docs"),
		sessionGranting("calendly", "booking", "shares-create_share"),
		sessionGranting("github", "code", "issue_write"),
		sessionGranting("hubspot", "crm", "manage_crm_objects", "query_crm_data"),
		sessionGranting("linear_app", "tracker", "save_issue"),
		sessionGranting("notion", "docs", "notion-update-page", "notion-connector-search"),
		sessionGranting("sentry", "errors", "update_issue", "search_issues"),
	}})

	s.Len(value(created.Connectors), 7)
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

func callsExample() map[string]any {
	return map[string]any{
		"end_call": map[string]any{"enabled": true, "when": "the caller says goodbye"},
		"press":    map[string]any{"enabled": false},
		"transfer": map[string]any{"enabled": true, "destinations": []any{
			map[string]any{"name": "Support", "number": "+15550001111", "when": "billing questions", "warm": true},
			map[string]any{"name": "Sales", "number": "+442071234567"},
		}},
	}
}

func (s *ConfigsSuite) readCalls(id string) map[string]any {
	var read map[string]any
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/configs/"+id, nil, &read))
	calls, ok := read["calls"].(map[string]any)
	s.Require().True(ok, "calls rendered")
	return calls
}

func (s *ConfigsSuite) TestAConfigWithoutCallsReadsTheDefaults() {
	created := s.createConfig(map[string]any{"name": "support"})
	calls := s.readCalls(created.Id)
	s.Equal(false, calls["end_call"].(map[string]any)["enabled"])
	s.Equal(true, calls["press"].(map[string]any)["enabled"])
	transfer := calls["transfer"].(map[string]any)
	s.Equal(true, transfer["enabled"])
	s.Empty(transfer["destinations"])
}

func (s *ConfigsSuite) TestAConfigRemembersItsCalls() {
	created := s.createConfig(map[string]any{"name": "support", "calls": callsExample()})
	s.Equal(callsExample(), s.readCalls(created.Id))
}

func (s *ConfigsSuite) TestACallsToolLeftOutOfAnUpdateKeepsItsSettings() {
	created := s.createConfig(map[string]any{"name": "support", "calls": callsExample()})

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"calls": map[string]any{"end_call": map[string]any{"enabled": false}}}, nil))
	calls := s.readCalls(created.Id)
	s.Equal(false, calls["end_call"].(map[string]any)["enabled"])
	s.Len(calls["transfer"].(map[string]any)["destinations"], 2, "transfer was not sent, so it stays")

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support", "calls": map[string]any{"transfer": map[string]any{"enabled": true}}}, nil))
	calls = s.readCalls(created.Id)
	s.Empty(calls["transfer"].(map[string]any)["destinations"], "a tool sent replaces its settings whole")

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/configs/"+created.Id,
		map[string]any{"name": "support"}, nil))
	s.Equal(false, s.readCalls(created.Id)["end_call"].(map[string]any)["enabled"], "calls left out keeps everything")
}

func (s *ConfigsSuite) TestCallsAreChecked() {
	destination := func(name, number string) map[string]any {
		return map[string]any{"name": name, "number": number}
	}
	many := []any{}
	for i := range 21 {
		many = append(many, destination(fmt.Sprintf("Desk %d", i), "+15550001111"))
	}
	for name, destinations := range map[string][]any{
		"number without plus":  {destination("Sales", "5550001111")},
		"number too short":     {destination("Sales", "+1555")},
		"duplicate name":       {destination("Sales", "+15550001111"), destination(" sales ", "+15550002222")},
		"empty name":           {destination("", "+15550001111")},
		"blank name":           {destination("  ", "+15550001111")},
		"too many":             many,
		"when over 1000 chars": {map[string]any{"name": "Sales", "number": "+15550001111", "when": strings.Repeat("a", 1001)}},
	} {
		s.Run(name, func() {
			calls := map[string]any{"transfer": map[string]any{"enabled": true, "destinations": destinations}}
			status, _ := s.serverClient.failure(http.MethodPost, "/v1/agents/configs",
				map[string]any{"name": "support-" + s.utils.uuid(), "calls": calls})
			s.Equal(http.StatusBadRequest, status)

			created := s.createConfig(map[string]any{"name": "ok-" + s.utils.uuid()})
			status, _ = s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
				map[string]any{"calls": calls})
			s.Equal(http.StatusBadRequest, status)
		})
	}
}

func (s *ConfigsSuite) TestCallsWhenIsCountedInCharactersNotBytes() {
	long := strings.Repeat("щ", 1000)
	created := s.createConfig(map[string]any{"name": "support", "calls": map[string]any{
		"end_call": map[string]any{"enabled": true, "when": long},
		"transfer": map[string]any{"enabled": true, "destinations": []any{
			map[string]any{"name": "Sales", "number": "+15550001111", "when": long}}},
	}})
	calls := s.readCalls(created.Id)
	s.Equal(long, calls["end_call"].(map[string]any)["when"])

	status, _ := s.serverClient.failure(http.MethodPatch, "/v1/agents/configs/"+created.Id,
		map[string]any{"calls": map[string]any{"end_call": map[string]any{"when": strings.Repeat("a", 1001)}}})
	s.Equal(http.StatusBadRequest, status)
}
