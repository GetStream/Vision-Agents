//go:build integration

package api

import (
	"net/http"
	"testing"
)

type RouterConfigsSuite struct {
	RouterSuite
}

func TestRouterConfigsSuite(t *testing.T) {
	runSuite(t, new(RouterConfigsSuite))
}

// SetupTest gives every test an app of its own, because a config's name has to be free.
func (s *RouterConfigsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *RouterConfigsSuite) TestARouterConfigSurvivesBeingStoredAndReadBack() {
	created := s.createConfig(map[string]any{
		"name": "healthcare",
		"stt": map[string]any{
			"target": "en-low-latency", "providers": []string{"stub/stub-model", "en-low-latency"},
			"diarize": true, "keyterms": []string{"perioperative"},
		},
		"tts": map[string]any{"target": "en-low-latency", "voice": "aurora", "speed": 1.1},
		"llm": map[string]any{"providers": []string{"stub/stub-model", "llm-flow"}, "temperature": 0.2},
		"search": map[string]any{
			"providers": []string{"stub/stub-model", "en-low-latency"},
			"depth":     "standard", "include_domains": []string{"nice.org.uk"},
		},
	})
	s.Require().NotEmpty(created.Id)

	var read RouterConfig
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/router/configs/"+created.Id, nil, &read))

	s.Require().NotNil(read.Stt)
	s.Equal("en-low-latency", value(read.Stt.Target))
	s.Equal([]string{"stub/stub-model", "en-low-latency"}, value(read.Stt.Providers),
		"a priority list is the order it was written in, which is the point of writing one")
	s.Equal([]string{"perioperative"}, value(read.Stt.Keyterms))
	s.Require().NotNil(read.Tts)
	s.InDelta(1.1, value(read.Tts.Speed), 0.001)
	s.Require().NotNil(read.Llm)
	s.InDelta(0.2, value(read.Llm.Temperature), 0.001)
	s.Equal([]string{"stub/stub-model", "llm-flow"}, value(read.Llm.Providers))
	s.Require().NotNil(read.Search)
	s.Equal([]string{"nice.org.uk"}, value(read.Search.IncludeDomains))
	s.Equal([]string{"stub/stub-model", "en-low-latency"}, value(read.Search.Providers))
}

func (s *RouterConfigsSuite) TestAConfigFallingBackToASearchNobodyOffersIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/router/configs",
		map[string]any{
			"name":   "clinic",
			"search": map[string]any{"providers": []string{"stub/stub-model", "altavista"}},
		})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "altavista")
}

func (s *RouterConfigsSuite) TestARouterConfigIsFoundByNameAsWellAsById() {
	// Naming a config is what a caller writes in their own code, so the id they never saw
	// cannot be the only way back to it.
	s.createConfig(map[string]any{
		"name": "clinic", "stt": map[string]any{"target": "en-low-latency"},
	})

	s.Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/stt/recordings",
		map[string]any{
			"config_id": "clinic",
			"source":    map[string]any{"url": "https://example.test/call.mp3"},
		}, nil))
}

func (s *RouterConfigsSuite) TestUpdatingARouterConfigReplacesWhatItWas() {
	created := s.createConfig(map[string]any{
		"name": "clinic", "stt": map[string]any{"target": "en-low-latency", "diarize": true},
	})

	var updated RouterConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/router/configs/"+created.Id,
		map[string]any{"name": "clinic", "stt": map[string]any{"target": "llm-flow"}}, &updated))

	s.Equal(created.Id, updated.Id, "an update keeps the id callers already hold")
	s.Equal("llm-flow", value(updated.Stt.Target))
	s.Nil(updated.Stt.Diarize, "a field left out of a replacement is gone from it")
}

func (s *RouterConfigsSuite) TestAConfigNamingAVoiceThisDeploymentHasNeverHeardOfIsRefused() {
	// Storing it would leave a config that fails every call made under it, which is worth
	// hearing about while it is being written rather than once a socket is open.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/router/configs",
		map[string]any{"name": "clinic", "tts": map[string]any{"providers": []string{"vocalizer"}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "vocalizer")
}

func (s *RouterConfigsSuite) TestAConfigWithOverwritesForAVoiceNobodyOffersIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/router/configs",
		map[string]any{"name": "clinic", "tts": map[string]any{
			"overwrites": map[string]any{"vocalizer": map[string]any{"voice_id": "v-1"}},
		}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "no voice for")
}

func (s *RouterConfigsSuite) TestAConfigAskingAVoiceForARetentionNothingCanBeComparedAgainstIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/router/configs",
		map[string]any{"name": "clinic", "tts": map[string]any{
			"data_policy": map[string]any{"retention": "ages"},
		}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "ages")
}

func (s *RouterConfigsSuite) TestAConfigHoldingASystemPromptForAConversationIsRefused() {
	// The agent that holds the conversation has instructions of its own and sends them
	// when it opens the session. A config that also carried some would overwrite them
	// from somewhere nobody thought to look.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/router/configs",
		map[string]any{"name": "clinic", "sts": map[string]any{
			"target": "sts-fast", "instructions": "Be brief.",
		}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "instructions")
}

func (s *RouterConfigsSuite) TestARouterConfigNobodyHasIsRefusedRatherThanIgnored() {
	// A caller that named a config meant it: transcribing at whatever the fallback happens
	// to be is not what they asked for.
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/stt/recordings",
		map[string]any{
			"config_id": "nope",
			"source":    map[string]any{"url": "https://example.test/call.mp3"},
		})

	s.Equal(http.StatusNotFound, status)
	s.Contains(failure, errUnknownRouterConfig.Message)
}

func (s *RouterConfigsSuite) TestAnotherAppsRouterConfigIsNotFound() {
	created := s.createConfig(map[string]any{
		"name": "clinic", "stt": map[string]any{"target": "en-low-latency"},
	})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/router/configs/"+created.Id, nil, nil)
	})
}

func (s *RouterConfigsSuite) TestOnlyTheAppsOwnBackendMayStoreARouterConfig() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/router/configs",
			map[string]any{"name": "config-" + s.utils.uuid()}, nil)
	})
}

// createConfig stores a router config the router must accept.
func (s *RouterConfigsSuite) createConfig(body map[string]any) RouterConfig {
	var created RouterConfig
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/router/configs", body, &created))
	return created
}
