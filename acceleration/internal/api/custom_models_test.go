//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
)

type CustomModelsSuite struct {
	RouterSuite
}

func TestCustomModelsSuite(t *testing.T) {
	runSuite(t, new(CustomModelsSuite))
}

// SetupTest gives every test an app of its own, because the models listed are everything
// one customer brought.
func (s *CustomModelsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *CustomModelsSuite) TestAModelIsStoredAndReadBackWithoutItsKey() {
	created := s.createModel(map[string]any{
		"name": "support-qwen", "base_url": "https://8.8.8.8/v1", "model": "Qwen/Qwen3.8-27B",
		"api_key": "sk-secret", "context_window": 131072, "per_million_input_tokens": 0.2,
		"trains_on_data": "no", "retention": "none",
	})

	var read CustomModel
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/models/"+created.Id, nil, &read))
	s.Equal("custom/support-qwen", read.Target)
	s.Equal("Qwen/Qwen3.8-27B", read.Model)
	s.True(read.HasApiKey)
	s.EqualValues(131072, read.ContextWindow)
	s.Equal("no", value(read.TrainsOnData))

	stored, err := s.store.CustomModel(context.Background(), s.customerID(), created.Id)
	s.Require().NoError(err)
	s.NotContains(string(stored.APIKeySealed), "sk-secret", "the key is sealed at rest")
}

func (s *CustomModelsSuite) TestModelsAreListedNewestFirstAPageAtATime() {
	first := s.createModel(map[string]any{"name": "first", "base_url": "https://8.8.8.8/v1", "model": "a"})
	second := s.createModel(map[string]any{"name": "second", "base_url": "https://8.8.8.8/v1", "model": "b"})

	var page CustomModelPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/models?limit=1", nil, &page))
	s.Require().Len(page.Items, 1)
	s.Equal(second.Id, page.Items[0].Id)
	s.Require().True(page.HasMore)

	var next CustomModelPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/models?limit=1&cursor="+value(page.NextCursor), nil, &next))
	s.Require().Len(next.Items, 1)
	s.Equal(first.Id, next.Items[0].Id)
	s.False(next.HasMore)
}

func (s *CustomModelsSuite) TestAnUpdateWithoutAKeyKeepsTheStoredOne() {
	created := s.createModel(map[string]any{
		"name": "kept", "base_url": "https://8.8.8.8/v1", "model": "a", "api_key": "sk-secret",
	})

	var updated CustomModel
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/models/"+created.Id,
		map[string]any{"name": "renamed", "base_url": "https://8.8.8.8/v1", "model": "b"}, &updated))

	s.Equal("custom/renamed", updated.Target)
	s.True(updated.HasApiKey)
}

func (s *CustomModelsSuite) TestAnEmptyKeyRemovesTheStoredOne() {
	created := s.createModel(map[string]any{
		"name": "cleared", "base_url": "https://8.8.8.8/v1", "model": "a", "api_key": "sk-secret",
	})

	var updated CustomModel
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/models/"+created.Id,
		map[string]any{"name": "cleared", "base_url": "https://8.8.8.8/v1", "model": "a", "api_key": ""}, &updated))

	s.False(updated.HasApiKey)
}

func (s *CustomModelsSuite) TestAPrivateEndpointIsRefusedOnASharedRouter() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/models", map[string]any{
		"name": "internal", "base_url": "https://10.0.0.5/v1", "model": "a",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "public https URL")
}

func (s *CustomModelsSuite) TestHalfADataPolicyIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/models", map[string]any{
		"name": "half", "base_url": "https://8.8.8.8/v1", "model": "a", "trains_on_data": "no",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "retention")
}

func (s *CustomModelsSuite) TestASecondModelOfTheSameNameIsAConflict() {
	s.createModel(map[string]any{"name": "twice", "base_url": "https://8.8.8.8/v1", "model": "a"})

	s.Equal(http.StatusConflict, s.serverClient.do(http.MethodPost, "/v1/agents/models",
		map[string]any{"name": "twice", "base_url": "https://8.8.8.8/v1", "model": "b"}, nil))
}

func (s *CustomModelsSuite) TestADeletedModelIsGone() {
	created := s.createModel(map[string]any{"name": "gone", "base_url": "https://8.8.8.8/v1", "model": "a"})

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/models/"+created.Id, nil, nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodGet, "/v1/agents/models/"+created.Id, nil, nil))
}

func (s *CustomModelsSuite) TestOnlyTheAppsOwnBackendMayAddAModel() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/models", map[string]any{
			"name": "posture-" + s.utils.uuid(), "base_url": "https://8.8.8.8/v1", "model": "a",
		}, nil)
	})
}

func (s *CustomModelsSuite) TestAnotherAppsModelIsNotFound() {
	created := s.createModel(map[string]any{"name": "mine", "base_url": "https://8.8.8.8/v1", "model": "a"})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/models/"+created.Id, nil, nil)
	})
}

func (s *CustomModelsSuite) TestARouterConfigMayNameAModelOfTheCustomersOwn() {
	s.Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/router/configs", map[string]any{
		"name": "own-" + s.utils.uuid(), "llm": map[string]any{"providers": []string{"custom/support-qwen"}},
	}, nil))
}

func (s *CustomModelsSuite) createModel(body map[string]any) CustomModel {
	var created CustomModel
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/models", body, &created))
	return created
}
