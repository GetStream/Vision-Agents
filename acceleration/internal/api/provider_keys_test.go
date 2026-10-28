//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The keys the suite's customers keep for their own turbopuffer accounts.
const (
	turbopufferKey      = "tpuf_customer_0123456789abcdef"
	turbopufferRotation = "tpuf_customer_fedcba9876543210"
)

type ProviderKeysSuite struct {
	RouterSuite
}

func TestProviderKeysSuite(t *testing.T) {
	runSuite(t, new(ProviderKeysSuite))
}

// SetupTest gives every test an app of its own, because the keys listed are everything one
// customer keeps.
func (s *ProviderKeysSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// put keeps a key, and answers what the router said of it.
func (s *ProviderKeysSuite) put(body map[string]any) ProviderKey {
	var kept ProviderKey
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/provider-keys", body, &kept))
	return kept
}

// listed is every key the app keeps, as the API lists them.
func (s *ProviderKeysSuite) listed() []ProviderKey {
	var page ProviderKeyPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/provider-keys", nil, &page))
	return page.Items
}

// refused is the code and message of a failure.
func (s *ProviderKeysSuite) refused(method, path string, body any) (int, ErrorDetail) {
	status, payload := s.serverClient.call(method, path, body)
	var answered ErrorResponse
	s.Require().NoError(json.Unmarshal(payload, &answered), string(payload))
	return status, answered.Error
}

func (s *ProviderKeysSuite) TestAKeyIsKeptSealedAndNeverReturnedOnlyItsLastFourCharacters() {
	kept := s.put(map[string]any{"provider": "turbopuffer", "api_key": turbopufferKey})

	status, listing := s.serverClient.call(http.MethodGet, "/v1/agents/provider-keys", nil)

	s.Equal("turbopuffer", kept.Provider)
	s.Equal("us", kept.Region, "a key that names no region is the us's")
	s.Equal("cdef", kept.KeyHint)
	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(listing), turbopufferKey)
	s.Contains(string(listing), `"key_hint":"cdef"`)
	stored, err := s.store.ProviderKeys(context.Background(), s.customerID())
	s.Require().NoError(err)
	s.Require().Len(stored, 1)
	s.NotContains(string(stored[0].APIKeySealed), turbopufferKey, "the key is sealed at rest")
}

func (s *ProviderKeysSuite) TestAKeyIsKeptForEachRegionApartAndDeletedInItsOwn() {
	s.put(map[string]any{"provider": "turbopuffer", "region": "us", "api_key": turbopufferKey})
	s.put(map[string]any{"provider": "turbopuffer", "region": "eu", "api_key": turbopufferRotation})
	var regions []string
	for _, key := range s.listed() {
		regions = append(regions, key.Region)
	}
	s.Equal([]string{"eu", "us"}, regions)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/provider-keys/turbopuffer?region=eu", nil, nil))

	left := s.listed()
	s.Require().Len(left, 1)
	s.Equal("us", left[0].Region)
}

func (s *ProviderKeysSuite) TestAKeyPutAgainReplacesTheOneKeptForItsProviderAndRegion() {
	first := s.put(map[string]any{"provider": "daytona", "api_key": turbopufferKey})

	second := s.put(map[string]any{"provider": "daytona", "api_key": turbopufferRotation})

	listed := s.listed()
	s.Require().Len(listed, 1)
	s.Equal("3210", listed[0].KeyHint)
	s.Equal(first.CreatedAt.UTC(), second.CreatedAt.UTC(), "the provider's key was first put then")
	s.False(second.UpdatedAt.Before(first.UpdatedAt))
}

func (s *ProviderKeysSuite) TestAProviderWhoseWorkIsNeverRoutedOnACustomersKeyIsRefused() {
	status, failure := s.refused(http.MethodPut, "/v1/agents/provider-keys", map[string]any{
		"provider": "stub", "api_key": turbopufferKey,
	})

	s.Equal(http.StatusBadRequest, status)
	s.Equal("provider_takes_no_key", failure.Code)
	s.Contains(failure.Message, "daytona, turbopuffer")
	s.Empty(s.listed())
}

func (s *ProviderKeysSuite) TestAKeyNoVendorIssuesIsRefused() {
	status, failure := s.refused(http.MethodPut, "/v1/agents/provider-keys", map[string]any{
		"provider": "turbopuffer", "api_key": "tpuf_one,tpuf_two",
	})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure.Message, "no space, comma or control character")
	s.NotContains(failure.Message, "tpuf_one")
}

func (s *ProviderKeysSuite) TestARegionOfAnotherFormIsRefused() {
	status, failure := s.refused(http.MethodPut, "/v1/agents/provider-keys", map[string]any{
		"provider": "turbopuffer", "region": "EU", "api_key": turbopufferKey,
	})

	s.Equal(http.StatusBadRequest, status)
	s.Equal("validation_failed", failure.Code)
}

func (s *ProviderKeysSuite) TestDeletingAKeyThatIsNotKeptIsNotFound() {
	status, failure := s.refused(http.MethodDelete, "/v1/agents/provider-keys/turbopuffer", nil)

	s.Equal(http.StatusNotFound, status)
	s.Equal("provider_key_not_found", failure.Code)
}

func (s *ProviderKeysSuite) TestOnlyTheAppsOwnBackendMayPutListOrDeleteAKey() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, "/v1/agents/provider-keys", map[string]any{"provider": "turbopuffer", "api_key": turbopufferKey}, nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/provider-keys", nil, nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		status := as.do(http.MethodDelete, "/v1/agents/provider-keys/turbopuffer", nil, nil)
		if status == http.StatusNoContent {
			s.put(map[string]any{"provider": "turbopuffer", "api_key": turbopufferKey})
		}
		return status
	})
}

func (s *ProviderKeysSuite) TestAnotherAppNeitherSeesNorDeletesTheKey() {
	s.put(map[string]any{"provider": "turbopuffer", "api_key": turbopufferKey})
	stranger := s.data.backendOfAnotherApp()

	var page ProviderKeyPage
	s.Require().Equal(http.StatusOK, stranger.do(http.MethodGet, "/v1/agents/provider-keys", nil, &page))
	s.Empty(page.Items)
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/provider-keys/turbopuffer", nil, nil)
	})
	s.Len(s.listed(), 1)
}

func (s *ProviderKeysSuite) TestPuttingReplacingAndDeletingAKeyIsAuditedWithItsLastFourCharactersAtMost() {
	s.put(map[string]any{"provider": "turbopuffer", "api_key": turbopufferKey})
	s.put(map[string]any{"provider": "turbopuffer", "api_key": turbopufferRotation})
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/provider-keys/turbopuffer", nil, nil))

	entries, err := s.store.AuditEntries(context.Background(), s.customerID(), store.AuditEntryFilter{ResourceType: store.AuditProviderKey})

	s.Require().NoError(err)
	var actions []string
	for _, entry := range entries {
		actions = append(actions, entry.Action)
		s.Equal("turbopuffer/us", entry.ResourceID)
		changes, err := json.Marshal(entry.Changes)
		s.Require().NoError(err)
		s.NotContains(string(changes), turbopufferKey)
		s.NotContains(string(changes), turbopufferRotation)
	}
	slices.Sort(actions)
	s.Equal([]string{store.AuditCreated, store.AuditDeleted, store.AuditUpdated}, actions)
	replaced := entries[slices.IndexFunc(entries, func(entry store.AuditEntry) bool { return entry.Action == store.AuditUpdated })]
	s.Require().Len(replaced.Changes, 1)
	s.Equal(store.AuditChange{Field: "api_key", Before: "****cdef", After: "****3210"}, replaced.Changes[0])
}
