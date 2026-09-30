//go:build integration

package api

import (
	"net/http"
	"testing"
	"time"
)

type KnowledgeUrlsSuite struct {
	RouterSuite
}

func TestKnowledgeUrlsSuite(t *testing.T) {
	runSuite(t, new(KnowledgeUrlsSuite))
}

// SetupTest gives every test an app of its own, because a list of pages is everything one
// knowledge base subscribed to.
func (s *KnowledgeUrlsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *KnowledgeUrlsSuite) TestAKnowledgeUrlIsReadStoredAndListed() {
	created := s.subscribe(map[string]any{"namespace": "docs", "url": "https://example.com/pricing"})
	s.Equal(KnowledgeUrlStatePending, created.State, "the page is read after the request, not during it")

	read := s.indexed(created.Id, nil)
	s.Equal(KnowledgeUrlStateIndexed, read.State)
	s.Equal(1, read.Passages)
	s.Equal("Pricing", value(read.Title))

	var listed []KnowledgeUrl
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/knowledge/urls?namespace=docs", nil, &listed))
	s.Require().Len(listed, 1)
	s.Equal("https://example.com/pricing", listed[0].Url)
}

func (s *KnowledgeUrlsSuite) TestAPageIsNamedByWhatSubscribedToItRatherThanWhatItCallsItself() {
	created := s.subscribe(map[string]any{
		"namespace": "docs", "url": "https://example.com/pricing",
		"title": "What a call costs", "description": "The page sales points at.",
	})
	s.Equal("What a call costs", value(created.Title), "the crawler called it Pricing")
	s.Equal("The page sales points at.", value(created.Description))

	// The same declaration applied again is a re-read, so a caller with a file of pages
	// does not have to work out which of them the base already has.
	again := s.subscribe(map[string]any{
		"namespace": "docs", "url": "https://example.com/pricing", "title": "What a call costs",
	})
	s.Equal(created.Id, again.Id)
	s.Nil(again.Description)
}

func (s *KnowledgeUrlsSuite) TestADeletedKnowledgeUrlIsNoLongerSubscribedTo() {
	created := s.subscribe(map[string]any{"namespace": "docs", "url": "https://example.com/pricing"})

	s.Require().Equal(http.StatusNoContent,
		s.serverClient.do(http.MethodDelete, "/v1/agents/knowledge/urls/"+created.Id, nil, nil))
	s.Equal(http.StatusNotFound,
		s.serverClient.do(http.MethodGet, "/v1/agents/knowledge/urls/"+created.Id, nil, nil))

	// The url is free again, which is what makes removing one usable rather than final.
	s.subscribe(map[string]any{"namespace": "docs", "url": "https://example.com/pricing"})
}

func (s *KnowledgeUrlsSuite) TestReadingAPageAgainMovesWhenItWasLastIndexed() {
	created := s.subscribe(map[string]any{"namespace": "docs", "url": "https://example.com/pricing"})
	first := s.indexed(created.Id, nil)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		"/v1/agents/knowledge/urls/"+created.Id+"/index", nil, nil))

	s.Equal(created.Id, s.indexed(created.Id, first.LastIndexedAt).Id)
}

func (s *KnowledgeUrlsSuite) TestSomethingThatIsNotAFetchablePageIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/knowledge/urls",
		map[string]any{"namespace": "docs", "url": "mailto:sales@example.com"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "mailto:sales@example.com")
}

func (s *KnowledgeUrlsSuite) TestAnotherAppsKnowledgeUrlIsNotFound() {
	created := s.subscribe(map[string]any{"namespace": "docs", "url": "https://example.com/pricing"})

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/knowledge/urls/"+created.Id, nil, nil)
	})
}

func (s *KnowledgeUrlsSuite) TestOnlyTheAppsOwnBackendMaySubscribeToAPage() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/knowledge/urls",
			map[string]any{"namespace": "docs", "url": "https://example.com/" + s.utils.uuid()}, nil)
	})
}

// subscribe points the knowledge base at a page.
func (s *KnowledgeUrlsSuite) subscribe(body map[string]any) KnowledgeUrl {
	var created KnowledgeUrl
	s.Require().Equal(http.StatusCreated,
		s.serverClient.do(http.MethodPost, "/v1/agents/knowledge/urls", body, &created))
	return created
}

// indexed waits for the worker to have read a page, and returns it as the API then
// describes it. When after is set, the read has to be a later one than that.
func (s *KnowledgeUrlsSuite) indexed(id string, after *time.Time) KnowledgeUrl {
	var page KnowledgeUrl
	s.Require().Eventually(func() bool {
		page = KnowledgeUrl{}
		if s.serverClient.do(http.MethodGet, "/v1/agents/knowledge/urls/"+id, nil, &page) != http.StatusOK {
			return false
		}
		return page.LastIndexedAt != nil && (after == nil || page.LastIndexedAt.After(*after))
	}, settleFor, 10*time.Millisecond, "the page was never read")
	return page
}
