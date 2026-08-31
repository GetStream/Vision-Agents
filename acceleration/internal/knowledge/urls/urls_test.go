//go:build integration

package urls

import (
	"context"
	"errors"
	"os"
	"slices"
	"sort"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// dsnEnvVar is where the tests look for a Postgres to run against.
const dsnEnvVar = "ROUTER_POSTGRES_DSN"

// crawler is a reader with no web behind it: it answers with whatever the test wrote down.
type crawler struct {
	page search.Page
	err  error
}

func (c *crawler) Read(_ context.Context, address string) (search.Page, error) {
	if c.err != nil {
		return search.Page{}, c.err
	}
	page := c.page
	page.URL = address
	return page, nil
}

// base is a knowledge base kept in memory, keyed the way a real one is.
type base struct {
	mu       sync.Mutex
	passages map[string]knowledge.Document
}

func newBase() *base {
	return &base{passages: map[string]knowledge.Document{}}
}

func (b *base) Upsert(_ context.Context, _ string, documents []knowledge.Document) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, document := range documents {
		b.passages[document.ID] = document
	}
	return nil
}

func (b *base) Delete(_ context.Context, _ string, ids []string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, id := range ids {
		delete(b.passages, id)
	}
	return nil
}

// ids is what is currently findable, sorted so assertions read the same way twice.
func (b *base) ids() []string {
	b.mu.Lock()
	defer b.mu.Unlock()

	stored := make([]string, 0, len(b.passages))
	for id := range b.passages {
		stored = append(stored, id)
	}
	sort.Strings(stored)
	return stored
}

type URLsSuite struct {
	suite.Suite
	ctx     context.Context
	store   *store.Store
	reader  *crawler
	base    *base
	service *Service
}

func TestURLsSuite(t *testing.T) {
	suite.Run(t, new(URLsSuite))
}

func (s *URLsSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}

	s.ctx = context.Background()

	opened, err := store.Open(dsn)
	s.Require().NoError(err)
	s.store = opened
	s.Require().NoError(opened.Migrate(s.ctx))
}

func (s *URLsSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *URLsSuite) SetupTest() {
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE knowledge_urls CASCADE")
	s.Require().NoError(err)

	s.reader = &crawler{page: search.Page{
		Title: "Pricing",
		Text:  "# Pricing\n\nA call costs a penny.\n\n# Support\n\nWe answer within a day.\n",
	}}
	s.base = newBase()

	// A small chunk so a page cuts into several passages without needing a long fixture,
	// which is what the tests about orphans and removal depend on.
	service, err := New(Options{
		Store: s.store, Reader: s.reader, Writer: s.base, ChunkSize: 200,
	})
	s.Require().NoError(err)
	s.service = service
}

func (s *URLsSuite) TestAddingAPageIndexesItAndStampsWhenItWasRead() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)

	s.Equal(store.KnowledgeURLIndexed, page.State)
	s.Equal("Pricing", page.Title)
	s.Equal(2, page.Passages, "the page is cut at its headings")
	s.Require().NotNil(page.LastIndexedAt)
	s.Empty(page.Error)

	s.Equal([]string{
		"https://example.com/pricing#0",
		"https://example.com/pricing#1",
	}, s.base.ids(), "passages are keyed by the url they came from")
}

func (s *URLsSuite) TestAPageThatCouldNotBeReadIsKeptWithTheReason() {
	// The caller asked for this url to be part of the knowledge base. Telling them why it
	// is not, on a row they can retry, is more use than refusing and forgetting.
	s.reader.err = errors.New("CRAWL_NOT_FOUND")

	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/gone")
	s.Require().NoError(err)

	s.Equal(store.KnowledgeURLFailed, page.State)
	s.Contains(page.Error, "CRAWL_NOT_FOUND")
	s.Nil(page.LastIndexedAt, "nothing was ever read, which is not the same as broken since")
	s.Zero(page.Passages)
	s.Empty(s.base.ids(), "a page that could not be read writes nothing")
}

func (s *URLsSuite) TestRemovingAPageTakesItsPassagesWithIt() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)
	s.Require().NotEmpty(s.base.ids())

	s.Require().NoError(s.service.Remove(s.ctx, "acme", page.ID))

	s.Empty(s.base.ids(),
		"an agent answering out of a page nobody subscribes to is worse than one that cannot")
	_, err = s.service.Get(s.ctx, "acme", page.ID)
	s.Error(err)
}

func (s *URLsSuite) TestAPageThatGotShorterLeavesNoOrphans() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)
	s.Require().Equal(2, page.Passages)

	s.reader.page.Text = "# Pricing\n\nA call costs tuppence.\n"

	reindexed, err := s.service.Reindex(s.ctx, "acme", page.ID)
	s.Require().NoError(err)

	s.Equal(1, reindexed.Passages)
	s.Equal([]string{"https://example.com/pricing#0"}, s.base.ids(),
		"the old tail is no longer findable")
}

func (s *URLsSuite) TestReindexingRecordsWhenThePageWasLastRead() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)
	first := *page.LastIndexedAt

	reindexed, err := s.service.Reindex(s.ctx, "acme", page.ID)
	s.Require().NoError(err)

	s.Require().NotNil(reindexed.LastIndexedAt)
	s.False(reindexed.LastIndexedAt.Before(first))
}

func (s *URLsSuite) TestAPageThatBrokeKeepsTheDateItLastWorked() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)
	worked := *page.LastIndexedAt

	s.reader.err = errors.New("SOURCE_NOT_AVAILABLE")
	broken, err := s.service.Reindex(s.ctx, "acme", page.ID)
	s.Require().NoError(err)

	s.Equal(store.KnowledgeURLFailed, broken.State)
	s.Require().NotNil(broken.LastIndexedAt)
	s.Equal(worked.UTC(), broken.LastIndexedAt.UTC(),
		"when it last worked is what says how stale the answers now are")
}

func (s *URLsSuite) TestListingIsScopedToOneKnowledgeBase() {
	_, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)
	_, err = s.service.Add(s.ctx, "acme", "handbook", "https://example.com/leave")
	s.Require().NoError(err)

	listed, err := s.service.List(s.ctx, "acme", "docs")
	s.Require().NoError(err)

	s.Require().Len(listed, 1)
	s.Equal("https://example.com/pricing", listed[0].URL)

	all, err := s.service.List(s.ctx, "acme", "")
	s.Require().NoError(err)
	s.Len(all, 2)
}

func (s *URLsSuite) TestAnotherCustomersPageIsNotThereToRead() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)

	_, err = s.service.Get(s.ctx, "globex", page.ID)

	s.Error(err)
}

func (s *URLsSuite) TestSomethingThatIsNotAFetchablePageIsRefused() {
	for _, address := range []string{"", "  ", "not a url at all", "mailto:sales@example.com", "file:///etc/passwd"} {
		_, err := s.service.Add(s.ctx, "acme", "docs", address)
		s.Errorf(err, "%q should not be stored as a page to crawl", address)
	}

	listed, err := s.service.List(s.ctx, "acme", "docs")
	s.Require().NoError(err)
	s.Empty(listed)
}

func (s *URLsSuite) TestKnowledgeIsNeverSharedSoANamespaceIsRequired() {
	_, err := s.service.Add(s.ctx, "acme", "  ", "https://example.com/pricing")

	s.ErrorContains(err, "namespace")
}

func (s *URLsSuite) TestAddingTheSamePageTwiceIsRefusedRatherThanDuplicated() {
	// Both copies would write the same passage ids, so the second row would be a
	// subscription that removing the first one silently breaks.
	_, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)

	_, err = s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")

	s.Error(err)
}

func (s *URLsSuite) TestPassageIDsCoverExactlyWhatWasWritten() {
	page, err := s.service.Add(s.ctx, "acme", "docs", "https://example.com/pricing")
	s.Require().NoError(err)

	written := s.base.ids()
	for _, id := range passageIDs(page.URL, 0, page.Passages) {
		s.Truef(slices.Contains(written, id), "%s was counted but never written", id)
	}
	s.Len(written, page.Passages)
}
