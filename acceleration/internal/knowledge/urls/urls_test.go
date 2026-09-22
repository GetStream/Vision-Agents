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
	"time"

	"github.com/hibiken/asynq"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// dsnEnvVar is where the tests look for a Postgres to run against.
const dsnEnvVar = "ROUTER_POSTGRES_DSN"

// redisEnvVar is where the tests look for a Redis to queue the reads on.
const redisEnvVar = "ROUTER_REDIS_ADDR"

// redisDB keeps the tests' queue apart from a router running against the same Redis,
// which would otherwise take the reads for itself.
const redisDB = 13

// crawler is a reader with no web behind it: it answers with whatever the test wrote down.
// failures is how many reads fail before one works, which is what a crawl that failed for
// a reason that passed looks like.
type crawler struct {
	mu       sync.Mutex
	page     search.Page
	err      error
	failures int
}

func (c *crawler) Read(_ context.Context, address string) (search.Page, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.failures > 0 {
		c.failures--
		return search.Page{}, errors.New("CRAWL_TIMEOUT")
	}
	if c.err != nil {
		return search.Page{}, c.err
	}
	page := c.page
	page.URL = address
	return page, nil
}

func (c *crawler) serve(text string, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.page.Text = text
	c.err = err
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
	redis   asynq.RedisClientOpt
	reader  *crawler
	base    *base
	service *Service
}

func TestURLsSuite(t *testing.T) {
	suite.Run(t, new(URLsSuite))
}

func (s *URLsSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	address := os.Getenv(redisEnvVar)
	if dsn == "" || address == "" {
		s.T().Skipf("%s and %s must be set", dsnEnvVar, redisEnvVar)
	}

	s.ctx = context.Background()
	s.redis = asynq.RedisClientOpt{Addr: address, DB: redisDB}

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

	// Whatever a test before this one left queued would be read into this one's base.
	inspector := asynq.NewInspector(s.redis)
	_ = inspector.DeleteQueue(queue, true)
	s.Require().NoError(inspector.Close())

	// A small chunk so a page cuts into several passages without needing a long fixture,
	// which is what the tests about orphans and removal depend on.
	service, err := New(Options{
		Store: s.store, Redis: s.redis, Reader: s.reader, Writer: s.base,
		ChunkSize: 200, CheckInterval: 10 * time.Millisecond,
	})
	s.Require().NoError(err)
	s.Require().NoError(service.Start())
	s.service = service
}

func (s *URLsSuite) TearDownTest() {
	s.Require().NoError(s.service.Close())
}

// drained waits for the worker to be done with a page, retries and all. The task is gone
// from the queue once nothing more will be tried.
func (s *URLsSuite) drained(page store.KnowledgeURL) {
	inspector := asynq.NewInspector(s.redis)
	defer inspector.Close()
	s.Require().Eventually(func() bool {
		_, err := inspector.GetTaskInfo(queue, page.ID)
		return errors.Is(err, asynq.ErrTaskNotFound) || errors.Is(err, asynq.ErrQueueNotFound)
	}, 10*time.Second, 10*time.Millisecond)
}

// settled waits for the worker to be done with a page and returns it as it then is.
func (s *URLsSuite) settled(page store.KnowledgeURL) store.KnowledgeURL {
	s.drained(page)
	current, err := s.service.Get(s.ctx, page.CustomerID, page.ID)
	s.Require().NoError(err)
	return current
}

// added subscribes to a page and waits for its first read.
func (s *URLsSuite) added(wanted Subscription) store.KnowledgeURL {
	page, err := s.service.Add(s.ctx, "acme", wanted)
	s.Require().NoError(err)
	return s.settled(page)
}

// reindexed reads a page again and waits for it.
func (s *URLsSuite) reindexed(page store.KnowledgeURL) store.KnowledgeURL {
	queued, err := s.service.Reindex(s.ctx, page.CustomerID, page.ID)
	s.Require().NoError(err)
	return s.settled(queued)
}

func (s *URLsSuite) TestAddingAPageQueuesItsReadRatherThanWaitingOnIt() {
	page, err := s.service.Add(s.ctx, "acme", Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().NoError(err)

	s.Equal(store.KnowledgeURLPending, page.State, "a crawl takes seconds, so the caller is not kept waiting on it")
	s.Nil(page.LastIndexedAt)
}

func (s *URLsSuite) TestAddingAPageIndexesItAndStampsWhenItWasRead() {
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})

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

func (s *URLsSuite) TestAReadThatFailedOnceIsTriedAgain() {
	s.reader.failures = 2

	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})

	s.Equal(store.KnowledgeURLIndexed, page.State, "a crawl that failed for a reason that passed is not broken")
	s.Empty(page.Error)
	s.Equal(2, page.Passages)
}

func (s *URLsSuite) TestAPageThatCouldNotBeReadIsKeptWithTheReason() {
	// The caller asked for this url to be part of the knowledge base. Telling them why it
	// is not, on a row they can retry, is more use than refusing and forgetting.
	s.reader.serve("", errors.New("CRAWL_NOT_FOUND"))

	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/gone"})

	s.Equal(store.KnowledgeURLFailed, page.State, "every retry failed")
	s.Contains(page.Error, "CRAWL_NOT_FOUND")
	s.Nil(page.LastIndexedAt, "nothing was ever read, which is not the same as broken since")
	s.Zero(page.Passages)
	s.Empty(s.base.ids(), "a page that could not be read writes nothing")
}

func (s *URLsSuite) TestAPageThatFailedCanBeReadAgainOnceItIsBack() {
	s.reader.serve("", errors.New("CRAWL_NOT_FOUND"))
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().Equal(store.KnowledgeURLFailed, page.State)

	s.reader.serve("# Pricing\n\nA call costs a penny.\n", nil)
	again := s.reindexed(page)

	s.Equal(store.KnowledgeURLIndexed, again.State, "a page that failed is not stuck failed")
	s.Empty(again.Error)
}

func (s *URLsSuite) TestRemovingAPageTakesItsPassagesWithIt() {
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().NotEmpty(s.base.ids())

	s.Require().NoError(s.service.Remove(s.ctx, "acme", page.ID))

	s.Empty(s.base.ids(),
		"an agent answering out of a page nobody subscribes to is worse than one that cannot")
	_, err := s.service.Get(s.ctx, "acme", page.ID)
	s.ErrorIs(err, store.ErrNoKnowledgeURL)
}

func (s *URLsSuite) TestAPageRemovedBeforeItWasReadWritesNothing() {
	// Nothing is reading until the page is gone, so the read finds nothing to read into.
	s.Require().NoError(s.service.Close())
	service, err := New(Options{Store: s.store, Redis: s.redis, Reader: s.reader, Writer: s.base, CheckInterval: 10 * time.Millisecond})
	s.Require().NoError(err)
	s.service = service

	page, err := service.Add(s.ctx, "acme", Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().NoError(err)
	s.Require().NoError(service.Remove(s.ctx, "acme", page.ID))
	s.Require().NoError(service.Start())
	s.drained(page)

	s.Empty(s.base.ids())
}

func (s *URLsSuite) TestAPageThatGotShorterLeavesNoOrphans() {
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().Equal(2, page.Passages)

	s.reader.serve("# Pricing\n\nA call costs tuppence.\n", nil)
	reindexed := s.reindexed(page)

	s.Equal(1, reindexed.Passages)
	s.Equal([]string{"https://example.com/pricing#0"}, s.base.ids(),
		"the old tail is no longer findable")
}

func (s *URLsSuite) TestReindexingRecordsWhenThePageWasLastRead() {
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	first := *page.LastIndexedAt

	reindexed := s.reindexed(page)

	s.Require().NotNil(reindexed.LastIndexedAt)
	s.True(reindexed.LastIndexedAt.After(first))
}

func (s *URLsSuite) TestAPageThatBrokeKeepsTheDateItLastWorked() {
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	worked := *page.LastIndexedAt

	s.reader.serve("", errors.New("SOURCE_NOT_AVAILABLE"))
	broken := s.reindexed(page)

	s.Equal(store.KnowledgeURLFailed, broken.State)
	s.Require().NotNil(broken.LastIndexedAt)
	s.Equal(worked.UTC(), broken.LastIndexedAt.UTC(),
		"when it last worked is what says how stale the answers now are")
}

func (s *URLsSuite) TestListingIsScopedToOneKnowledgeBase() {
	_, err := s.service.Add(s.ctx, "acme", Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().NoError(err)
	_, err = s.service.Add(s.ctx, "acme", Subscription{Namespace: "handbook", URL: "https://example.com/leave"})
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
	page, err := s.service.Add(s.ctx, "acme", Subscription{Namespace: "docs", URL: "https://example.com/pricing"})
	s.Require().NoError(err)

	_, err = s.service.Get(s.ctx, "globex", page.ID)

	s.ErrorIs(err, store.ErrNoKnowledgeURL)
}

func (s *URLsSuite) TestSomethingThatIsNotAFetchablePageIsRefused() {
	for _, address := range []string{"", "  ", "not a url at all", "mailto:sales@example.com", "file:///etc/passwd"} {
		_, err := s.service.Add(s.ctx, "acme", Subscription{Namespace: "docs", URL: address})
		s.Errorf(err, "%q should not be stored as a page to crawl", address)
	}

	listed, err := s.service.List(s.ctx, "acme", "docs")
	s.Require().NoError(err)
	s.Empty(listed)
}

func (s *URLsSuite) TestKnowledgeIsNeverSharedSoANamespaceIsRequired() {
	_, err := s.service.Add(s.ctx, "acme", Subscription{Namespace: "  ", URL: "https://example.com/pricing"})

	s.ErrorContains(err, "namespace")
}

func (s *URLsSuite) TestAddingTheSamePageAgainRereadsItRatherThanDuplicatingIt() {
	// Two rows would write the same passage ids, so the second would be a subscription
	// that removing the first one silently breaks. Re-reading instead is also what lets a
	// declaration of what an agent reads be applied again as it stands.
	first := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing", Title: "Pricing"})

	again := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing", Description: "What a call costs."})

	s.Equal(first.ID, again.ID)
	s.Equal("What a call costs.", again.Description, "the declaration is the one last written")
	s.Empty(again.DeclaredTitle, "a page that no longer says what it is called does not keep the old name")
	s.True(again.LastIndexedAt.After(*first.LastIndexedAt), "adding it again read it again")

	listed, err := s.service.List(s.ctx, "acme", "docs")
	s.Require().NoError(err)
	s.Len(listed, 1)
}

func (s *URLsSuite) TestWhatAPageWasSubscribedAsIsKeptThroughAReread() {
	page := s.added(Subscription{
		Namespace: "docs", URL: "https://example.com/pricing",
		Title: "What a call costs", Description: "The page sales points at.",
	})

	s.Equal("What a call costs", page.DeclaredTitle)
	s.Equal("Pricing", page.Title, "what the page calls itself is kept apart from what it was filed as")

	reindexed := s.reindexed(page)

	s.Equal("What a call costs", reindexed.DeclaredTitle, "a read must not overwrite the caller's words")
	s.Equal("The page sales points at.", reindexed.Description)
}

func (s *URLsSuite) TestPassageIDsCoverExactlyWhatWasWritten() {
	page := s.added(Subscription{Namespace: "docs", URL: "https://example.com/pricing"})

	written := s.base.ids()
	for _, id := range ingest.IDs(page.URL, 0, page.Passages) {
		s.Truef(slices.Contains(written, id), "%s was counted but never written", id)
	}
	s.Len(written, page.Passages)
}
