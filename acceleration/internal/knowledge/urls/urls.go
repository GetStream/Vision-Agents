// Package urls keeps a knowledge base filled from pages published elsewhere.
//
// Ingesting a document is a thing that happens once: somebody posts a handbook and it is
// cut into passages. A URL is a subscription instead, because the page behind it changes
// and nobody re-posts it. So this holds a row per url, remembers when each was last read
// and what it became, and can take one away again along with everything it wrote.
//
// Reading the page is the search provider's job rather than this one's: a crawler that
// renders JavaScript, handles PDFs and strips the navigation out is not worth writing
// twice, and Exa already returns a page as the markdown a knowledge base wants.
//
// A read is a task on an asynq queue in Redis rather than part of the request that asked
// for it. A live crawl takes seconds and sometimes fails for reasons that pass, so the
// caller gets the row back straight away and the worker retries what did not work.
package urls

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"strings"
	"time"

	"github.com/hibiken/asynq"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TaskIndex is the task that reads one page into its knowledge base.
const TaskIndex = "knowledge:url:index"

const (
	// queue keeps crawls off whatever else shares the broker.
	queue = "knowledge"
	// indexTimeout is past the crawler's own timeout, so a slow read fails as the
	// crawler's error rather than as the task being cut off.
	indexTimeout = 2 * time.Minute
	// maxRetry is how many more times a read that failed is tried before the page is
	// marked failed.
	maxRetry = 3
	// concurrency is how many pages are read at once.
	concurrency = 4
)

// Options configures a Service. All four dependencies are required: a page needs somewhere
// to be recorded, a queue to be read from, somebody to read it and somewhere to put what
// was read.
type Options struct {
	Store *store.Store
	// Redis is the broker the reads are queued on.
	Redis  asynq.RedisConnOpt
	Reader search.Reader
	Writer knowledge.Writer
	// ChunkSize is how much of a page goes in one passage. Zero is the ingest default.
	ChunkSize int
	// CheckInterval is how often the worker looks for reads that are due, and how long a
	// failed one waits before it is tried again. Zero is asynq's defaults: a second between
	// checks and exponential backoff between retries.
	CheckInterval time.Duration
	Logger        *slog.Logger
}

// Service is the control plane for the pages a knowledge base is kept filled from, and the
// worker that reads them.
type Service struct {
	store     *store.Store
	queue     *asynq.Client
	worker    *asynq.Server
	reader    search.Reader
	writer    knowledge.Writer
	chunkSize int
	logger    *slog.Logger
}

// New validates the options and returns a Service. Nothing is read until Start.
func New(options Options) (*Service, error) {
	if options.Store == nil {
		return nil, errors.New("urls: a store is required")
	}
	if options.Redis == nil {
		return nil, errors.New("urls: a redis to queue the reads on is required")
	}
	if options.Reader == nil {
		return nil, errors.New("urls: something has to read the pages")
	}
	if options.Writer == nil {
		return nil, errors.New("urls: a knowledge base is required")
	}
	if options.ChunkSize <= 0 {
		options.ChunkSize = ingest.DefaultChunk
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	config := asynq.Config{
		Concurrency: concurrency,
		Queues:      map[string]int{queue: 1},
		LogLevel:    asynq.WarnLevel,
	}
	if interval := options.CheckInterval; interval > 0 {
		config.TaskCheckInterval = interval
		config.DelayedTaskCheckInterval = interval
		config.RetryDelayFunc = func(int, error, *asynq.Task) time.Duration { return interval }
	}

	return &Service{
		store:     options.Store,
		queue:     asynq.NewClient(options.Redis),
		worker:    asynq.NewServer(options.Redis, config),
		reader:    options.Reader,
		writer:    options.Writer,
		chunkSize: options.ChunkSize,
		logger:    options.Logger,
	}, nil
}

// Start begins reading the pages that are queued, including any left from before a
// restart: the queue is in Redis, so nothing asked for is lost with the process.
func (s *Service) Start() error {
	mux := asynq.NewServeMux()
	mux.HandleFunc(TaskIndex, s.process)
	return s.worker.Start(mux)
}

// Close lets the reads in flight finish, then stops the worker and the queue.
func (s *Service) Close() error {
	s.worker.Shutdown()
	return s.queue.Close()
}

// Subscription is a page a knowledge base is to be kept filled from: the url, and what the
// caller says it is. Title and Description are optional, and are theirs to write rather
// than anything the page claims about itself.
type Subscription struct {
	Namespace   string
	URL         string
	Title       string
	Description string
}

// Add subscribes a knowledge base to a page and queues its first read.
//
// The row is written before anything is queued, so a read that never happens still leaves
// something saying it was asked for. A page that could not be read is still a row, in the
// failed state with the reason on it: the caller asked for this url to be part of the
// knowledge base, and telling them why it is not is more use than refusing and forgetting.
//
// A page the base already has is re-read rather than subscribed to twice, since the
// subscription is the url and both would write the same passages anyway. That is what lets
// a declaration of what an agent reads be applied again as it stands, without the caller
// first working out which pages are new.
func (s *Service) Add(ctx context.Context, customerID string, wanted Subscription) (store.KnowledgeURL, error) {
	namespace := strings.TrimSpace(wanted.Namespace)
	if namespace == "" {
		return store.KnowledgeURL{}, errors.New("urls: a namespace is required, knowledge is never shared")
	}
	address, err := clean(wanted.URL)
	if err != nil {
		return store.KnowledgeURL{}, err
	}

	page, subscribed, err := s.store.SubscribedKnowledgeURL(ctx, customerID, namespace, address)
	if err != nil {
		return store.KnowledgeURL{}, err
	}
	page.DeclaredTitle = wanted.Title
	page.Description = wanted.Description
	if subscribed {
		if err := s.store.SaveKnowledgeURL(ctx, &page); err != nil {
			return store.KnowledgeURL{}, err
		}
	} else {
		page.CustomerID = customerID
		page.Namespace = namespace
		page.URL = address
		page.State = store.KnowledgeURLPending
		if err := s.store.CreateKnowledgeURL(ctx, &page); err != nil {
			return store.KnowledgeURL{}, err
		}
	}
	return page, s.enqueue(ctx, page)
}

// List returns the pages a knowledge base is filled from, newest first. An empty namespace
// lists every one the customer has.
func (s *Service) List(ctx context.Context, customerID, namespace string) ([]store.KnowledgeURL, error) {
	return s.store.CustomerKnowledgeURLs(ctx, customerID, strings.TrimSpace(namespace))
}

// Get returns one page.
func (s *Service) Get(ctx context.Context, customerID, id string) (store.KnowledgeURL, error) {
	return s.store.KnowledgeURL(ctx, customerID, id)
}

// Remove takes a page out of the knowledge base, passages and all.
//
// The passages go first, for the same reason a voice is unregistered from its providers
// before its row is deleted: a page we have forgotten we subscribe to but are still
// answering out of is the failure worth avoiding.
func (s *Service) Remove(ctx context.Context, customerID, id string) error {
	page, err := s.store.KnowledgeURL(ctx, customerID, id)
	if err != nil {
		return err
	}

	base := knowledge.Scoped(page.CustomerID, page.Namespace)
	if err := s.writer.Delete(ctx, base, ingest.IDs(page.URL, 0, page.Passages)); err != nil {
		return err
	}
	return s.store.DeleteKnowledgeURL(ctx, customerID, page.ID)
}

// Reindex queues a page to be read again, replacing what it wrote last time. The row comes
// back as it is now; the read lands on it when the worker gets to it.
func (s *Service) Reindex(ctx context.Context, customerID, id string) (store.KnowledgeURL, error) {
	page, err := s.store.KnowledgeURL(ctx, customerID, id)
	if err != nil {
		return store.KnowledgeURL{}, err
	}
	return page, s.enqueue(ctx, page)
}

// indexPayload names the page a task reads. The customer is in it so the worker looks the
// row up the same way a request does, rather than by id alone.
type indexPayload struct {
	CustomerID string `json:"customer_id"`
	ID         string `json:"id"`
}

// enqueue queues a read of the page. A read of it already waiting is enough: the task id
// is the page, so asking twice before the worker gets to it reads it once.
func (s *Service) enqueue(ctx context.Context, page store.KnowledgeURL) error {
	payload, err := json.Marshal(indexPayload{CustomerID: page.CustomerID, ID: page.ID})
	if err != nil {
		return fmt.Errorf("urls: queue a read of %s: %w", page.URL, err)
	}

	_, err = s.queue.EnqueueContext(ctx, asynq.NewTask(TaskIndex, payload),
		asynq.Queue(queue),
		asynq.TaskID(page.ID),
		asynq.MaxRetry(maxRetry),
		asynq.Timeout(indexTimeout),
	)
	if errors.Is(err, asynq.ErrTaskIDConflict) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("urls: queue a read of %s: %w", page.URL, err)
	}
	return nil
}

// process is the worker's half of a read: it reads the page, writes its passages and
// records what happened.
//
// A read that failed is returned to asynq to be tried again, with the reason kept on the
// row meanwhile. Only the last attempt marks the page failed, so a crawl that failed once
// for a reason that passed is not reported as broken. A page removed since it was queued
// has nothing left to read into.
func (s *Service) process(ctx context.Context, task *asynq.Task) error {
	var payload indexPayload
	if err := json.Unmarshal(task.Payload(), &payload); err != nil {
		return fmt.Errorf("urls: read the task: %v: %w", err, asynq.SkipRetry)
	}

	page, err := s.store.KnowledgeURL(ctx, payload.CustomerID, payload.ID)
	if errors.Is(err, store.ErrNoKnowledgeURL) {
		return nil
	}
	if err != nil {
		return err
	}

	written := page.Passages
	read, err := s.reader.Read(ctx, page.URL)
	if err == nil {
		err = s.write(ctx, &page, read)
	}
	last := false
	if err != nil {
		retried, _ := asynq.GetRetryCount(ctx)
		limit, _ := asynq.GetMaxRetry(ctx)
		last = retried >= limit
		s.logger.Warn("could not read a page into a knowledge base",
			"url", page.URL, "namespace", page.Namespace, "attempt", retried+1, "error", err)
		page.Error = err.Error()
		if last {
			page.State = store.KnowledgeURLFailed
		}
	}

	if saveErr := s.store.SaveKnowledgeURL(ctx, &page); saveErr != nil {
		return errors.Join(err, saveErr)
	}
	// The last failure is on the row, so the task ends there rather than being archived:
	// an archived task keeps its id, and the page could not be queued to be read again.
	if err != nil {
		if last {
			return nil
		}
		return err
	}
	s.logger.Info("read a page into a knowledge base",
		"url", page.URL, "namespace", page.Namespace,
		"passages", page.Passages, "replaced", written)
	return nil
}

// write cuts the page into passages, writes them, and removes whatever the last read left
// past the end of this one. Writing first means a lookup landing mid-update reads the new
// passages or the old ones rather than a gap.
func (s *Service) write(ctx context.Context, page *store.KnowledgeURL, read search.Page) error {
	passages := ingest.Split(page.URL, read.Text, s.chunkSize)
	if len(passages) == 0 {
		return fmt.Errorf("urls: there is nothing to read at %s", page.URL)
	}
	base := knowledge.Scoped(page.CustomerID, page.Namespace)
	if err := s.writer.Upsert(ctx, base, passages); err != nil {
		return err
	}
	// A page that got shorter would otherwise leave its old tail behind, still findable
	// and no longer on the page it claims to come from.
	stale := ingest.IDs(page.URL, len(passages), page.Passages)
	if err := s.writer.Delete(ctx, base, stale); err != nil {
		return err
	}

	indexed := time.Now().UTC()
	page.Title = read.Title
	page.State = store.KnowledgeURLIndexed
	page.Error = ""
	page.Passages = len(passages)
	page.LastIndexedAt = &indexed
	return nil
}

// clean reports the url as it will be stored, or why it is not one.
//
// It is strict about the scheme because the rest of this hands the address to a crawler
// and then keys passages by it: a file path or a mailto would be recorded, fetched, and
// fail, which is a slower way of saying no.
func clean(address string) (string, error) {
	address = strings.TrimSpace(address)
	if address == "" {
		return "", errors.New("urls: a url is required")
	}
	parsed, err := url.Parse(address)
	if err != nil {
		return "", fmt.Errorf("urls: %s is not a url: %w", address, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("urls: %s is not a page that can be fetched", address)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("urls: %s names no host", address)
	}
	return address, nil
}
