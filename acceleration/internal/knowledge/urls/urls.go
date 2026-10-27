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
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TaskIndex is the task that reads one page into its knowledge base.
const TaskIndex = "knowledge:url:index"

const (
	// defaultQueue keeps crawls off whatever else shares the broker.
	defaultQueue = "knowledge"
	// againSuffix names the one read queued behind a read already under way, which may have
	// fetched the page before the read was asked for again.
	againSuffix = ":again"
	// indexTimeout is past the crawler's own timeout, so a slow read fails as the
	// crawler's error rather than as the task being cut off.
	indexTimeout = 2 * time.Minute
	// maxRetry is how many more times a read that failed is tried before the page is
	// marked failed.
	maxRetry = 3
	// concurrency is how many pages are read at once.
	concurrency = 4
	// sweep is how often pages are looked over for one due a refresh, unless
	// RefreshInterval says otherwise. A refresh is counted in hours, so a minute late is on
	// time.
	sweep = time.Minute
	// sweepBatch is how many due pages one sweep queues. The rest are queued by the next.
	sweepBatch = 100
)

// Options configures a Service. All four dependencies are required: a page needs somewhere
// to be recorded, a queue to be read from, somebody to read it and somewhere to put what
// was read.
type Options struct {
	Store *store.Store
	// Redis is the broker the reads are queued on.
	Redis  asynq.RedisConnOpt
	Reader search.Reader
	// Knowledge is where what was read is written, each call recorded for the customer: under
	// the key that asked for a read, or as the crawler's own work for a refresh.
	Knowledge *metered.Knowledge
	// ChunkSize is how much of a page goes in one passage. Zero is the ingest default.
	ChunkSize int
	// Limits bound how much larger a page's passages can be than the page, as the router's
	// settings name them.
	Limits ingest.Limits
	// CheckInterval is how often the worker looks for reads that are due, and how long a
	// failed one waits before it is tried again. Zero is asynq's defaults: a second between
	// checks and exponential backoff between retries.
	CheckInterval time.Duration
	// RefreshInterval is how often pages are looked over for one due a refresh. Zero is a
	// minute.
	RefreshInterval time.Duration
	// Queue is the asynq queue the reads go through, "knowledge" when empty. Services
	// sharing one Redis are kept apart by naming theirs differently.
	Queue  string
	Logger *slog.Logger
}

// Service is the control plane for the pages a knowledge base is kept filled from, and the
// worker that reads them.
type Service struct {
	store     *store.Store
	queue     *asynq.Client
	tasks     *asynq.Inspector
	queueName string
	worker    *asynq.Server
	reader    search.Reader
	knowledge *metered.Knowledge
	chunkSize int
	limits    ingest.Limits
	sweep     time.Duration
	logger    *slog.Logger
	// stop ends the sweep, and swept is closed once it has. Both are nil until Start.
	stop  chan struct{}
	swept chan struct{}
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
	if options.Knowledge == nil {
		return nil, errors.New("urls: a knowledge base is required")
	}
	if options.ChunkSize <= 0 {
		options.ChunkSize = ingest.DefaultChunk
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Queue == "" {
		options.Queue = defaultQueue
	}

	every := sweep
	if options.RefreshInterval > 0 {
		every = options.RefreshInterval
	}
	config := asynq.Config{
		Concurrency: concurrency,
		Queues:      map[string]int{options.Queue: 1},
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
		tasks:     asynq.NewInspector(options.Redis),
		queueName: options.Queue,
		worker:    asynq.NewServer(options.Redis, config),
		reader:    options.Reader,
		knowledge: options.Knowledge,
		chunkSize: options.ChunkSize,
		limits:    options.Limits,
		sweep:     every,
		logger:    options.Logger,
	}, nil
}

// Start begins reading the pages that are queued, including any left from before a
// restart: the queue is in Redis, so nothing asked for is lost with the process. It also
// starts queuing the pages due a refresh.
func (s *Service) Start() error {
	mux := asynq.NewServeMux()
	mux.HandleFunc(TaskIndex, s.process)
	if err := s.worker.Start(mux); err != nil {
		return err
	}
	s.stop, s.swept = make(chan struct{}), make(chan struct{})
	go s.refresh()
	return nil
}

// Close stops queuing refreshes, lets the reads in flight finish, then stops the worker and
// the queue.
func (s *Service) Close() error {
	if s.stop != nil {
		close(s.stop)
		<-s.swept
	}
	s.worker.Shutdown()
	return errors.Join(s.tasks.Close(), s.queue.Close())
}

// Subscription is a page a knowledge base is to be kept filled from: the url, and what the
// caller says it is. Title and Description are optional, and are theirs to write rather
// than anything the page claims about itself.
type Subscription struct {
	Namespace   string
	URL         string
	Title       string
	Description string
	// RefreshHours is how often the page is read again on its own. Zero is never.
	RefreshHours int
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
		return store.KnowledgeURL{}, stack.Wrap(errors.New("urls: a namespace is required, knowledge is never shared"))
	}
	address, err := clean(wanted.URL)
	if err != nil {
		return store.KnowledgeURL{}, err
	}
	if wanted.RefreshHours < 0 {
		return store.KnowledgeURL{}, stack.Wrap(fmt.Errorf("urls: refresh_hours cannot be negative, got %d", wanted.RefreshHours))
	}

	page, subscribed, err := s.store.SubscribedKnowledgeURL(ctx, customerID, namespace, address)
	if err != nil {
		return store.KnowledgeURL{}, err
	}
	page.DeclaredTitle = wanted.Title
	page.Description = wanted.Description
	page.RefreshHours = wanted.RefreshHours
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
	return page, s.readAgain(ctx, page)
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

	owner := routing.Owner{CustomerID: page.CustomerID, Attribution: routing.AttributionFrom(ctx)}
	if err := s.knowledge.For(owner).Delete(ctx, page.Namespace, ingest.IDs(page.URL, 0, page.Passages)); err != nil {
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
	return page, s.readAgain(ctx, page)
}

// refresh queues a read of every page due one, each sweep, until Close. Every router runs
// it, and a page queued by two of them is read once: the task id is the page.
func (s *Service) refresh() {
	defer close(s.swept)
	ticker := time.NewTicker(s.sweep)
	defer ticker.Stop()
	for {
		select {
		case <-s.stop:
			return
		case <-ticker.C:
		}
		ctx, cancel := context.WithTimeout(context.Background(), s.sweep)
		due, err := s.store.DueKnowledgeURLs(ctx, time.Now().UTC(), sweepBatch)
		if err != nil {
			s.logger.Warn("could not look for pages due a refresh", "error", err)
		}
		for _, page := range due {
			if err := s.enqueue(ctx, page); err != nil {
				s.logger.Warn("could not queue a refresh", "url", page.URL, "error", err)
			}
		}
		cancel()
	}
}

// crawlerOrigin is the key a read nobody asked for is recorded under: a refresh, or a task
// queued before the asker was kept on it.
var crawlerOrigin = routing.JobOrigin("crawler")

// indexPayload names the page a task reads. The customer is in it so the worker looks the
// row up the same way a request does, rather than by id alone. Attribution is who asked for
// the read, which its writes are recorded under; a refresh has none.
type indexPayload struct {
	CustomerID  string               `json:"customer_id"`
	ID          string               `json:"id"`
	Attribution *routing.Attribution `json:"attribution,omitempty"`
}

// owner is who the read's writes are recorded for.
func (p indexPayload) owner() routing.Owner {
	owner := routing.Owner{CustomerID: p.CustomerID}
	if p.Attribution != nil {
		owner.Attribution = *p.Attribution
	}
	if owner.APIKey == "" {
		owner.APIKey = crawlerOrigin
	}
	return owner
}

// enqueue queues a read of the page, for whoever ctx says asked. A read of it already waiting
// or under way is enough: the task id is the page, so asking twice before the worker gets to
// it reads it once. That is what a scheduled refresh wants: the page read, by any read.
func (s *Service) enqueue(ctx context.Context, page store.KnowledgeURL) error {
	if err := s.enqueueAs(ctx, page, page.ID); err != nil && !errors.Is(err, asynq.ErrTaskIDConflict) {
		return err
	}
	return nil
}

// readAgain queues a read someone asked for, by adding the page or adding it again, or by
// reindexing it, after whatever the page last showed. A read of it still waiting is enough. A read already under way is not: it may have fetched the page
// before this was asked, and it holds the page's id until asynq lets go of it, after the page
// shows it read. One more read is queued behind it, under an id of its own so that asking
// again meanwhile still reads the page once more rather than once per ask.
func (s *Service) readAgain(ctx context.Context, page store.KnowledgeURL) error {
	err := s.enqueueAs(ctx, page, page.ID)
	if !errors.Is(err, asynq.ErrTaskIDConflict) {
		return err
	}
	if running, inspectErr := s.tasks.GetTaskInfo(s.queueName, page.ID); inspectErr != nil ||
		running.State != asynq.TaskStateActive {
		// Waiting, or gone between the two calls, in which case the read just finished and
		// whoever asked can ask again: either way nothing is lost by stopping here.
		return nil
	}
	if err := s.enqueueAs(ctx, page, page.ID+againSuffix); err != nil && !errors.Is(err, asynq.ErrTaskIDConflict) {
		return err
	}
	return nil
}

// enqueueAs queues a read of the page under id, and returns asynq.ErrTaskIDConflict when a
// task holds that id already.
func (s *Service) enqueueAs(ctx context.Context, page store.KnowledgeURL, id string) error {
	queued := indexPayload{CustomerID: page.CustomerID, ID: page.ID}
	if asked := routing.AttributionFrom(ctx); asked.APIKey != "" {
		queued.Attribution = &asked
	}
	payload, err := json.Marshal(queued)
	if err != nil {
		return stack.Wrap(fmt.Errorf("urls: queue a read of %s: %w", page.URL, err))
	}

	_, err = s.queue.EnqueueContext(ctx, asynq.NewTask(TaskIndex, payload),
		asynq.Queue(s.queueName),
		asynq.TaskID(id),
		asynq.MaxRetry(maxRetry),
		asynq.Timeout(indexTimeout),
	)
	if errors.Is(err, asynq.ErrTaskIDConflict) {
		return err
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("urls: queue a read of %s: %w", page.URL, err))
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
		err = s.write(ctx, s.knowledge.For(payload.owner()), &page, read)
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
func (s *Service) write(ctx context.Context, writer knowledge.Writer, page *store.KnowledgeURL, read search.Page) error {
	passages := ingest.Split(page.URL, read.Text, s.chunkSize, s.limits)
	if len(passages) == 0 {
		return fmt.Errorf("urls: there is nothing to read at %s", page.URL)
	}
	if err := writer.Upsert(ctx, page.Namespace, passages); err != nil {
		return err
	}
	// A page that got shorter would otherwise leave its old tail behind, still findable
	// and no longer on the page it claims to come from.
	stale := ingest.IDs(page.URL, len(passages), page.Passages)
	if err := writer.Delete(ctx, page.Namespace, stale); err != nil {
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
		return "", stack.Wrap(errors.New("urls: a url is required"))
	}
	parsed, err := url.Parse(address)
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("urls: %s is not a url: %w", address, err))
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", stack.Wrap(fmt.Errorf("urls: %s is not a page that can be fetched", address))
	}
	if parsed.Host == "" {
		return "", stack.Wrap(fmt.Errorf("urls: %s names no host", address))
	}
	return address, nil
}
