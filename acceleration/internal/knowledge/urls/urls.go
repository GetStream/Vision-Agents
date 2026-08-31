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
package urls

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Options configures a Service. All three dependencies are required: a page needs somewhere
// to be recorded, somebody to read it and somewhere to put what was read.
type Options struct {
	Store  *store.Store
	Reader search.Reader
	Writer knowledge.Writer
	// ChunkSize is how much of a page goes in one passage. Zero is the ingest default.
	ChunkSize int
	Logger    *slog.Logger
}

// Service is the control plane for the pages a knowledge base is kept filled from.
type Service struct {
	store     *store.Store
	reader    search.Reader
	writer    knowledge.Writer
	chunkSize int
	logger    *slog.Logger
}

// New validates the options and returns a Service.
func New(options Options) (*Service, error) {
	if options.Store == nil {
		return nil, errors.New("urls: a store is required")
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

	return &Service{
		store:     options.Store,
		reader:    options.Reader,
		writer:    options.Writer,
		chunkSize: options.ChunkSize,
		logger:    options.Logger,
	}, nil
}

// Add subscribes a knowledge base to a page and reads it for the first time.
//
// The row is written before the page is fetched, so a read that dies halfway through
// leaves something saying it was asked for rather than nothing at all. A page that could
// not be read is still a row, in the failed state with the reason on it: the caller asked
// for this url to be part of the knowledge base, and telling them why it is not is more
// use than refusing and forgetting.
func (s *Service) Add(ctx context.Context, customerID, namespace, address string) (store.KnowledgeURL, error) {
	namespace = strings.TrimSpace(namespace)
	if namespace == "" {
		return store.KnowledgeURL{}, errors.New("urls: a namespace is required, knowledge is never shared")
	}
	address, err := clean(address)
	if err != nil {
		return store.KnowledgeURL{}, err
	}

	page := store.KnowledgeURL{
		CustomerID: customerID,
		Namespace:  namespace,
		URL:        address,
		State:      store.KnowledgeURLPending,
	}
	if err := s.store.CreateKnowledgeURL(ctx, &page); err != nil {
		return store.KnowledgeURL{}, err
	}
	return s.index(ctx, page), nil
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

	if err := s.writer.Delete(ctx, page.Namespace, passageIDs(page.URL, 0, page.Passages)); err != nil {
		return err
	}
	return s.store.DeleteKnowledgeURL(ctx, customerID, page.ID)
}

// Reindex reads a page again and replaces what it wrote last time.
func (s *Service) Reindex(ctx context.Context, customerID, id string) (store.KnowledgeURL, error) {
	page, err := s.store.KnowledgeURL(ctx, customerID, id)
	if err != nil {
		return store.KnowledgeURL{}, err
	}
	return s.index(ctx, page), nil
}

// index reads the page, writes its passages and records what happened.
//
// It returns the row rather than an error, because a page that could not be read is a
// state this keeps rather than a request that failed. What it cannot do is report a
// database that would not take the update, which is logged: the read already happened and
// the passages are already written, so there is nothing to undo and nothing to retry.
func (s *Service) index(ctx context.Context, page store.KnowledgeURL) store.KnowledgeURL {
	written := page.Passages

	read, err := s.reader.Read(ctx, page.URL)
	if err == nil {
		err = s.write(ctx, &page, read)
	}
	if err != nil {
		s.logger.Warn("could not read a page into a knowledge base",
			"url", page.URL, "namespace", page.Namespace, "error", err)
		page.State = store.KnowledgeURLFailed
		page.Error = err.Error()
	}

	if saveErr := s.store.SaveKnowledgeURL(ctx, &page); saveErr != nil {
		s.logger.Error("could not record what reading a page made of it",
			"url", page.URL, "error", saveErr)
	}
	if err == nil {
		s.logger.Info("read a page into a knowledge base",
			"url", page.URL, "namespace", page.Namespace,
			"passages", page.Passages, "replaced", written)
	}
	return page
}

// write cuts the page into passages, writes them, and removes whatever the last read left
// past the end of this one. Writing first means a lookup landing mid-update reads the new
// passages or the old ones rather than a gap.
func (s *Service) write(ctx context.Context, page *store.KnowledgeURL, read search.Page) error {
	passages := ingest.Split(page.URL, read.Text, s.chunkSize)
	if len(passages) == 0 {
		return fmt.Errorf("urls: there is nothing to read at %s", page.URL)
	}
	if err := s.writer.Upsert(ctx, page.Namespace, passages); err != nil {
		return err
	}
	// A page that got shorter would otherwise leave its old tail behind, still findable
	// and no longer on the page it claims to come from.
	stale := passageIDs(page.URL, len(passages), page.Passages)
	if err := s.writer.Delete(ctx, page.Namespace, stale); err != nil {
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

// passageIDs is what ingest keyed the passages in [from, to) by.
func passageIDs(address string, from, to int) []string {
	if to <= from {
		return nil
	}
	ids := make([]string, 0, to-from)
	for index := from; index < to; index++ {
		ids = append(ids, fmt.Sprintf("%s%s%d", address, ingest.IDSeparator, index))
	}
	return ids
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
