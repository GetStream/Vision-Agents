// Package turbopuffer looks knowledge up in turbopuffer's hosted search.
//
// turbopuffer publishes no Go SDK, so this speaks its v2 REST API directly: one call that
// ranks a namespace against a question. Ranking is BM25 rather than vector search because
// this service embeds nothing: an agent looking something up mid-sentence cannot wait for
// an embedding round trip on top of the search, and a caller's words are already the terms
// the handbook was written in.
package turbopuffer

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
)

const (
	apiKeyEnvVar = "TURBOPUFFER_API_KEY"
	regionEnvVar = "TURBOPUFFER_REGION"
)

const (
	defaultRegion = "gcp-us-central1"
	defaultLimit  = 5
	// upsertBatch caps how many passages go in one write, so ingesting a large body of
	// documentation is several requests rather than one that is refused for its size.
	upsertBatch = 256
	// defaultTimeout is short because a lookup happens while somebody is waiting on the
	// phone. A search that has not answered by now is worth less than the pause it costs.
	defaultTimeout = 5 * time.Second
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// The attributes a namespace is expected to carry. They match what the Python
// turbopuffer plugin writes, so a knowledge base indexed there is readable from here.
const (
	textAttribute   = "text"
	sourceAttribute = "source"
	distanceField   = "$dist"
	idField         = "id"
)

// Options configures a Store. The key falls back to the environment, the way every other
// provider in this service is configured.
type Options struct {
	// APIKey defaults to TURBOPUFFER_API_KEY.
	APIKey string
	// Region defaults to TURBOPUFFER_REGION, then to gcp-us-central1.
	Region string
	// BaseURL replaces the region-derived host, which is what a test or a proxy needs.
	BaseURL string
	// Timeout bounds one lookup. It happens mid-conversation, so it cannot hang a call.
	Timeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Store is a turbopuffer knowledge base. It satisfies knowledge.Store.
type Store struct {
	apiKey  string
	baseURL string
	client  *http.Client
	logger  *slog.Logger
}

// New validates the options and returns a Store.
func New(options Options) (*Store, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("turbopuffer: " + apiKeyEnvVar + " is required")
	}
	if options.BaseURL == "" {
		region := options.Region
		if region == "" {
			region = os.Getenv(regionEnvVar)
		}
		if region == "" {
			region = defaultRegion
		}
		options.BaseURL = "https://" + region + ".turbopuffer.com"
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Store{
		apiKey:  options.APIKey,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		client:  options.HTTPClient,
		logger:  options.Logger,
	}, nil
}

// Search returns the passages that bear on the question, most relevant first.
//
// A namespace nobody has written to yet is not an error: an agent configured against a
// knowledge base that is still empty should say it does not know, not fail the turn.
func (s *Store) Search(ctx context.Context, query knowledge.Query) ([]knowledge.Document, error) {
	if err := query.Validate(); err != nil {
		return nil, err
	}

	limit := query.Limit
	if limit <= 0 {
		limit = defaultLimit
	}

	body := queryRequest{
		RankBy:            []any{textAttribute, "BM25", query.Text},
		Limit:             limit,
		IncludeAttributes: []string{textAttribute, sourceAttribute},
	}

	path := "/v2/namespaces/" + url.PathEscape(query.Namespace) + "/query"
	var response queryResponse
	found, err := s.call(ctx, path, body, &response)
	if err != nil {
		return nil, err
	}
	if !found {
		s.logger.Debug("nothing has been written to this knowledge base yet",
			"namespace", query.Namespace)
		return nil, nil
	}

	documents := make([]knowledge.Document, 0, len(response.Rows))
	for _, row := range response.Rows {
		text := attribute(row, textAttribute)
		if text == "" {
			continue
		}
		documents = append(documents, knowledge.Document{
			ID:     attribute(row, idField),
			Text:   text,
			Source: attribute(row, sourceAttribute),
			Score:  score(row),
		})
	}
	return documents, nil
}

// Upsert writes passages into a namespace, replacing any that already have those ids.
//
// It is what fills a knowledge base rather than what an agent does with one, which is why
// it is here rather than on knowledge.Store: reading is the same everywhere and writing is
// the provider's own. The schema is declared on every write because turbopuffer creates a
// namespace on first use, and a passage nobody indexed for BM25 could not be found again.
func (s *Store) Upsert(ctx context.Context, namespace string, documents []knowledge.Document) error {
	if strings.TrimSpace(namespace) == "" {
		return errors.New("turbopuffer: a namespace is required")
	}

	path := "/v2/namespaces/" + url.PathEscape(namespace)
	for batch := range slices.Chunk(documents, upsertBatch) {
		rows := make([]upsertRow, 0, len(batch))
		for _, document := range batch {
			if document.ID == "" || strings.TrimSpace(document.Text) == "" {
				return errors.New("turbopuffer: every passage needs an id and something to read")
			}
			rows = append(rows, upsertRow{
				ID:     document.ID,
				Text:   document.Text,
				Source: document.Source,
			})
		}

		body := writeRequest{
			UpsertRows: rows,
			Schema: &writeSchema{
				Text:   attributeSchema{Type: "string", FullTextSearch: true},
				Source: attributeSchema{Type: "string"},
			},
		}
		var response writeResponse
		found, err := s.call(ctx, path, body, &response)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("turbopuffer: %s does not exist and was not created", namespace)
		}
		s.logger.Debug("wrote passages", "namespace", namespace, "rows", response.RowsUpserted)
	}
	return nil
}

// Delete removes passages by id.
//
// It is what lets a document be taken out of a knowledge base rather than only replaced:
// re-writing a source that got shorter leaves the passages past its new end behind, and an
// agent answering out of half a page nobody subscribes to any more is worse than one that
// says it does not know.
func (s *Store) Delete(ctx context.Context, namespace string, ids []string) error {
	if strings.TrimSpace(namespace) == "" {
		return errors.New("turbopuffer: a namespace is required")
	}
	if len(ids) == 0 {
		return nil
	}

	path := "/v2/namespaces/" + url.PathEscape(namespace)
	for batch := range slices.Chunk(ids, upsertBatch) {
		var response writeResponse
		found, err := s.call(ctx, path, writeRequest{Deletes: batch}, &response)
		if err != nil {
			return err
		}
		// A namespace nobody has written to holds none of these passages, which is the
		// state the caller asked for.
		if !found {
			return nil
		}
		s.logger.Debug("removed passages", "namespace", namespace, "rows", response.RowsDeleted)
	}
	return nil
}

// Provider is the name this store is recorded under.
func (s *Store) Provider() string { return "turbopuffer" }

// Client exposes the HTTP client, so a caller can reach parts of turbopuffer's API this
// does not wrap without building a second client.
func (s *Store) Client() *http.Client { return s.client }

// Close releases nothing: the store holds no connection of its own.
func (s *Store) Close() error { return nil }

// call posts to turbopuffer, reporting whether the namespace exists at all.
func (s *Store) call(ctx context.Context, path string, body, into any) (bool, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return false, fmt.Errorf("turbopuffer: encode %s: %w", path, err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.baseURL+path, bytes.NewReader(encoded))
	if err != nil {
		return false, fmt.Errorf("turbopuffer: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+s.apiKey)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := s.client.Do(request)
	if err != nil {
		return false, fmt.Errorf("turbopuffer: %s: %w", path, err)
	}
	defer response.Body.Close()

	if response.StatusCode == http.StatusNotFound {
		return false, nil
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return false, fmt.Errorf("turbopuffer: %s: %s: %s",
			path, response.Status, strings.TrimSpace(string(detail)))
	}

	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return false, fmt.Errorf("turbopuffer: decode %s: %w", path, err)
	}
	return true, nil
}

// attribute reads one field off a row as text. Rows are decoded loosely because a caller
// chooses what a document carries, and an id may be a number or a string depending on how
// the namespace was written.
func attribute(row map[string]json.RawMessage, name string) string {
	raw, ok := row[name]
	if !ok {
		return ""
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text
	}
	return strings.Trim(string(raw), `"`)
}

// score is the row's BM25 relevance, where turbopuffer reported one.
func score(row map[string]json.RawMessage) float64 {
	raw, ok := row[distanceField]
	if !ok {
		return 0
	}
	var relevance float64
	if err := json.Unmarshal(raw, &relevance); err != nil {
		return 0
	}
	return relevance
}

type queryRequest struct {
	// RankBy is turbopuffer's positional ranking expression: attribute, function, query.
	RankBy            []any    `json:"rank_by"`
	Limit             int      `json:"limit"`
	IncludeAttributes []string `json:"include_attributes"`
}

type queryResponse struct {
	Rows []map[string]json.RawMessage `json:"rows"`
}

// writeRequest is one write, which turbopuffer takes as any combination of writing and
// removing. The fields are omitted when empty because a delete carries no schema, and a
// schema declaring nothing is refused rather than ignored.
type writeRequest struct {
	UpsertRows []upsertRow  `json:"upsert_rows,omitempty"`
	Deletes    []string     `json:"deletes,omitempty"`
	Schema     *writeSchema `json:"schema,omitempty"`
}

// upsertRow is one passage. There is no vector on it, which is what lets the namespace be
// written without an embedding provider and read with BM25 alone.
type upsertRow struct {
	ID     string `json:"id"`
	Text   string `json:"text"`
	Source string `json:"source"`
}

// writeSchema is what a knowledge namespace holds: the passage, indexed for full-text
// search, and where it was read from.
type writeSchema struct {
	Text   attributeSchema `json:"text"`
	Source attributeSchema `json:"source"`
}

type attributeSchema struct {
	Type           string `json:"type"`
	FullTextSearch bool   `json:"full_text_search,omitempty"`
}

type writeResponse struct {
	RowsUpserted int `json:"rows_upserted"`
	RowsDeleted  int `json:"rows_deleted"`
}
