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
	"math"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// kilobyte is the unit turbopuffer's batch discount is reckoned in. It bills decimal units
// throughout: a GB is 10^9 bytes.
const kilobyte = 1000

// listPage is how many namespaces one listing asks for, turbopuffer's most.
const listPage = 1000

// Cost is what turbopuffer charges per unit, by meter: bytes_queried and bytes_returned for a
// query, bytes_written for a write (its minimum is per request), byte_hours for storage. It is
// the deployment's config, so a change in turbopuffer's prices is not a change of code. A
// meter it leaves out costs nothing.
type Cost map[pricing.Meter]pricing.Rate

// query is what a query cost under turbopuffer's formula: bytes queried, its 1.28 GB minimum
// and large-namespace discounts already in the figure it reports, and bytes returned.
func (c Cost) query(billing queryBilling) int64 {
	return c[pricing.BytesQueried].Nanos(billing.BytesQueried) + c[pricing.BytesReturned].Nanos(billing.BytesReturned)
}

// write is what one write request cost: at least the rate's minimum, less turbopuffer's batch
// discount, min(0.5, max(0, 0.2 × (log10(KB) − 1))), worked out once on the whole request.
func (c Cost) write(bytes int64) int64 {
	rate, ok := c[pricing.BytesWritten]
	if !ok {
		return 0
	}
	billable := max(bytes, rate.Minimum)
	if billable <= 0 {
		return 0
	}
	discount := min(0.5, max(0, 0.2*(math.Log10(float64(billable)/kilobyte)-1)))
	rate.Minimum = 0
	return rate.Nanos(int64(math.Round(float64(billable) * (1 - discount))))
}

// stored is what an hour of a namespace's logical bytes cost.
func (c Cost) stored(bytes int64) int64 { return c[pricing.ByteHours].Nanos(bytes) }

// Options configures a Store. The key falls back to the environment, the way every other
// provider in this service is configured.
type Options struct {
	// APIKey defaults to TURBOPUFFER_API_KEY.
	APIKey string
	// Cost is what turbopuffer charges, which each call's cost to the operator is worked out
	// from. Nil works out nothing.
	Cost Cost
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

// Store is a turbopuffer knowledge base. It satisfies knowledge.Backend: every call says what
// turbopuffer billed it, read from the response's billing object.
type Store struct {
	apiKey  string
	baseURL string
	cost    Cost
	client  *http.Client
	logger  *slog.Logger
}

// New validates the options and returns a Store.
func New(options Options) (*Store, error) {
	keys := keypool.FromEnv(options.APIKey, apiKeyEnvVar)
	if keys.Len() == 0 {
		return nil, errors.New("turbopuffer: " + apiKeyEnvVar + " is required")
	}
	options.APIKey = keys.Peek()
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
		cost:    options.Cost,
		client:  keys.WrapClient(options.HTTPClient, keypool.Bearer),
		logger:  options.Logger,
	}, nil
}

// Search returns the passages that bear on the question, most relevant first, and what
// turbopuffer billed the query.
//
// A namespace nobody has written to yet is not an error: an agent configured against a
// knowledge base that is still empty should say it does not know, not fail the turn.
func (s *Store) Search(ctx context.Context, query knowledge.Query) (knowledge.Found, error) {
	if err := query.Validate(); err != nil {
		return knowledge.Found{}, err
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
	found, err := s.call(ctx, http.MethodPost, path, body, &response)
	if err != nil {
		return knowledge.Found{}, err
	}
	if !found {
		s.logger.Debug("nothing has been written to this knowledge base yet",
			"namespace", query.Namespace)
		return knowledge.Found{Billed: knowledge.Billed{Requests: 1}}, nil
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
	return knowledge.Found{Documents: documents, Billed: s.queried(response)}, nil
}

// Upsert writes passages into a namespace, replacing any that already have those ids, and
// says what turbopuffer billed every request it took.
//
// It is what fills a knowledge base rather than what an agent does with one, which is why
// it is here rather than on knowledge.Store: reading is the same everywhere and writing is
// the provider's own. The schema is declared on every write because turbopuffer creates a
// namespace on first use, and a passage nobody indexed for BM25 could not be found again.
func (s *Store) Upsert(ctx context.Context, namespace string, documents []knowledge.Document) (knowledge.Billed, error) {
	if strings.TrimSpace(namespace) == "" {
		return knowledge.Billed{}, stack.Wrap(errors.New("turbopuffer: a namespace is required"))
	}
	// Checked before anything is written, so a bad passage late in the list does not leave the
	// batches before it written and billed under a call that failed.
	for _, document := range documents {
		if document.ID == "" || strings.TrimSpace(document.Text) == "" {
			return knowledge.Billed{}, stack.Wrap(errors.New("turbopuffer: every passage needs an id and something to read"))
		}
	}

	path := "/v2/namespaces/" + url.PathEscape(namespace)
	var billed knowledge.Billed
	for batch := range slices.Chunk(documents, upsertBatch) {
		rows := make([]upsertRow, 0, len(batch))
		for _, document := range batch {
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
		found, err := s.call(ctx, http.MethodPost, path, body, &response)
		if err != nil {
			return billed, stack.Wrap(err)
		}
		if !found {
			return billed, stack.Wrap(fmt.Errorf("turbopuffer: %s does not exist and was not created", namespace))
		}
		billed = billed.Plus(s.written(response))
		s.logger.Debug("wrote passages", "namespace", namespace, "rows", response.RowsUpserted)
	}
	return billed, nil
}

// Delete removes passages by id, and says what turbopuffer billed every request it took.
//
// It is what lets a document be taken out of a knowledge base rather than only replaced:
// re-writing a source that got shorter leaves the passages past its new end behind, and an
// agent answering out of half a page nobody subscribes to any more is worse than one that
// says it does not know.
func (s *Store) Delete(ctx context.Context, namespace string, ids []string) (knowledge.Billed, error) {
	if strings.TrimSpace(namespace) == "" {
		return knowledge.Billed{}, stack.Wrap(errors.New("turbopuffer: a namespace is required"))
	}

	path := "/v2/namespaces/" + url.PathEscape(namespace)
	var billed knowledge.Billed
	for batch := range slices.Chunk(ids, upsertBatch) {
		var response writeResponse
		found, err := s.call(ctx, http.MethodPost, path, writeRequest{Deletes: batch}, &response)
		if err != nil {
			return billed, stack.Wrap(err)
		}
		// A namespace nobody has written to holds none of these passages, which is the
		// state the caller asked for.
		if !found {
			return billed.Plus(knowledge.Billed{Requests: 1}), nil
		}
		billed = billed.Plus(s.written(response))
		s.logger.Debug("removed passages", "namespace", namespace, "rows", response.RowsDeleted)
	}
	return billed, nil
}

// Fetch reads passages back by id, which is how somebody sees what a document became, and
// says what turbopuffer billed every query it took.
func (s *Store) Fetch(ctx context.Context, namespace string, ids []string) (knowledge.Found, error) {
	if strings.TrimSpace(namespace) == "" {
		return knowledge.Found{}, stack.Wrap(errors.New("turbopuffer: a namespace is required"))
	}

	path := "/v2/namespaces/" + url.PathEscape(namespace) + "/query"
	byID := make(map[string]knowledge.Document, len(ids))
	var billed knowledge.Billed
	for batch := range slices.Chunk(ids, upsertBatch) {
		body := fetchRequest{
			Filters:           []any{idField, "In", batch},
			RankBy:            []any{idField, "asc"},
			Limit:             len(batch),
			IncludeAttributes: []string{textAttribute, sourceAttribute},
		}
		var response queryResponse
		found, err := s.call(ctx, http.MethodPost, path, body, &response)
		if err != nil {
			return knowledge.Found{Billed: billed}, stack.Wrap(err)
		}
		if !found {
			return knowledge.Found{Billed: billed.Plus(knowledge.Billed{Requests: 1})}, nil
		}
		billed = billed.Plus(s.queried(response))
		for _, row := range response.Rows {
			id := attribute(row, idField)
			byID[id] = knowledge.Document{
				ID:     id,
				Text:   attribute(row, textAttribute),
				Source: attribute(row, sourceAttribute),
			}
		}
	}

	documents := make([]knowledge.Document, 0, len(byID))
	for _, id := range ids {
		if document, ok := byID[id]; ok {
			documents = append(documents, document)
		}
	}
	return knowledge.Found{Documents: documents, Billed: billed}, nil
}

// Namespaces lists every namespace whose name starts with prefix, a page at a time.
// turbopuffer bills no listing.
func (s *Store) Namespaces(ctx context.Context, prefix string) ([]string, error) {
	var names []string
	cursor := ""
	for {
		query := url.Values{"page_size": {strconv.Itoa(listPage)}}
		if prefix != "" {
			query.Set("prefix", prefix)
		}
		if cursor != "" {
			query.Set("cursor", cursor)
		}
		var page namespacesResponse
		if _, err := s.call(ctx, http.MethodGet, "/v1/namespaces?"+query.Encode(), nil, &page); err != nil {
			return nil, stack.Wrap(err)
		}
		for _, namespace := range page.Namespaces {
			names = append(names, namespace.ID)
		}
		if page.NextCursor == "" || page.NextCursor == cursor {
			return names, nil
		}
		cursor = page.NextCursor
	}
}

// Stored reads a namespace's size from its metadata, as an hour of it. turbopuffer does not
// update the figure while a write is still being indexed, so a sample can lag a large write by
// an hour.
func (s *Store) Stored(ctx context.Context, namespace string) (knowledge.Billed, error) {
	var metadata metadataResponse
	found, err := s.call(ctx, http.MethodGet, "/v1/namespaces/"+url.PathEscape(namespace)+"/metadata", nil, &metadata)
	if err != nil {
		return knowledge.Billed{}, stack.Wrap(err)
	}
	if !found {
		return knowledge.Billed{}, stack.Wrap(fmt.Errorf("%w: %s", knowledge.ErrNoNamespace, namespace))
	}
	return knowledge.Billed{
		Requests:  1,
		ByteHours: metadata.ApproxLogicalBytes,
		CostNanos: s.cost.stored(metadata.ApproxLogicalBytes),
	}, nil
}

// Provider is the name this store is recorded under.
func (s *Store) Provider() string { return "turbopuffer" }

// Client exposes the HTTP client, so a caller can reach parts of turbopuffer's API this
// does not wrap without building a second client.
func (s *Store) Client() *http.Client { return s.client }

// Close releases nothing: the store holds no connection of its own.
func (s *Store) Close() error { return nil }

// queried is what one query was billed.
func (s *Store) queried(response queryResponse) knowledge.Billed {
	return knowledge.Billed{
		Requests:      1,
		BytesQueried:  response.Billing.BytesQueried,
		BytesReturned: response.Billing.BytesReturned,
		Results:       int64(len(response.Rows)),
		CostNanos:     s.cost.query(response.Billing),
	}
}

// written is what one write request was billed.
func (s *Store) written(response writeResponse) knowledge.Billed {
	return knowledge.Billed{
		Requests:     1,
		BytesWritten: response.Billing.BytesWritten,
		CostNanos:    s.cost.write(response.Billing.BytesWritten),
	}
}

// call asks turbopuffer, reporting whether what it names exists at all. A nil body sends none.
func (s *Store) call(ctx context.Context, method, path string, body, into any) (bool, error) {
	var sent io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return false, fmt.Errorf("turbopuffer: encode %s: %w", path, err)
		}
		sent = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, s.baseURL+path, sent)
	if err != nil {
		return false, fmt.Errorf("turbopuffer: %s: %w", path, err)
	}
	request.Header.Set("Authorization", "Bearer "+s.apiKey)
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
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

// fetchRequest reads rows by id rather than ranking them against a question.
type fetchRequest struct {
	Filters           []any    `json:"filters"`
	RankBy            []any    `json:"rank_by"`
	Limit             int      `json:"limit"`
	IncludeAttributes []string `json:"include_attributes"`
}

type queryResponse struct {
	Rows    []map[string]json.RawMessage `json:"rows"`
	Billing queryBilling                 `json:"billing"`
}

// queryBilling is what turbopuffer billed a query: the bytes it counts as searched, after
// its 1.28 GB minimum and its discounts on large namespaces, and the bytes it returned.
type queryBilling struct {
	BytesQueried  int64 `json:"billable_logical_bytes_queried"`
	BytesReturned int64 `json:"billable_logical_bytes_returned"`
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
	RowsUpserted int          `json:"rows_upserted"`
	RowsDeleted  int          `json:"rows_deleted"`
	Billing      writeBilling `json:"billing"`
}

// writeBilling is what turbopuffer billed a write: the logical bytes written, after its
// discount on attributes nothing filters on and before its discount on large batches, which
// Cost works out.
type writeBilling struct {
	BytesWritten int64 `json:"billable_logical_bytes_written"`
}

type namespacesResponse struct {
	Namespaces []struct {
		ID string `json:"id"`
	} `json:"namespaces"`
	NextCursor string `json:"next_cursor"`
}

// metadataResponse is the part of a namespace's metadata storage is billed by.
type metadataResponse struct {
	ApproxLogicalBytes int64 `json:"approx_logical_bytes"`
}
