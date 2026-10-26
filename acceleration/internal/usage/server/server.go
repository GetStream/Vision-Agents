// Package server is the usage service's HTTP API: producers ship events to it, and routers
// read what subjects spent, organizations' standings and the accounts as the organizations
// themselves may see them from it, with the ingest token; the
// operator manages tiers, accounts and credit through the management API, with the manage
// key. The health check takes neither.
package server

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humachi"
	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

const (
	// tag keeps every operation out of the public SDKs: none is a customer's to call.
	tag = "Staff"
	// ingestScheme is the security scheme of every operation but the health check.
	ingestScheme = "IngestToken"
	// ingestOperation is the operation producers ship batches to.
	ingestOperation = "ingestUsageEvents"
	// healthTimeout bounds the health check's database ping.
	healthTimeout = 2 * time.Second
	// maxLoggedEvent is the most of a rejected event its log line carries.
	maxLoggedEvent = 16 << 10
	// routerSource starts the source of a router's events, the one producer that ships as it
	// works; rollups and reconcile jobs ship late by design.
	routerSource = "router/"
)

// Options configure a Server.
type Options struct {
	Ledger *ledger.Ledger
	// IngestToken is the bearer token of the producers' and routers' operations.
	IngestToken string
	// ManageKey is what the management API takes in X-Manage-Key; empty turns it off.
	ManageKey string
	Logger    *slog.Logger
	// Now is the clock answers are stamped with; nil is time.Now.
	Now func() time.Time
}

// Server answers the usage service's API.
type Server struct {
	ledger    *ledger.Ledger
	token     [sha256.Size]byte
	manageKey [sha256.Size]byte
	manageOn  bool
	logger    *slog.Logger
	now       func() time.Time
}

// New returns a server over options.Ledger.
func New(options Options) (*Server, error) {
	if options.Ledger == nil || options.IngestToken == "" {
		return nil, errors.New("server: a ledger and an ingest token are required")
	}
	if options.ManageKey == options.IngestToken {
		return nil, errors.New("server: the manage key cannot be the ingest token, which every producer holds")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return &Server{
		ledger: options.Ledger, token: sha256.Sum256([]byte(options.IngestToken)),
		manageKey: sha256.Sum256([]byte(options.ManageKey)), manageOn: options.ManageKey != "",
		logger: options.Logger, now: options.Now,
	}, nil
}

// Handler serves the API.
func (s *Server) Handler() http.Handler {
	router := chi.NewRouter()
	s.register(router)
	return router
}

// register declares every operation on router and returns the API describing them.
func (s *Server) register(router chi.Router) huma.API {
	config := huma.Config{
		OpenAPI: &huma.OpenAPI{
			OpenAPI: "3.1.0",
			Info: &huma.Info{
				Title:   "Usage service",
				Version: "0.1.0",
				Description: "Records the usage events producers ship, once per usage id, keeps what every subject spent in " +
					"every daily, weekly, bi-weekly and monthly window, and mirrors those totals to Redis. It keeps each " +
					"organization's account (credit, spend, status and its own settings), debited in the transaction that " +
					"records its events, and what each tier string decides; routers read both in organizations' standings. " +
					"It does not price (the producer sends what the work cost), authenticate customers, or admit requests. " +
					"Reachable only inside the cluster, from the producers, the routers and the operator's management " +
					"tools.\n\nA request whose shape the schema refuses is a 422. The management API's own failures carry a " +
					"`code` beside the problem's fields: `invalid_request` (400), `unauthenticated` (401), `not_configured` " +
					"(404: the service has no manage key), `default_tier` and `idempotency_key_reused` (409).",
			},
			Servers: []*huma.Server{{URL: "http://localhost:8070"}},
			Components: &huma.Components{
				Schemas: huma.NewMapRegistry("#/components/schemas/", huma.DefaultSchemaNamer),
				SecuritySchemes: map[string]*huma.SecurityScheme{
					ingestScheme: {
						Type:        "http",
						Scheme:      "bearer",
						Description: "The service's ingest token, USAGE_INGEST_TOKEN, held by the producers and routers.",
					},
					manageScheme: {
						Type:        "apiKey",
						In:          "header",
						Name:        manageKeyHeader,
						Description: "The service's manage key, USAGE_MANAGE_KEY, held by the operator's management tools alone.",
					},
				},
			},
		},
		Formats:       huma.DefaultFormats,
		DefaultFormat: "application/json",
		Transformers:  []huma.Transformer{dropBodyEcho},
	}
	// A producer newer than the service sends fields it does not know yet.
	config.AllowAdditionalPropertiesByDefault = true
	api := humachi.New(router, config)
	api.UseMiddleware(countRefused, s.requireToken(api))

	security := []map[string][]string{{ingestScheme: {}}}
	huma.Register(api, huma.Operation{
		OperationID: "getUsageHealth",
		Method:      http.MethodGet,
		Path:        "/healthz",
		Summary:     "Whether the service can record events",
		Description: "200 when the service's database answers, 503 when it does not. Takes no token.",
		Tags:        []string{tag},
		Security:    []map[string][]string{},
		Errors:      []int{http.StatusServiceUnavailable},
	}, s.health)
	huma.Register(api, huma.Operation{
		OperationID: ingestOperation,
		Method:      http.MethodPost,
		Path:        usage.PathEvents,
		Summary:     "Record usage events",
		Description: "Records each event that passes the contract's checks and is not recorded yet, and adds it to every " +
			"window of every subject it names, in one transaction; then writes those windows' totals to Redis before " +
			"answering, and the totals of the windows a duplicate counts for too, so sending a batch again repairs " +
			"Redis. Each event is decoded and checked on its own: one that fails is answered in `rejected` with its " +
			"index in the batch and its reason, and the rest are recorded. An event accepted or a duplicate is recorded, " +
			"and the producer may forget it; `mirrored` says whether Redis took the totals. Sending a batch again is " +
			"safe. At most " + strconv.Itoa(usage.MaxEvents) + " events and 2 MiB, else 413; a body that is not JSON " +
			"with an `events` array is a 422; 503 when the database fails, and the producer sends the whole batch again.",
		Tags:             []string{tag},
		Security:         security,
		MaxBodyBytes:     usage.MaxBodyBytes,
		SkipValidateBody: true,
		Errors:           []int{http.StatusUnauthorized, http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity, http.StatusServiceUnavailable},
	}, s.ingest)
	huma.Register(api, huma.Operation{
		OperationID: "readUsageWindows",
		Method:      http.MethodPost,
		Path:        usage.PathWindows,
		Summary:     "Read what subjects spent in their current windows",
		Description: "Each subject's current daily, weekly, bi-weekly and monthly window from the database, a window nobody " +
			"has spent in at zero, and writes them to Redis, so a key that was flushed or evicted is back after one read. " +
			"At most " + strconv.Itoa(usage.MaxSubjects) + " subjects, else 413.",
		Tags:     []string{tag},
		Security: security,
		Errors:   []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusRequestEntityTooLarge, http.StatusServiceUnavailable},
	}, s.windows)
	huma.Register(api, huma.Operation{
		OperationID: "readStandings",
		Method:      http.MethodPost,
		Path:        usage.PathStandings,
		Summary:     "Read organizations' standings",
		Description: "One standing per organization asked about, in the order asked, each on the tier asked about: its " +
			"tier's settings (the `default` tier's when the tier has no row, with `fallback_tier`), its own overrides " +
			"and the settings in force, its status, what was granted and spent, its balance, and its spend in the " +
			"current UTC day and calendar month. Read from the database every time, so a change shows in the next " +
			"read. An organization with no account is active with nothing granted or spent. At most " +
			strconv.Itoa(usage.MaxOrganizations) + " organizations, else 413.",
		Tags:     []string{tag},
		Security: security,
		Errors:   []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusRequestEntityTooLarge, http.StatusServiceUnavailable},
	}, s.standings)
	s.registerAccounts(api)
	s.registerManage(api)
	return api
}

// countRefused counts the batches answered 4xx, whose events are lost until the producer
// ships them right.
func countRefused(ctx huma.Context, next func(huma.Context)) {
	next(ctx)
	if status := ctx.Status(); ctx.Operation().OperationID == ingestOperation && status >= 400 && status < 500 {
		BatchesRefused.WithLabelValues(strconv.Itoa(status)).Inc()
	}
}

// dropBodyEcho drops the request body Huma repeats in an error about it: a batch is up to
// 2 MiB, which the producer already has.
func dropBodyEcho(_ huma.Context, _ string, v any) (any, error) {
	if model, ok := v.(*huma.ErrorModel); ok {
		for _, detail := range model.Errors {
			if detail != nil && detail.Location == "body" {
				detail.Value = nil
			}
		}
	}
	return v, nil
}

// requireToken answers 401 to an operation that takes the ingest token when the request does
// not carry it, before its body is read, and hands a management operation to
// requireManageKey: neither credential opens the other's operations.
func (s *Server) requireToken(api huma.API) func(huma.Context, func(huma.Context)) {
	return func(ctx huma.Context, next func(huma.Context)) {
		security := ctx.Operation().Security
		if len(security) == 0 {
			next(ctx)
			return
		}
		if _, manage := security[0][manageScheme]; manage {
			s.requireManageKey(ctx, next)
			return
		}
		sent, ok := strings.CutPrefix(ctx.Header("Authorization"), "Bearer ")
		// Hashed first, so the comparison takes as long whatever the length sent.
		digest := sha256.Sum256([]byte(sent))
		if !ok || subtle.ConstantTimeCompare(digest[:], s.token[:]) != 1 {
			_ = huma.WriteErr(api, ctx, http.StatusUnauthorized, "the usage service needs its ingest token: Authorization: Bearer <USAGE_INGEST_TOKEN>")
			return
		}
		next(ctx)
	}
}

type healthOutput struct {
	Body struct {
		Status string `json:"status" doc:"ok"`
	}
}

func (s *Server) health(ctx context.Context, _ *struct{}) (*healthOutput, error) {
	ctx, cancel := context.WithTimeout(ctx, healthTimeout)
	defer cancel()
	if err := s.ledger.Ping(ctx); err != nil {
		return nil, huma.Error503ServiceUnavailable("the database does not answer")
	}
	output := &healthOutput{}
	output.Body.Status = "ok"
	return output, nil
}

type ingestInput struct {
	Body batch
}

// batch is a Batch as sent, each event decoded on its own, so an event with a field of the
// wrong type is rejected alone rather than failing the batch.
type batch struct {
	Events []json.RawMessage `json:"events"`
}

// Schema describes the body as the Batch it is.
func (batch) Schema(r huma.Registry) *huma.Schema {
	return r.Schema(reflect.TypeFor[usage.Batch](), true, "")
}

type ingestOutput struct {
	Body usage.Ack
}

func (s *Server) ingest(ctx context.Context, input *ingestInput) (*ingestOutput, error) {
	started := time.Now()
	defer func() { ingestSeconds.Observe(time.Since(started).Seconds()) }()
	sent := input.Body.Events
	if sent == nil {
		return nil, huma.Error422UnprocessableEntity("a batch is {\"events\": [...]}: events is missing or null")
	}
	if len(sent) > usage.MaxEvents {
		return nil, huma.Error413RequestEntityTooLarge(fmt.Sprintf("a batch carries at most %d events, this one %d", usage.MaxEvents, len(sent)))
	}
	eventsReceived.Add(float64(len(sent)))

	events := make([]usage.Event, 0, len(sent))
	// positions is each decoded event's index in the batch.
	positions := make([]int, 0, len(sent))
	var rejected []usage.Rejection
	for i, raw := range sent {
		var event usage.Event
		if err := json.Unmarshal(raw, &event); err != nil {
			rejected = append(rejected, malformed(i, raw, err))
			continue
		}
		events = append(events, event)
		positions = append(positions, i)
	}
	result, err := s.ledger.Ingest(ctx, events)
	if err != nil {
		IngestErrors.Inc()
		s.logger.Error("could not record a batch of usage events", "events", len(sent), "error", err)
		return nil, huma.Error503ServiceUnavailable("the batch was not recorded; send it again")
	}
	for _, rejection := range result.Rejected {
		rejection.Index = positions[rejection.Index]
		rejected = append(rejected, rejection)
	}
	slices.SortFunc(rejected, func(a, b usage.Rejection) int { return cmp.Compare(a.Index, b.Index) })
	result.Rejected = rejected
	s.rejected(sent, rejected)
	eventsWritten.Add(float64(len(result.Accepted)))
	eventsDuplicate.Add(float64(result.Duplicates))
	for _, event := range result.Accepted {
		if strings.HasPrefix(event.Source, routerSource) {
			EventLag.Observe(max(0, result.ReceivedAt.Sub(event.OccurredAt).Seconds()))
		}
	}
	return &ingestOutput{Body: result.Ack()}, nil
}

// malformed is the rejection of the event at index that does not decode, with its usage id
// when that much of it does.
func malformed(index int, raw json.RawMessage, err error) usage.Rejection {
	var id struct {
		UsageID uuid.UUID `json:"usage_id"`
	}
	_ = json.Unmarshal(raw, &id)
	return usage.Rejection{Index: index, UsageID: id.UsageID, Reason: usage.ReasonMalformed, Message: err.Error()}
}

// rejected counts and logs, as sent, every event refused in a batch recorded: each is revenue
// lost until its producer is fixed and ships it again. An event is logged up to
// maxLoggedEvent bytes, since one may be most of a 2 MiB body.
func (s *Server) rejected(sent []json.RawMessage, rejections []usage.Rejection) {
	for _, rejection := range rejections {
		EventsRejected.WithLabelValues(rejection.Reason).Inc()
		event := string(sent[rejection.Index])
		if len(event) > maxLoggedEvent {
			event = fmt.Sprintf("%s... (truncated: %d bytes in all)", event[:maxLoggedEvent], len(event))
		}
		s.logger.Error("usage event rejected", "reason", rejection.Reason, "message", rejection.Message,
			"index", rejection.Index, "usage_id", rejection.UsageID, "event", event)
	}
}

type windowsInput struct {
	Body usage.WindowsQuery
}

type windowsOutput struct {
	Body usage.Windows
}

func (s *Server) windows(ctx context.Context, input *windowsInput) (*windowsOutput, error) {
	refs := input.Body.Subjects
	if len(refs) > usage.MaxSubjects {
		return nil, huma.Error413RequestEntityTooLarge(fmt.Sprintf("a read names at most %d subjects, this one %d", usage.MaxSubjects, len(refs)))
	}
	for i, ref := range refs {
		if problem := ref.Problem(); problem != "" {
			return nil, huma.Error400BadRequest(fmt.Sprintf("subjects[%d]: %s", i, problem))
		}
	}
	windowReads.Add(float64(len(refs)))
	asOf := s.now().UTC()
	windows, err := s.ledger.Current(ctx, refs)
	if err != nil {
		s.logger.Error("could not read usage windows", "subjects", len(refs), "error", err)
		return nil, huma.Error503ServiceUnavailable("the windows could not be read")
	}
	output := &windowsOutput{Body: usage.Windows{AsOf: asOf, Windows: make([]usage.WindowSpend, 0, len(windows))}}
	for _, window := range windows {
		output.Body.Windows = append(output.Body.Windows, window.Spend())
	}
	return output, nil
}

type standingsInput struct {
	Body usage.StandingQuery
}

type standingsOutput struct {
	Body usage.Standings
}

func (s *Server) standings(ctx context.Context, input *standingsInput) (*standingsOutput, error) {
	started := time.Now()
	defer func() { standingSeconds.Observe(time.Since(started).Seconds()) }()
	organizations := input.Body.Organizations
	if len(organizations) > usage.MaxOrganizations {
		return nil, huma.Error413RequestEntityTooLarge(fmt.Sprintf("a read names at most %d organizations, this one %d", usage.MaxOrganizations, len(organizations)))
	}
	for i, organization := range organizations {
		if organization.OrganizationID == "" {
			return nil, huma.Error400BadRequest(fmt.Sprintf("organizations[%d]: organization_id is required", i))
		}
	}
	asOf := s.now().UTC()
	standings, err := s.ledger.Standings(ctx, organizations)
	if err != nil {
		s.logger.Error("could not read standings", "organizations", len(organizations), "error", err)
		return nil, huma.Error503ServiceUnavailable("the standings could not be read")
	}
	return &standingsOutput{Body: usage.Standings{AsOf: asOf, Standings: standings}}, nil
}

// Spec renders the service's OpenAPI document, which internal/usage/server/openapi.yaml
// holds. It is apart from the router's: no SDK is generated from it.
func Spec() ([]byte, error) {
	rendered, err := (&Server{}).register(chi.NewRouter()).OpenAPI().Downgrade()
	if err != nil {
		return nil, fmt.Errorf("server: spec: %w", err)
	}
	var document map[string]any
	decoder := json.NewDecoder(bytes.NewReader(rendered))
	// Numbers stay as written rather than turning into floats yaml prints as 1e+08.
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("server: spec: %w", err)
	}
	numbers(document)
	var out bytes.Buffer
	out.WriteString("# Generated by `go run ./cmd/usage openapi` from the operations in internal/usage/server. Do not edit.\n")
	encoder := yaml.NewEncoder(&out)
	encoder.SetIndent(2)
	if err := encoder.Encode(document); err != nil {
		return nil, fmt.Errorf("server: spec: %w", err)
	}
	if err := encoder.Close(); err != nil {
		return nil, fmt.Errorf("server: spec: %w", err)
	}
	return out.Bytes(), nil
}

// numbers turns every json.Number in the document into an int64, or a float64 when it is not
// whole, since yaml quotes a json.Number and `minimum: "0"` is a string, not a bound.
func numbers(value any) any {
	switch value := value.(type) {
	case map[string]any:
		for key, child := range value {
			value[key] = numbers(child)
		}
	case []any:
		for i, child := range value {
			value[i] = numbers(child)
		}
	case json.Number:
		if integer, err := value.Int64(); err == nil {
			return integer
		}
		float, _ := value.Float64()
		return float
	}
	return value
}
