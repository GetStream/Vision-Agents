// Package api serves the router's HTTP surface. The types and routing in generated.go
// come from api/openapi.yaml, which is the source of truth: change the spec and
// regenerate rather than editing generated.go.
//
// Every routing path is scoped by modality. The server holds one router per modality it
// serves and looks the right one up per request, so adding a modality is a matter of
// passing another router in.
package api

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// CustomerHeader carries the trusted customer identifier. Real authentication is not part
// of this version, so the header is taken at face value.
const CustomerHeader = "X-Customer-Id"

// customerContextKey holds the customer identifier extracted from the request.
type customerContextKey struct{}

// Options configures a Server. The store and live client are optional; endpoints that
// need them report the dependency as unavailable rather than panicking.
type Options struct {
	// Routers is the router serving each modality. A modality that is absent is a 404.
	Routers map[routing.Modality]routing.Inspector
	Store   *store.Store
	Live    *live.Client
	// Phone serves the telephony paths. Absent when the deployment has no vendors, in
	// which case those paths say so rather than pretending numbers can be bought.
	Phone *phone.Service
	// Sessions runs conversations. Absent when the deployment only inspects routing, in
	// which case the session paths report that there are none rather than 500ing.
	Sessions *session.Manager
	// Streams serves the per-modality sockets, for callers running their own pipeline.
	// Absent when the deployment routes nothing itself.
	Streams *Streams
	// Transcripts reads back what was said on a call. Absent when the deployment has no
	// chat credentials, in which case nothing was written down to read.
	Transcripts *chatlog.Reader
	// Campaigns rings lists of people. Absent without telephony or sessions, in which
	// case a campaign can be written down but not run.
	Campaigns *campaign.Runner
	Logger    *slog.Logger
}

// Server implements the generated StrictServerInterface.
type Server struct {
	routers     map[routing.Modality]routing.Inspector
	store       *store.Store
	live        *live.Client
	phone       *phone.Service
	sessions    *session.Manager
	streams     *Streams
	transcripts *chatlog.Reader
	campaigns   *campaign.Runner
	logger      *slog.Logger
}

// NewServer wires the handlers.
func NewServer(options Options) (*Server, error) {
	if len(options.Routers) == 0 {
		return nil, errors.New("api: at least one router is required")
	}
	for modality, router := range options.Routers {
		if router == nil {
			return nil, errors.New("api: router for " + string(modality) + " is nil")
		}
	}

	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{
		routers:     options.Routers,
		store:       options.Store,
		live:        options.Live,
		phone:       options.Phone,
		sessions:    options.Sessions,
		streams:     options.Streams,
		transcripts: options.Transcripts,
		campaigns:   options.Campaigns,
		logger:      logger,
	}, nil
}

// Handler returns the HTTP handler for the whole API.
//
// The two sockets are registered first, on a mux the generated routes are then added to.
// They are excluded from generation because a strict server returns a response object and
// an upgrade returns a connection, so there is nothing for it to hand back.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", s.watchSession)
	mux.HandleFunc("GET /v1/{modality}/stream", s.streamModality)
	return withCustomer(HandlerFromMux(NewStrictHandler(s, nil), mux))
}

// withCustomer lifts the trusted customer header into the request context so handlers can
// read it without each one reaching into the raw request.
func withCustomer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		customerID := strings.TrimSpace(r.Header.Get(CustomerHeader))
		if customerID != "" {
			r = r.WithContext(context.WithValue(r.Context(), customerContextKey{}, customerID))
		}
		next.ServeHTTP(w, r)
	})
}

// CustomerFrom returns the customer identifier carried by the request.
func CustomerFrom(ctx context.Context) (string, bool) {
	customerID, ok := ctx.Value(customerContextKey{}).(string)
	return customerID, ok && customerID != ""
}

// routerFor returns the router serving a modality, or false when this deployment does not
// serve it.
func (s *Server) routerFor(modality Modality) (routing.Inspector, bool) {
	router, ok := s.routers[routing.Modality(modality)]
	return router, ok
}

// GetHealth reports whether the router and its dependencies are usable.
func (s *Server) GetHealth(ctx context.Context, _ GetHealthRequestObject) (GetHealthResponseObject, error) {
	dependencies := map[string]string{}
	healthy := true

	if s.store == nil {
		dependencies["postgres"] = "not configured"
	} else if err := s.store.Ping(ctx); err != nil {
		dependencies["postgres"] = err.Error()
		healthy = false
	} else {
		dependencies["postgres"] = "ok"
	}

	if s.live == nil {
		dependencies["redis"] = "not configured"
	} else if err := s.live.Ping(ctx); err != nil {
		dependencies["redis"] = err.Error()
		healthy = false
	} else {
		dependencies["redis"] = "ok"
	}

	for modality := range s.routers {
		dependencies[string(modality)] = "ok"
	}

	if !healthy {
		return GetHealth503JSONResponse{Status: Degraded, Dependencies: dependencies}, nil
	}
	return GetHealth200JSONResponse{Status: Ok, Dependencies: dependencies}, nil
}

// ListProviders returns the providers configured for a modality and their live health.
func (s *Server) ListProviders(ctx context.Context, request ListProvidersRequestObject) (ListProvidersResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return ListProviders401JSONResponse{missingCustomer()}, nil
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return ListProviders404JSONResponse{unknownModality(request.Modality)}, nil
	}

	candidates := router.Providers(ctx)
	providers := make([]Provider, 0, len(candidates))
	for _, candidate := range candidates {
		providers = append(providers, Provider{
			Provider:  candidate.Config.Provider,
			Model:     candidate.Config.Model,
			Languages: candidate.Config.Languages,
			Realtime:  candidate.Config.Realtime,
			Tier:      tierOf(candidate.Config),
			Health:    providerHealth(candidate.Health),
		})
	}
	return ListProviders200JSONResponse(providers), nil
}

// ResolveTarget explains which providers would serve a target, best first.
func (s *Server) ResolveTarget(ctx context.Context, request ResolveTargetRequestObject) (ResolveTargetResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return ResolveTarget401JSONResponse{missingCustomer()}, nil
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return ResolveTarget404JSONResponse{unknownModality(request.Modality)}, nil
	}

	var languageHints []string
	if request.Params.Language != nil {
		languageHints = *request.Params.Language
	}

	candidates, err := router.Resolve(ctx, request.Target, languageHints)
	if err != nil {
		return ResolveTarget404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}

	resolved := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		resolved = append(resolved, Candidate{
			Provider: candidate.Config.Provider,
			Model:    candidate.Config.Model,
			Health:   providerHealth(candidate.Health),
		})
	}
	return ResolveTarget200JSONResponse(resolved), nil
}

// GetStats returns the calling customer's aggregated usage for one modality.
func (s *Server) GetStats(ctx context.Context, request GetStatsRequestObject) (GetStatsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetStats401JSONResponse{missingCustomer()}, nil
	}
	// Statistics are not limited to the routed modalities: memory and phone are recorded
	// the same way and cost the same customer money.
	if !request.Params.To.After(request.Params.From) {
		return GetStats400JSONResponse{badRequest("to must be after from")}, nil
	}
	tags, err := parseTagFilter(request.Params.Tag)
	if err != nil {
		return GetStats400JSONResponse{badRequest(err.Error())}, nil
	}
	if s.store == nil {
		return GetStats400JSONResponse{badRequest("statistics are not available: no database configured")}, nil
	}

	granularity := granularityOf(request.Params.Granularity)
	buckets, err := s.store.CustomerStats(
		ctx, string(request.Modality), customerID, granularity, request.Params.From, request.Params.To, tags)
	if err != nil {
		return nil, err
	}

	stats := make([]StatsBucket, 0, len(buckets))
	for _, bucket := range buckets {
		stats = append(stats, StatsBucket{
			Provider:               bucket.Provider,
			Model:                  bucket.Model,
			Bucket:                 bucket.Bucket,
			AudioMsTotal:           bucket.AudioMsTotal,
			CharactersTotal:        bucket.CharactersTotal,
			InputTokensTotal:       bucket.InputTokensTotal,
			CachedInputTokensTotal: bucket.CachedInputTokensTotal,
			OutputTokensTotal:      bucket.OutputTokensTotal,
			CostMicrosTotal:        bucket.CostMicrosTotal,
			RequestCount:           bucket.RequestCount,
			ErrorCount:             bucket.ErrorCount,
			LatencyP50Ms:           bucket.LatencyP50Ms,
			LatencyP95Ms:           bucket.LatencyP95Ms,
			Uptime:                 bucket.Uptime,
		})
	}
	return GetStats200JSONResponse(stats), nil
}

// GetTagStats returns the calling customer's usage broken down by one cost label.
func (s *Server) GetTagStats(ctx context.Context, request GetTagStatsRequestObject) (GetTagStatsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetTagStats401JSONResponse{missingCustomer()}, nil
	}
	if !request.Params.To.After(request.Params.From) {
		return GetTagStats400JSONResponse{badRequest("to must be after from")}, nil
	}
	if s.store == nil {
		return GetTagStats400JSONResponse{badRequest("statistics are not available: no database configured")}, nil
	}

	granularity := granularityOf(request.Params.Granularity)
	buckets, err := s.store.CustomerTagStats(ctx, string(request.Modality), customerID,
		request.Params.Key, granularity, request.Params.From, request.Params.To)
	if err != nil {
		return GetTagStats400JSONResponse{badRequest(err.Error())}, nil
	}

	stats := make([]TagStatsBucket, 0, len(buckets))
	for _, bucket := range buckets {
		stats = append(stats, TagStatsBucket{
			TagKey:                 bucket.TagKey,
			TagValue:               bucket.TagValue,
			Bucket:                 bucket.Bucket,
			AudioMsTotal:           bucket.AudioMsTotal,
			CharactersTotal:        bucket.CharactersTotal,
			InputTokensTotal:       bucket.InputTokensTotal,
			CachedInputTokensTotal: bucket.CachedInputTokensTotal,
			OutputTokensTotal:      bucket.OutputTokensTotal,
			CostMicrosTotal:        bucket.CostMicrosTotal,
			RequestCount:           bucket.RequestCount,
			ErrorCount:             bucket.ErrorCount,
			LatencyP50Ms:           bucket.LatencyP50Ms,
			LatencyP95Ms:           bucket.LatencyP95Ms,
			Uptime:                 bucket.Uptime,
		})
	}
	return GetTagStats200JSONResponse(stats), nil
}

// GetTurnStats returns the calling customer's conversational latency.
func (s *Server) GetTurnStats(ctx context.Context, request GetTurnStatsRequestObject) (GetTurnStatsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetTurnStats401JSONResponse{missingCustomer()}, nil
	}
	if !request.Params.To.After(request.Params.From) {
		return GetTurnStats400JSONResponse{badRequest("to must be after from")}, nil
	}
	if s.store == nil {
		return GetTurnStats400JSONResponse{badRequest("statistics are not available: no database configured")}, nil
	}

	var agentID string
	if request.Params.AgentId != nil {
		agentID = *request.Params.AgentId
	}

	granularity := granularityOf(request.Params.Granularity)
	buckets, err := s.store.CustomerTurnStats(
		ctx, customerID, agentID, granularity, request.Params.From, request.Params.To)
	if err != nil {
		return nil, err
	}

	stats := make([]TurnStatsBucket, 0, len(buckets))
	for _, bucket := range buckets {
		stats = append(stats, TurnStatsBucket{
			AgentId:          bucket.AgentID,
			Bucket:           bucket.Bucket,
			TurnCount:        bucket.TurnCount,
			InterruptedCount: bucket.InterruptedCount,
			AudioOutMsTotal:  bucket.AudioOutMsTotal,
			SttLatencyP50Ms:  bucket.STTLatencyP50Ms,
			SttLatencyP95Ms:  bucket.STTLatencyP95Ms,
			LlmTtftP50Ms:     bucket.LLMTTFTP50Ms,
			LlmTtftP95Ms:     bucket.LLMTTFTP95Ms,
			TtsTtfbP50Ms:     bucket.TTSTTFBP50Ms,
			TtsTtfbP95Ms:     bucket.TTSTTFBP95Ms,
			RoundtripP50Ms:   bucket.RoundtripP50Ms,
			RoundtripP95Ms:   bucket.RoundtripP95Ms,
			RoundtripP99Ms:   bucket.RoundtripP99Ms,
		})
	}
	return GetTurnStats200JSONResponse(stats), nil
}

// RunRollup aggregates request rows into a rollup table.
func (s *Server) RunRollup(ctx context.Context, request RunRollupRequestObject) (RunRollupResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return RunRollup401JSONResponse{missingCustomer()}, nil
	}
	if request.Body == nil {
		return RunRollup400JSONResponse{badRequest("a request body is required")}, nil
	}
	if !request.Body.To.After(request.Body.From) {
		return RunRollup400JSONResponse{badRequest("to must be after from")}, nil
	}
	if s.store == nil {
		return RunRollup400JSONResponse{badRequest("rollups are not available: no database configured")}, nil
	}

	granularity := granularityOf(request.Body.Granularity)
	written, err := s.store.Rollup(ctx, granularity, request.Body.From, request.Body.To)
	if err != nil {
		return nil, err
	}

	return RunRollup200JSONResponse{
		Granularity:    Granularity(granularity),
		BucketsWritten: written,
	}, nil
}

// parseTagFilter turns repeated "key:value" query parameters into a label filter. A tag
// key never contains a colon, so the first one separates the two.
func parseTagFilter(raw *[]string) (map[string]string, error) {
	if raw == nil || len(*raw) == 0 {
		return nil, nil
	}

	tags := make(map[string]string, len(*raw))
	for _, entry := range *raw {
		key, value, found := strings.Cut(entry, ":")
		if !found || key == "" {
			return nil, fmt.Errorf("tag %q must be written key:value", entry)
		}
		tags[key] = value
	}
	return tags, nil
}

// granularityOf defaults to hourly, matching the spec.
func granularityOf(requested *Granularity) store.Granularity {
	if requested != nil && *requested == Daily {
		return store.Daily
	}
	return store.Hourly
}

// tierOf reports the effective tier, which is low-latency for a model that declares none.
func tierOf(config routing.ProviderConfig) Tier {
	if config.Tier == routing.HighQuality {
		return HighQuality
	}
	return LowLatency
}

func providerHealth(health live.Health) ProviderHealth {
	return ProviderHealth{
		Available:    health.Available,
		Requests:     health.Requests,
		Errors:       health.Errors,
		ErrorRate:    health.ErrorRate(),
		LatencyMsAvg: health.LatencyMsAvg,
	}
}

func missingCustomer() UnauthorizedJSONResponse {
	return UnauthorizedJSONResponse{Error: "the " + CustomerHeader + " header is required"}
}

func unknownModality(modality Modality) NotFoundJSONResponse {
	return NotFoundJSONResponse{Error: "this deployment does not route " + string(modality)}
}

func badRequest(message string) BadRequestJSONResponse {
	return BadRequestJSONResponse{Error: message}
}
