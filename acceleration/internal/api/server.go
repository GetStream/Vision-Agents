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

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
)

// CustomerHeader names the tenant directly. It is only read in noauth mode, where a proxy
// in front of the router has already decided who the caller is, and it is what a local
// deployment with no proxy and no keys uses.
const CustomerHeader = auth.CustomerHeader

// CustomerParam carries the same identifier on the sockets, because the browser WebSocket
// API cannot set a header.
const CustomerParam = auth.CustomerParam

// customerContextKey holds the customer identifier extracted from the request.
type customerContextKey struct{}

// organizationContextKey holds the organization the customer belongs to, which is what a
// rate limit and a bill are counted against.
type organizationContextKey struct{}

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
	// Simulations puts an agent through a conversation somebody wrote down and rules on
	// how it went. Absent without sessions or model routing, in which case a simulation
	// can be written down but not run.
	Simulations *simulation.Runner
	// Knowledge fills the bases a config's knowledge_namespace has an agent read from.
	// Absent when the deployment has no knowledge provider, in which case there is nothing
	// to fill and the path says so.
	Knowledge knowledge.Writer
	// KnowledgeURLs keeps those bases filled from pages published elsewhere. Absent
	// without a database or without something that can read a page, in which case the url
	// paths say so rather than accepting a subscription nothing would honour.
	KnowledgeURLs *urls.Service
	// Voices holds the voices customers brought with them. Absent when the deployment has
	// no object storage, in which case there is nowhere to keep a recording and the voice
	// paths say so.
	Voices *voices.Service
	// Dispatch holds the workers waiting to answer inbound calls. Absent when nothing is
	// meant to answer a phone, in which case the dispatch socket says so rather than
	// accepting a worker whose calls would never arrive.
	Dispatch *dispatch.Pool
	// StreamSecret signs the call events Stream sends. Without it the webhook refuses
	// every request, because an unsigned webhook is anyone who found the URL. It also
	// mints the tokens a browser joins a call with, which is why it never leaves here.
	StreamSecret string
	// StreamKey names the Stream app those tokens are for. A browser needs it to join,
	// so unlike the secret it is meant to be handed out.
	StreamKey string
	// CORSOrigins are the browser origins allowed to call this API directly, which is
	// what a dashboard talking to the router without a proxy in between needs. Empty
	// means no browser may, which is right for a deployment only servers reach.
	CORSOrigins []string
	// PublicURL is where this process is reachable, which plugin OAuth callbacks need.
	PublicURL string
	// DashboardURL is where a finished plugin login sends the browser.
	DashboardURL string
	// Auth decides who a request is from. Absent means noauth, which trusts the customer
	// header and is right for a local deployment and for one behind a proxy that has
	// already authenticated the caller.
	Auth   auth.Authenticator
	Logger *slog.Logger
}

// Server implements the generated StrictServerInterface.
type Server struct {
	routers       map[routing.Modality]routing.Inspector
	store         *store.Store
	live          *live.Client
	phone         *phone.Service
	sessions      *session.Manager
	streams       *Streams
	transcripts   *chatlog.Reader
	campaigns     *campaign.Runner
	simulations   *simulation.Runner
	knowledge     knowledge.Writer
	pages         *urls.Service
	voices        *voices.Service
	dispatch      *dispatch.Pool
	streamSecret  string
	streamKey     string
	corsOrigins   []string
	publicURL     string
	dashboardURL  string
	oauth         *plugins.Auth
	authenticator auth.Authenticator
	upgrader      websocket.Upgrader
	logger        *slog.Logger
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

	authenticator := options.Auth
	if authenticator == nil {
		// A deployment that names no mode is a local one, where the customer header is
		// the whole of the story.
		var err error
		if authenticator, err = auth.New(auth.NoAuth, nil); err != nil {
			return nil, err
		}
	}

	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{
		routers:       options.Routers,
		store:         options.Store,
		live:          options.Live,
		phone:         options.Phone,
		sessions:      options.Sessions,
		streams:       options.Streams,
		transcripts:   options.Transcripts,
		campaigns:     options.Campaigns,
		simulations:   options.Simulations,
		knowledge:     options.Knowledge,
		pages:         options.KnowledgeURLs,
		voices:        options.Voices,
		dispatch:      options.Dispatch,
		streamSecret:  options.StreamSecret,
		streamKey:     options.StreamKey,
		corsOrigins:   options.CORSOrigins,
		publicURL:     options.PublicURL,
		dashboardURL:  options.DashboardURL,
		authenticator: authenticator,
		upgrader:      newUpgrader(options.CORSOrigins),
		oauth: &plugins.Auth{
			PublicURL:    options.PublicURL,
			DashboardURL: options.DashboardURL,
		},
		logger: logger,
	}, nil
}

// Handler returns the HTTP handler for the whole API.
//
// The three sockets, the answer host and the call hook are registered first, on a mux the
// generated routes are then added to. The sockets are excluded from generation because a
// strict server returns a response object and an upgrade returns a connection, so there is
// nothing for it to hand back. The answer host is excluded because it serves a vendor's XML
// rather than this API's JSON, and the call hook because both are reached by somebody other
// than a customer: a telephony vendor and Stream.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", s.watchSession)
	mux.HandleFunc("GET /v1/{modality}/stream", s.streamModality)
	mux.HandleFunc("GET /v1/dispatch", s.dispatchCalls)
	mux.HandleFunc("GET /v1/phone/answer/{token}", s.answerPhoneCall)
	mux.HandleFunc("POST /v1/phone/answer/{token}", s.answerPhoneCall)
	mux.HandleFunc("POST "+phone.CallHookPath, s.receiveCallEvent)
	mux.HandleFunc("GET "+plugins.CallbackPath, s.finishPluginLogin)
	return withCORS(s.corsOrigins, s.withCustomer(HandlerFromMux(NewStrictHandler(s, nil), mux)))
}

// withCustomer lifts the authenticated principal into the request context so handlers can
// read it without each one reaching into the raw request.
//
// A request that does not authenticate is passed along without one rather than refused
// here. Every handler that needs a customer already reports a 401 when there is none, and
// the paths that legitimately have no customer — the health check, a vendor fetching a call
// plan, the hook Stream signs — are reached by somebody who has no key to present. Failing
// here instead would mean keeping a list of the exceptions in two places.
//
// It also means one 401 for every reason authentication failed. A caller that could tell an
// unknown key from a bad token could use the difference to find out which keys exist.
func (s *Server) withCustomer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		principal, err := s.authenticator.Authenticate(r.Context(), r)
		if err == nil && principal.AppID != "" {
			ctx := context.WithValue(r.Context(), customerContextKey{}, principal.AppID)
			ctx = context.WithValue(ctx, organizationContextKey{}, principal.OrganizationID)
			r = r.WithContext(ctx)
		}
		next.ServeHTTP(w, r)
	})
}

// withCORS lets a browser at the API from the origins the deployment named.
//
// It exists for the dashboard, which talks to the router directly rather than through a
// server of its own: an extra hop would only be there to move a header, and the router is
// already the thing that decides who may read a call.
func withCORS(allowed []string, next http.Handler) http.Handler {
	if len(allowed) == 0 {
		return next
	}
	permitted := make(map[string]struct{}, len(allowed))
	for _, origin := range allowed {
		permitted[strings.TrimSpace(origin)] = struct{}{}
	}
	_, anywhere := permitted["*"]

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		_, named := permitted[origin]
		if origin != "" && (named || anywhere) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Headers", CustomerHeader+", Content-Type")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Max-Age", "600")
		}
		// A preflight asks whether the real request would be allowed and carries nothing
		// worth routing, so it is answered here rather than by a handler that would only
		// report that nothing serves OPTIONS.
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// CustomerFrom returns the customer identifier carried by the request.
func CustomerFrom(ctx context.Context) (string, bool) {
	customerID, ok := ctx.Value(customerContextKey{}).(string)
	return customerID, ok && customerID != ""
}

// OrganizationFrom returns the organization the request's customer belongs to. It is empty
// in a deployment that names a customer without naming an organization.
func OrganizationFrom(ctx context.Context) string {
	organizationID, _ := ctx.Value(organizationContextKey{}).(string)
	return organizationID
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
