// Package api serves the router's HTTP surface on a chi router. Operations are declared
// in Go with Huma, and the Go structs are the source of truth: api/openapi.yaml is
// rendered from them by cmd/openapi. The operations not yet moved to Go are still
// generated into generated.go from api/legacy.yaml; change that file and regenerate
// rather than editing generated.go.
//
// Every routing path is scoped by modality. The server holds one router per modality it
// serves and looks the right one up per request, so adding a modality is a matter of
// passing another router in.
package api

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"mime"
	"net"
	"net/http"
	"net/netip"
	"runtime/debug"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	sentryhttp "github.com/getsentry/sentry-go/http"
	"github.com/go-chi/chi/v5"
	"github.com/gorilla/websocket"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/trace"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
)

// tracer records the spans this package opens around work a route pattern does not say.
var tracer = tracing.Tracer("api")

// CustomerHeader names the tenant directly, with no organization around it. It is what a
// local deployment with no proxy and no keys uses, and it is read in noauth and proxy
// modes and ignored entirely in api_key mode.
const CustomerHeader = auth.CustomerHeader

// CustomerParam carries the same identifier on the sockets, because the browser WebSocket
// API cannot set a header.
const CustomerParam = auth.CustomerParam

// customerContextKey holds the customer identifier extracted from the request.
type customerContextKey struct{}

// organizationContextKey holds the organization the customer belongs to, which is what a
// rate limit and a bill are counted against.
type organizationContextKey struct{}

// serverSideContextKey holds whether the caller is a process the customer runs.
type serverSideContextKey struct{}

// callerContextKey holds the end user the request is for and where they made it from, which
// is what a daily limit is counted against.
type callerContextKey struct{}

// kindContextKey holds what sort of caller it is, which is what qualifies the end user's
// name when one person's sessions are kept from another's.
type kindContextKey struct{}

// clientAccessibleExtension is what the spec marks the few operations an end user's device
// may reach with. Everything else is server-side only, and the check reads the mark from
// the operations themselves rather than from a list kept here, so what a generated SDK documents
// and what the server refuses cannot drift apart.
//
// The default is that way round because the two mistakes do not cost the same. An
// operation nobody thought about is refused to a browser, which arrives as a bug report;
// under the old default it was served to one, which arrives as a breach.
const clientAccessibleExtension = "x-client-accessible"

// Options configures a Server. The store and live client are optional; endpoints that
// need them report the dependency as unavailable rather than panicking.
type Options struct {
	// Routers is the router serving each modality. A modality that is absent is a 404.
	Routers map[routing.Modality]routing.Inspector
	Store   *store.Store
	// Configs reads and writes the tenant's configuration -- keys, policies, agent and
	// router configs, voices -- through whatever cache is in front of Postgres. Built
	// over Store when it is not given, in which case every read is a query.
	Configs *appconfig.Store
	Live    *live.Client
	// Phone serves the telephony paths. Absent when the deployment has no vendors, in
	// which case those paths say so rather than pretending numbers can be bought.
	Phone *phone.Service
	// Sessions runs conversations. Absent when the deployment only inspects routing, in
	// which case the session paths report that there are none rather than 500ing.
	Sessions *session.Manager
	// Relay reaches the sessions the other nodes of this deployment are running, so a
	// watcher's socket need not land on the node holding the conversation. Absent when
	// there is no Redis to carry it, in which case this node is the whole deployment as
	// far as a socket is concerned.
	Relay *relay.Bus
	// Directory says which node of this deployment is running which session, so a request
	// only that node can answer is carried to it rather than answered with a 404. Absent
	// when there is no Redis to keep it in, or no address this node's peers reach it at.
	Directory *node.Directory
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
	// VoiceLibrary reads the voices the speech providers themselves offer. Absent when no
	// provider that publishes one has a key here, in which case the library path says so
	// rather than answering with an empty catalogue.
	VoiceLibrary *voices.Catalogue
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
	// AuthMode is how this deployment decided that, which a handler needs when the mode
	// itself is the answer: moving a customer's data is refused outright in noauth,
	// where the tenant is a header rather than something anybody proved. Empty means
	// noauth, matching Auth being absent.
	AuthMode auth.Mode
	// DataRetention is how long a customer moving away has to finish, which is how long
	// their changes are recorded for.
	DataRetention time.Duration
	// Auth decides who a request is from. Absent means noauth, which reads the customer
	// header and takes every caller for that customer's own backend. That is the right
	// default for a server built in code rather than from configuration — a test, or a
	// deployment embedding this package — because there the absence is deliberate, where
	// an unset environment variable is somebody who has not thought about it yet and
	// gets api_key instead.
	Auth auth.Authenticator
	// Quota caps what one end user may spend in a day. Absent means nothing is capped,
	// which is right for a deployment with no Redis to count in and for one whose callers
	// are all backends the customer runs.
	Quota *quota.Limiter
	// Policies holds each organization's and app's budget, data policy and prompt
	// injection setting. Absent without a database, in which case the policy paths say so.
	Policies *policy.Enforcer
	// Connectors holds the adapters connectors are built from. A custom connector may only
	// name a scheme registered here, so with none registered every custom one is refused.
	Connectors core.Registry
	// TrustedProxies are the ranges this deployment's own proxies sit in, and they decide
	// how much of X-Forwarded-For is believed when working out who a request is from.
	// Empty means none of it is, and the connection's own address is used.
	TrustedProxies []netip.Prefix
	Logger         *slog.Logger
}

// Server implements the generated StrictServerInterface.
type Server struct {
	routers       map[routing.Modality]routing.Inspector
	store         *store.Store
	configs       *appconfig.Store
	live          *live.Client
	phone         *phone.Service
	sessions      *session.Manager
	relayed       *relayed
	directory     *node.Directory
	forwarder     *node.Forwarder
	streams       *Streams
	transcripts   *chatlog.Reader
	campaigns     *campaign.Runner
	simulations   *simulation.Runner
	knowledge     knowledge.Writer
	pages         *urls.Service
	voices        *voices.Service
	library       *voices.Catalogue
	dispatch      *dispatch.Pool
	streamSecret  string
	streamKey     string
	corsOrigins   []string
	publicURL     string
	dashboardURL  string
	oauth         *plugins.Auth
	authenticator auth.Authenticator
	authMode      auth.Mode
	dataRetention time.Duration
	quota         *quota.Limiter
	policies      *policy.Enforcer
	connectors    core.Registry
	trusted       []netip.Prefix
	// serverSide matches the requests the spec marks server-side only. It holds no
	// handlers: what is registered on it is the patterns, and matching one is the answer.
	serverSide *http.ServeMux
	upgrader   websocket.Upgrader
	popularity *popularity
	logger     *slog.Logger
}

// Option adjusts the options a server is built from. It exists for the settings a
// deployment supplies as code rather than as configuration, which cannot be written in the
// struct a configuration file is decoded into.
type Option func(*Options)

// WithAuthenticator supplies an authenticator of the deployment's own, which is the whole
// of auth.Custom: the mode names an answer this module does not have, and this is where
// the answer arrives. Anything satisfying auth.Authenticator will do, and auth.Func makes
// one out of a function.
//
// It overrides whatever ROUTER_AUTH_MODE asked for, because a deployment that compiled an
// authenticator in meant it.
func WithAuthenticator(authenticator auth.Authenticator) Option {
	return func(options *Options) { options.Auth = authenticator }
}

// NewServer wires the handlers.
func NewServer(options Options, with ...Option) (*Server, error) {
	for _, option := range with {
		option(&options)
	}
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

	authMode := options.AuthMode
	if authMode == "" {
		authMode = auth.NoAuth
	}
	// A deployment that names no window still records changes for somebody moving, for
	// as long as the settings say by default.
	retention := options.DataRetention
	if retention <= 0 {
		retention = 7 * 24 * time.Hour
	}

	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	configs := options.Configs
	if configs == nil && options.Store != nil {
		var err error
		if configs, err = appconfig.New(appconfig.Options{Store: options.Store, Logger: logger}); err != nil {
			return nil, err
		}
	}
	server := &Server{
		routers:       options.Routers,
		store:         options.Store,
		configs:       configs,
		live:          options.Live,
		phone:         options.Phone,
		sessions:      options.Sessions,
		directory:     options.Directory,
		streams:       options.Streams,
		transcripts:   options.Transcripts,
		campaigns:     options.Campaigns,
		simulations:   options.Simulations,
		knowledge:     options.Knowledge,
		pages:         options.KnowledgeURLs,
		voices:        options.Voices,
		library:       options.VoiceLibrary,
		dispatch:      options.Dispatch,
		streamSecret:  options.StreamSecret,
		streamKey:     options.StreamKey,
		corsOrigins:   options.CORSOrigins,
		publicURL:     options.PublicURL,
		dashboardURL:  options.DashboardURL,
		authenticator: authenticator,
		authMode:      authMode,
		dataRetention: retention,
		quota:         options.Quota,
		policies:      options.Policies,
		connectors:    options.Connectors,
		trusted:       options.TrustedProxies,
		upgrader:      newUpgrader(options.CORSOrigins),
		oauth: &plugins.Auth{
			PublicURL:    options.PublicURL,
			DashboardURL: options.DashboardURL,
		},
		popularity: newPopularity(options.Store, logger),
		logger:     logger,
	}
	serverSide, err := serverSideRoutes(server.newAPI(chi.NewRouter()).OpenAPI())
	if err != nil {
		return nil, err
	}
	server.serverSide = serverSide

	// The subscriptions outlive every request, so they are held against the process
	// rather than against a context a handler brought with it. Closing the Redis client
	// is what ends them, which is what shutting the deployment down already does.
	if options.Relay != nil {
		if server.relayed, err = server.newRelayed(context.Background(), options.Relay); err != nil {
			return nil, fmt.Errorf("api: subscribe to the session relay: %w", err)
		}
	}
	if options.Directory != nil {
		server.forwarder = node.NewForwarder(logger)
	}
	return server, nil
}

// Handler returns the HTTP handler for the whole API.
//
// The three sockets, the answer host and the call hook are registered first, on a router
// the Huma operations and then the generated routes are added to. The sockets are excluded from generation because a
// strict server returns a response object and an upgrade returns a connection, so there is
// nothing for it to hand back. The answer host is excluded because it serves a vendor's XML
// rather than this API's JSON, and the call hook because both are reached by somebody other
// than a customer: a telephony vendor and Stream.
func (s *Server) Handler() http.Handler {
	mux := chi.NewRouter()
	mux.HandleFunc("GET /v1/agents/logs", s.listAgentLogs)
	mux.HandleFunc("GET /v1/agents/logs/stream", s.streamAgentLogs)
	mux.HandleFunc("GET /v1/agents/logs/{id}", s.getAgentLog)
	mux.HandleFunc("GET /v1/data/export", s.exportData)
	mux.HandleFunc("POST /v1/data/import", s.importData)
	mux.HandleFunc("GET /v1/data/changes", s.listDataChanges)
	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", s.watchSession)
	mux.HandleFunc("GET /v1/agents/socket", s.openSocketSession)
	mux.HandleFunc("GET /v1/{modality}/stream", s.streamModality)
	mux.HandleFunc("GET /v1/dispatch", s.dispatchCalls)
	mux.HandleFunc("GET /v1/phone/answer/{token}", s.answerPhoneCall)
	mux.HandleFunc("POST /v1/phone/answer/{token}", s.answerPhoneCall)
	mux.HandleFunc("POST "+phone.CallHookPath, s.receiveCallEvent)
	mux.HandleFunc("POST "+chat.MessageHookPath, s.receiveMessageEvent)
	mux.HandleFunc("GET "+plugins.CallbackPath, s.finishPluginLogin)
	s.newAPI(mux)
	handler := HandlerFromMux(NewStrictHandler(s, nil), mux)
	// Sentry is outermost so it sees panics from every middleware below it, not
	// only from the route handlers.
	//
	// Tracing sits directly inside it, so a span covers authentication and the quota
	// as well as the handler, which is the whole of what a caller waited for.
	//
	// Repanic is false, which is a change in behaviour worth knowing about: this
	// service had no recovery anywhere, so a panic in one request used to take
	// the process down, and the router runs as a single pod -- every call it was
	// carrying went with it. Answering that one request with a 500 and leaving
	// the rest connected is the better trade.
	//
	// WaitForDelivery is false because most of what is served here is a long-
	// lived socket; blocking the handler's return on event delivery would hold
	// the connection open past its use. The flush in cmd/router covers shutdown.
	instrumented := sentryhttp.New(sentryhttp.Options{
		Repanic:         false,
		WaitForDelivery: false,
	})
	served := instrumented.Handle(withTrace(withTiming(withCORS(s.corsOrigins,
		s.onOwningNode(s.withCustomer(s.withRequestLog(s.withQuota(s.withServerSide(handler)))))))))
	if s.directory == nil {
		return served
	}
	// Peers reach this node on the same port its callers do, so what they forward is
	// served beside everything else rather than on a listener of its own.
	return node.Serve(served)
}

// withTiming reports how long the server itself spent, so a caller timing a call can tell
// a slow API from a slow network rather than having to guess which it is looking at.
//
// It is said twice because the two readers are different. Server-Timing is what a
// browser's network panel reads with nothing taught to it, beside the time on the wire.
// The duration field is what the rest of Stream's API already answers with, so an SDK
// reads it the way it reads every other response.
//
// Outermost of our own middlewares, so the number covers authentication, the quota and
// the policies as well as the handler: all of it is time the caller waited.
func withTiming(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(&timedResponse{ResponseWriter: w, started: time.Now()}, r)
	})
}

// timedResponse stamps the header and names the duration in the body, both at the moment
// the answer begins rather than when it ends: what is reported is how long the caller
// waited to be answered, which for a streamed response is not how long the stream ran.
type timedResponse struct {
	http.ResponseWriter
	started time.Time
	stamped bool
	opened  bool
}

func (t *timedResponse) WriteHeader(code int) {
	t.stamp()
	t.ResponseWriter.WriteHeader(code)
}

func (t *timedResponse) Write(body []byte) (int, error) {
	if !t.opened {
		t.opened = true
		if opening, ok := t.opening(body); ok {
			t.stamp()
			if _, err := t.ResponseWriter.Write(opening); err != nil {
				return 0, err
			}
			// The brace the body starts with has just been written as part of the
			// opening, so what is left is everything after it. A short write is reported
			// as it happened; a complete one consumed the whole of what was handed in.
			written, err := t.ResponseWriter.Write(body[1:])
			if err != nil {
				return written, err
			}
			return len(body), nil
		}
	}
	t.stamp()
	return t.ResponseWriter.Write(body)
}

// Unwrap lets flushing, deadlines and a socket upgrade reach the writer underneath through
// http.ResponseController.
func (t *timedResponse) Unwrap() http.ResponseWriter { return t.ResponseWriter }

func (t *timedResponse) stamp() {
	if t.stamped {
		return
	}
	t.stamped = true
	t.ResponseWriter.Header().Set("Server-Timing", fmt.Sprintf("app;dur=%.2f", t.spent()))
}

// spent is how long the server has had the request, in milliseconds.
func (t *timedResponse) spent() float64 {
	return float64(time.Since(t.started).Microseconds()) / 1000
}

// opening returns what to write in place of the brace a JSON object starts with, naming
// the duration as its first field.
//
// Only a JSON object gets one: an array has nowhere to put it, and a stream, a recording
// and a socket are not documents. A response that declared its length is left alone too,
// since lengthening it afterwards would make the header a lie.
func (t *timedResponse) opening(body []byte) ([]byte, bool) {
	header := t.ResponseWriter.Header()
	if header.Get("Content-Length") != "" {
		return nil, false
	}
	media, _, err := mime.ParseMediaType(header.Get("Content-Type"))
	if err != nil || media != "application/json" {
		return nil, false
	}
	if len(body) < 2 || body[0] != '{' {
		return nil, false
	}

	// What follows the brace says whether the field needs a comma after it, so a body
	// that has not got that far yet is left alone rather than guessed at.
	rest := bytes.TrimLeft(body[1:], " \t\r\n")
	if len(rest) == 0 {
		return nil, false
	}
	opening := fmt.Sprintf(`{"duration":"%.2fms"`, t.spent())
	if rest[0] == '}' {
		return []byte(opening), true
	}
	return []byte(opening + ","), true
}

// withTrace opens a span for the whole request and names it after the route that served
// it.
//
// The name is settled afterwards because the pattern is not known until chi has matched
// one, and a span per session id is a trace nobody can group by. A route context is seeded
// here so that the mux fills in the one this can read back; chi only makes its own when
// there is none.
func withTrace(next http.Handler) http.Handler {
	traced := otelhttp.NewHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		routes := chi.NewRouteContext()
		r = r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, routes))
		next.ServeHTTP(w, r)
		if pattern := routes.RoutePattern(); pattern != "" {
			trace.SpanFromContext(r.Context()).SetName(r.Method + " " + pattern)
		}
	}), "router", otelhttp.WithSpanNameFormatter(func(_ string, r *http.Request) string {
		return r.Method
	}))
	return traced
}

// withRequestLog records one line per request served.
//
// It sits inside withCustomer so it can name the caller, and outside withServerSide so a
// refusal is a logged 403 rather than a request that appears not to have arrived. It is
// inside withCORS, which means a preflight goes unlogged: it carries nothing worth routing
// and doubling the volume to record that a browser asked permission is a poor trade.
//
// The path is logged without the query, because a socket names its customer there and a
// vendor names a token, and an access log is the last place either should end up.
//
// A 5xx is logged at error level. An access log at a busy deployment is the one stream
// nobody reads all of, and a server error that only appears in it is a server error nobody
// notices.
//
// A panic is logged with its stack and answered with a 500 here, then panicked again so
// Sentry still reports it. Sentry recovers without writing a status, which net/http sends
// as an empty 200, and a request that panicked would otherwise leave no line at all.
func (s *Server) withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		recorder := &loggedResponse{ResponseWriter: w}
		defer func() {
			recovered := recover()
			if recovered == nil {
				return
			}
			// ErrAbortHandler is net/http's own way of dropping a connection, not a bug.
			if recovered != http.ErrAbortHandler {
				customer, _ := CustomerFrom(r.Context())
				s.logger.Error("a request panicked",
					"method", r.Method, "path", r.URL.Path, "customer", customer,
					"panic", fmt.Sprint(recovered), "stack", string(debug.Stack()))
				if recorder.code == 0 && recorder.written == 0 && !recorder.hijacked {
					http.Error(recorder, `{"error":"internal error"}`, http.StatusInternalServerError)
				}
			}
			panic(recovered)
		}()
		next.ServeHTTP(recorder, r)

		customer, _ := CustomerFrom(r.Context())
		// A socket reports the life of the connection rather than a time to respond,
		// since it is logged once it has closed.
		fields := []any{
			"method", r.Method,
			"path", r.URL.Path,
			"status", recorder.status(),
			"duration", time.Since(started).Round(time.Millisecond),
			"customer", customer,
		}
		if recorder.written > 0 {
			fields = append(fields, "bytes", recorder.written)
		}
		if recorder.status() >= http.StatusInternalServerError {
			s.logger.Error("served a request", fields...)
			return
		}
		s.logger.Info("served a request", fields...)
	})
}

// loggedResponse remembers what was answered so it can be logged once the handler is done.
type loggedResponse struct {
	http.ResponseWriter
	code     int
	written  int64
	hijacked bool
}

// status reports what the caller was told, filling in the two codes a handler can answer
// with without ever naming: writing a body implies a 200, and writing nothing at all is the
// 200 net/http sends when the handler returns.
func (l *loggedResponse) status() int {
	switch {
	case l.code != 0:
		return l.code
	case l.hijacked:
		return http.StatusSwitchingProtocols
	default:
		return http.StatusOK
	}
}

func (l *loggedResponse) WriteHeader(code int) {
	if l.code == 0 {
		l.code = code
	}
	l.ResponseWriter.WriteHeader(code)
}

func (l *loggedResponse) Write(body []byte) (int, error) {
	written, err := l.ResponseWriter.Write(body)
	l.written += int64(written)
	return written, err
}

// Hijack hands the connection over for a socket upgrade, and is declared here rather than
// left to Unwrap so that the upgrade is what gets logged instead of an empty 200.
func (l *loggedResponse) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	conn, buffered, err := http.NewResponseController(l.ResponseWriter).Hijack()
	if err == nil {
		l.hijacked = true
	}
	return conn, buffered, err
}

// Unwrap lets a handler reach the flushing and deadline setting of the writer underneath
// through http.ResponseController, which a streamed response needs.
func (l *loggedResponse) Unwrap() http.ResponseWriter {
	return l.ResponseWriter
}

// unspecifiedRoutes are the hand-written handlers, and whether a client may reach each.
//
// They are named here because they are excluded from generation — a strict server can
// express neither an upgrade nor a stream — and excluding an operation drops it from the
// embedded spec. An operation the spec cannot see is the one place an inverted default
// could fail open, so this is the complement's other half rather than a note about
// sockets. A test holds it to naming exactly what api/oapi-codegen.yaml excludes.
var unspecifiedRoutes = map[string]bool{
	"GET /v1/agents/sessions/{id}/events": true,
	"GET /v1/{modality}/stream":           false,
	"GET /v1/dispatch":                    false,
	"GET /v1/agents/logs":                 false,
	"GET /v1/agents/logs/stream":          false,
	"GET /v1/agents/logs/{id}":            false,
	"GET /v1/data/export":                 false,
	"POST /v1/data/import":                false,
	"GET /v1/data/changes":                false,
	// Reached before there is a caller to classify: the browser arrives from the identity
	// provider and the state parameter is the secret.
	"GET /v1/agents/plugins/callback": true,
}

// serverSideRoutes builds the matcher for every operation an end user's device may not
// reach, which is every operation the spec does not mark client-accessible.
//
// The spec's own path templates are the patterns, because OpenAPI writes a parameter as
// {id} and so does ServeMux: a route is registered rather than translated. Matching is
// then the same routing the generated handlers get, so an operation cannot be reached by
// a path that spells it differently.
//
// An operation declaring no security at all is skipped in both directions. It is reached
// before there is a caller to classify — the health check and the plugin redirect, where
// the browser arrives from the identity provider — so there is nobody to refuse.
func serverSideRoutes(document *huma.OpenAPI) (*http.ServeMux, error) {
	operations, err := specifiedOperations(document)
	if err != nil {
		return nil, err
	}

	routes := http.NewServeMux()
	nothing := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	for _, operation := range operations {
		if operation.public || operation.open {
			continue
		}
		routes.Handle(operation.method+" "+operation.path, nothing)
	}
	for route, open := range unspecifiedRoutes {
		if !open {
			routes.Handle(route, nothing)
		}
	}
	return routes, nil
}

// withServerSide refuses the generated operations only a backend may reach.
//
// It sits after withCustomer, because refusing a caller for what it is means having worked
// out what it is first. The three sockets are left out of the embedded spec by being left
// out of generation, so socketRoutes puts them back rather than leaving them to be open by
// omission.
//
// A caller that authenticated and asked for one of these gets a 403 rather than a 401: it
// has already proved who it is, so there is nothing to be learned from a specific answer
// and a caller told "unauthenticated" would go looking for a credential problem it does
// not have. One that never authenticated is left to the handler's own 401.
func (s *Server) withServerSide(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, matched := s.serverSide.Handler(r); matched != "" {
			if s.refuseClientSide(w, r) {
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// refuseClientSide answers a caller that has authenticated as an end user's device and
// asked for something only a backend may have, and reports whether it did.
func (s *Server) refuseClientSide(w http.ResponseWriter, r *http.Request) bool {
	if _, known := CustomerFrom(r.Context()); !known || ServerSideFrom(r.Context()) {
		return false
	}
	s.logger.Debug("refused a client-side caller a server-side operation",
		"method", r.Method, "path", r.URL.Path)
	writeError(w, http.StatusForbidden, "this operation is server-side only: it needs "+
		auth.AuthTypeHeader+": "+auth.AuthTypeServer+" and a token carrying server: true")
	return true
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
//
// The one failure it does answer is a caller whose level the app turns away. That caller
// proved who it is, so there is nothing to protect by staying quiet, and the advice it
// needs is the opposite of the advice a 401 gives: its credential is fine and this app
// does not take guests.
//
// The caller is recorded for a backend too, even though a backend is charged no limit. It
// is how a backend says which of its users it is opening a session for, so that the user's
// own device can reach that session afterwards; what keeps the limit off it is that
// withQuota looks at whether the caller is server-side rather than at whether there is one.
func (s *Server) withCustomer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, span := tracer.Start(r.Context(), "auth.authenticate")
		principal, err := s.authenticator.Authenticate(ctx, r)
		span.End()
		if errors.Is(err, auth.ErrLevelRefused) {
			s.logger.Debug("refused a level of user this app turns away",
				"method", r.Method, "path", r.URL.Path, "kind", principal.Kind)
			writeError(w, http.StatusForbidden, "this app does not accept "+
				"requests from this level of user")
			return
		}
		if err == nil && principal.AppID != "" {
			ctx := context.WithValue(r.Context(), customerContextKey{}, principal.AppID)
			ctx = context.WithValue(ctx, organizationContextKey{}, principal.OrganizationID)
			ctx = context.WithValue(ctx, serverSideContextKey{}, principal.ServerSide)
			ctx = context.WithValue(ctx, kindContextKey{}, principal.Kind)
			ctx = context.WithValue(ctx, callerContextKey{}, routing.Caller{
				UserID: principal.UserID,
				IP:     clientIP(r, s.trusted),
			})
			r = r.WithContext(ctx)
			s.policies.Join(principal.AppID, principal.OrganizationID)
		}
		next.ServeHTTP(w, r)
	})
}

// corsRequestHeaders are the request headers a browser may send.
//
// They cover both ways a caller proves itself, because one deployment's browser is not the
// other's: reached through Stream's proxy a token arrives in Authorization with its kind
// named in Stream-Auth-Type, while a deployment running without a proxy and without keys
// names its tenant in X-Customer-Id instead. X-Stream-Client is what Stream's own clients
// tag themselves with, and arrives whether or not anything here reads it.
//
// A preflight refuses any header it was not asked about, and the browser reports that as a
// blocked request naming only the header, so a list covering one mode alone fails in a way
// that looks like the origin was never allowed.
const corsRequestHeaders = "Authorization, " + auth.AuthTypeHeader + ", " + auth.APIKeyHeader +
	", X-Stream-Client, " + CustomerHeader + ", Content-Type"

// corsMethods are the methods this API serves. PUT belongs here because a live session's
// instructions are replaced with one; PATCH does not, because the spec serves none.
const corsMethods = "GET, POST, PUT, DELETE, OPTIONS"

// withCORS lets a browser at the API from the origins the deployment named.
//
// It exists for a browser app that talks to the router directly rather than through a
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
			w.Header().Set("Access-Control-Allow-Headers", corsRequestHeaders)
			w.Header().Set("Access-Control-Allow-Methods", corsMethods)
			w.Header().Set("Access-Control-Expose-Headers", "Server-Timing")
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

// ServerSideFrom reports whether the request came from a process the customer runs rather
// than from an end user's device.
func ServerSideFrom(ctx context.Context) bool {
	serverSide, _ := ctx.Value(serverSideContextKey{}).(bool)
	return serverSide
}

// CallerFrom returns the end user the request is for and where they made it from, which is
// what a daily limit is counted against. It is empty for a request nobody authenticated,
// and empty means nothing is counted. A backend acting for a named user has one and is
// still counted nothing, which withQuota decides by asking whether the caller is
// server-side rather than by asking whether there is one.
func CallerFrom(ctx context.Context) routing.Caller {
	caller, _ := ctx.Value(callerContextKey{}).(routing.Caller)
	return caller
}

// KindFrom returns what sort of caller made the request.
func KindFrom(ctx context.Context) auth.Kind {
	kind, _ := ctx.Value(kindContextKey{}).(auth.Kind)
	return kind
}

// OwnerFrom is who the request may reach sessions as: the customer, the end user behind
// it and which sort of caller that is. A backend reaches all of its customer's sessions
// whether or not it names a user; anybody else reaches only what they opened themselves,
// or what their own backend opened in their name.
func OwnerFrom(ctx context.Context) session.Owner {
	customerID, _ := CustomerFrom(ctx)
	return session.Owner{
		CustomerID: customerID,
		UserID:     CallerFrom(ctx).UserID,
		Kind:       KindFrom(ctx),
	}
}

// routerFor returns the router serving a modality, or false when this deployment does not
// serve it.
func (s *Server) routerFor(modality Modality) (routing.Inspector, bool) {
	router, ok := s.routers[routing.Modality(modality)]
	return router, ok
}

// HealthStatus is whether the router is serving, and how each dependency answered.
type HealthStatus struct {
	Status       HealthStatusStatus `json:"status" enum:"ok,degraded"`
	Dependencies map[string]string  `json:"dependencies" doc:"Dependency name to \"ok\" or a failure description." example:"{\"postgres\":\"ok\",\"redis\":\"ok\"}"`
}

// HealthStatusStatus is whether every dependency answered.
type HealthStatusStatus string

const (
	Ok       HealthStatusStatus = "ok"
	Degraded HealthStatusStatus = "degraded"
)

type healthResponse struct {
	Status int
	Body   HealthStatus
}

func (s *Server) registerHealth(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getHealth",
		Method:      http.MethodGet,
		Path:        "/health",
		Summary:     "Liveness and dependency check",
		Security:    []map[string][]string{},
		Responses: map[string]*huma.Response{
			"200": {Description: "The router is serving"},
			"503": {
				Description: "A dependency is unavailable",
				Content: map[string]*huma.MediaType{
					"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/HealthStatus"}},
				},
			},
		},
	}, s.getHealth)
}

// getHealth reports whether the router and its dependencies are usable.
func (s *Server) getHealth(ctx context.Context, _ *struct{}) (*healthResponse, error) {
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
		return &healthResponse{
			Status: http.StatusServiceUnavailable,
			Body:   HealthStatus{Status: Degraded, Dependencies: dependencies},
		}, nil
	}
	return &healthResponse{
		Status: http.StatusOK,
		Body:   HealthStatus{Status: Ok, Dependencies: dependencies},
	}, nil
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
	shares := s.popularity.shares(ctx, string(request.Modality))
	providers := make([]Provider, 0, len(candidates))
	for _, candidate := range candidates {
		share := shares[candidate.Config.Name()]
		providers = append(providers, Provider{
			Provider:    candidate.Config.Provider,
			Model:       candidate.Config.Model,
			Description: &candidate.Config.Description,
			Languages:   candidate.Config.Languages,
			Realtime:    candidate.Config.Realtime,
			Tier:        tierOf(candidate.Config),
			Health:      providerHealth(candidate.Health),
			UsageShare:  &share,
			Benchmark:   providerBenchmark(candidate.Config.Benchmark),
			Price:       providerPrice(candidate.Config.Price),
		})
	}
	return ListProviders200JSONResponse(providers), nil
}

// ListRoutes returns the shortcuts offered as a choice and what each resolves to now.
func (s *Server) ListRoutes(ctx context.Context, request ListRoutesRequestObject) (ListRoutesResponseObject, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return ListRoutes401JSONResponse{missingCustomer()}, nil
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return ListRoutes404JSONResponse{unknownModality(request.Modality)}, nil
	}

	config := router.Config()
	offered := config.Offered()
	routes := make([]Route, 0, len(offered))
	for _, name := range offered {
		candidates, err := router.Resolve(ctx, name, nil)
		if err != nil {
			return nil, err
		}
		resolved := make([]Candidate, 0, len(candidates))
		for _, candidate := range candidates {
			resolved = append(resolved, Candidate{
				Provider: candidate.Config.Provider,
				Model:    candidate.Config.Model,
				Health:   providerHealth(candidate.Health),
			})
		}
		alias := config.Aliases[name]
		routes = append(routes, Route{
			Id:          name,
			Title:       alias.Title,
			Description: alias.Description,
			Candidates:  resolved,
		})
	}
	return ListRoutes200JSONResponse(routes), nil
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
			ImagesTotal:            bucket.ImagesTotal,
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
			ImagesTotal:            bucket.ImagesTotal,
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

// GetSpend returns what the calling customer spent, grouped.
func (s *Server) GetSpend(ctx context.Context, request GetSpendRequestObject) (GetSpendResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetSpend401JSONResponse{missingCustomer()}, nil
	}
	if !request.Params.To.After(request.Params.From) {
		return GetSpend400JSONResponse{badRequest("to must be after from")}, nil
	}
	tags, err := parseTagFilter(request.Params.Tag)
	if err != nil {
		return GetSpend400JSONResponse{badRequest(err.Error())}, nil
	}
	if s.store == nil {
		return GetSpend400JSONResponse{badRequest("statistics are not available: no database configured")}, nil
	}

	groupBy := defaultSpendGroupBy
	if request.Params.GroupBy != nil && *request.Params.GroupBy != "" {
		groupBy = *request.Params.GroupBy
	}
	limit := defaultSpendGroups
	if request.Params.Limit != nil {
		limit = *request.Params.Limit
	}

	buckets, err := s.store.CustomerSpend(ctx, customerID, groupBy,
		granularityOf(request.Params.Granularity), request.Params.From, request.Params.To, limit, tags)
	if err != nil {
		return GetSpend400JSONResponse{badRequest(err.Error())}, nil
	}

	spend := make([]SpendBucket, 0, len(buckets))
	for _, bucket := range buckets {
		spend = append(spend, SpendBucket{
			Bucket:          bucket.Bucket,
			Value:           bucket.Value,
			CostMicrosTotal: bucket.CostMicrosTotal,
			RequestCount:    bucket.RequestCount,
		})
	}
	return GetSpend200JSONResponse(spend), nil
}

// GetTagKeys returns which cost labels the calling customer's spend carries.
func (s *Server) GetTagKeys(ctx context.Context, request GetTagKeysRequestObject) (GetTagKeysResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetTagKeys401JSONResponse{missingCustomer()}, nil
	}
	if !request.Params.To.After(request.Params.From) {
		return GetTagKeys400JSONResponse{badRequest("to must be after from")}, nil
	}
	tags, err := parseTagFilter(request.Params.Tag)
	if err != nil {
		return GetTagKeys400JSONResponse{badRequest(err.Error())}, nil
	}
	if s.store == nil {
		return GetTagKeys400JSONResponse{badRequest("statistics are not available: no database configured")}, nil
	}

	found, err := s.store.CustomerTagKeys(ctx, customerID, request.Params.From, request.Params.To, tags)
	if err != nil {
		return nil, err
	}

	keys := make([]TagKeySummary, 0, len(found))
	for _, key := range found {
		values := make([]TagValueSummary, 0, len(key.TopValues))
		for _, value := range key.TopValues {
			values = append(values, TagValueSummary{
				Value:           value.Value,
				CostMicrosTotal: value.CostMicrosTotal,
				RequestCount:    value.RequestCount,
				Share:           value.Share,
			})
		}
		keys = append(keys, TagKeySummary{
			Key:             key.Key,
			ValueCount:      key.ValueCount,
			CostMicrosTotal: key.CostMicrosTotal,
			RequestCount:    key.RequestCount,
			Coverage:        key.Coverage,
			TopValues:       values,
		})
	}
	return GetTagKeys200JSONResponse(keys), nil
}

// GetActivity returns who used the calling customer's agents, and how much.
func (s *Server) GetActivity(ctx context.Context, request GetActivityRequestObject) (GetActivityResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetActivity401JSONResponse{missingCustomer()}, nil
	}
	if !request.Params.To.After(request.Params.From) {
		return GetActivity400JSONResponse{badRequest("to must be after from")}, nil
	}
	if s.store == nil {
		return GetActivity400JSONResponse{badRequest("statistics are not available: no database configured")}, nil
	}

	buckets, err := s.store.CustomerActivity(ctx, customerID,
		activityGranularityOf(request.Params.Granularity), request.Params.From, request.Params.To)
	if err != nil {
		return nil, err
	}

	activity := make([]ActivityBucket, 0, len(buckets))
	for _, bucket := range buckets {
		activity = append(activity, ActivityBucket{
			Bucket:       bucket.Bucket,
			ActiveUsers:  bucket.ActiveUsers,
			Sessions:     bucket.Sessions,
			Messages:     bucket.Messages,
			Calls:        bucket.Calls,
			VoiceMinutes: bucket.VoiceMinutes,
			PhoneMinutes: bucket.PhoneMinutes,
		})
	}
	return GetActivity200JSONResponse(activity), nil
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
	if requested != nil && *requested == GranularityDaily {
		return store.Daily
	}
	return store.Hourly
}

// The spend defaults, matching the spec: the whole bill by where it went, and few enough
// groups to read.
const (
	defaultSpendGroupBy = "modality"
	defaultSpendGroups  = 6
)

// activityGranularityOf defaults to daily, matching the spec.
func activityGranularityOf(requested *ActivityGranularity) store.ActivityGranularity {
	if requested != nil && *requested == ActivityGranularityMonthly {
		return store.ActivityMonthly
	}
	return store.ActivityDaily
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

// providerBenchmark leaves out what was not measured, and is nil for a model with no
// measurements at all.
func providerBenchmark(benchmark routing.Benchmark) *ProviderBenchmark {
	if benchmark == (routing.Benchmark{}) {
		return nil
	}
	measured := func(v float64) *float64 {
		if v == 0 {
			return nil
		}
		return &v
	}
	counted := func(v int) *int {
		if v == 0 {
			return nil
		}
		return &v
	}
	return &ProviderBenchmark{
		Elo:                   counted(benchmark.Elo),
		CharactersPerSecond:   measured(benchmark.CharactersPerSecond),
		WordErrorRate:         measured(benchmark.WordErrorRate),
		LatencyMs:             counted(benchmark.LatencyMs),
		SearchIndex:           counted(benchmark.SearchIndex),
		CostPerTask:           measured(benchmark.CostPerTask),
		IntelligenceIndex:     counted(benchmark.IntelligenceIndex),
		OutputTokensPerSecond: measured(benchmark.OutputTokensPerSecond),
	}
}

// providerPrice is the token rates a model is billed at, and nil for a model not billed
// by the token.
func providerPrice(price routing.Price) *ProviderPrice {
	if price.PerMillionInputTokens == 0 && price.PerMillionOutputTokens == 0 {
		return nil
	}
	return &ProviderPrice{
		PerMillionInputTokens:  &price.PerMillionInputTokens,
		PerMillionOutputTokens: &price.PerMillionOutputTokens,
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
