// Package api serves the router's HTTP surface on a chi router. Operations are declared
// in Go with Huma, and the Go structs are the source of truth: api/openapi.yaml is
// rendered from them by cmd/openapi and never edited by hand.
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
	"sync"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tracing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/GetStream/Vision-Agents/acceleration/internal/users"
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
	// Users writes down the end users each app is seen acting for. Built over Store when
	// it is not given, in which case it caches in this process alone.
	Users *users.Recorder
	Live  *live.Client
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
	// Stream resolves which Stream app, and which credential, the router acts in for each
	// customer: tokens, guests and the transcripts read back. Absent when the deployment
	// has no Stream app at all, in which case those paths say so.
	Stream *streamapp.Clients
	// ProxyDeclaresKind is auth.proxy_declares_kind: the proxy in front says which kind of
	// caller it verified. Registering a Stream app behind a proxy needs it, since without
	// it every caller passes as a backend.
	ProxyDeclaresKind bool
	// TrustAPIKeyHeader lets X-Stream-Api-Key choose which of the calling app's registered
	// keys mints its tokens.
	TrustAPIKeyHeader bool
	// DenyRegistration are Stream app ids that may never be registered.
	DenyRegistration []string
	// Campaigns rings lists of people. Absent without telephony or sessions, in which
	// case a campaign can be written down but not run.
	Campaigns *campaign.Runner
	// Simulations puts an agent through a conversation somebody wrote down and rules on
	// how it went. Absent without sessions or model routing, in which case a simulation
	// can be written down but not run.
	Simulations *simulation.Runner
	// PluginEvents subscribes agents to their plugins' MCP events and answers deliveries.
	// Absent without a database or sessions, in which case a config's plugin_events are
	// stored but nothing subscribes to them.
	PluginEvents *pluginevents.Service
	// Channels answers what arrives on the app's WhatsApp, text and iMessage lines. Absent
	// without a database, sessions or a key encryption key, in which case a line cannot be
	// connected and nothing is delivered.
	Channels *channels.Service
	// DLC reviews and registers the app's 10DLC use cases. Absent without a database, in
	// which case the 10DLC paths say so.
	DLC *dlc.Service
	// Gate is what the sandbox and opt-out paths read. Nil enforces nothing.
	Gate *dlc.Gate
	// OpsKey is what Stream staff's review paths are reached with. Empty turns them off.
	OpsKey string
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
	// HookSecret is the deployment app's secret, which signs the events Stream sends to
	// the hooks. Without it the hooks refuse every request, because an unsigned hook is
	// anyone who found the URL.
	HookSecret string
	// CORSOrigins are the browser origins allowed to call this API directly, which is
	// what a dashboard talking to the router without a proxy in between needs. Empty
	// means no browser may, which is right for a deployment only servers reach.
	CORSOrigins []string
	// Secrets seals the credentials this API is given to keep: a channel's provider tokens.
	// Absent when the deployment has no key encryption key, in which case the paths that
	// would store one say so rather than holding it in the clear.
	Secrets *auth.Sealer
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
	// ConnectorSecrets is the keyring connector consents and credentials are sealed under.
	// Absent when connectors are off, in which case no consent can be started.
	ConnectorSecrets *auth.Sealer
	// ConnectorResolver is the one door to a connection's credential. The events endpoint
	// revokes through it when a provider says a grant ended. Absent when connectors are off,
	// in which case the endpoint takes no events.
	ConnectorResolver core.Resolver
	// ConnectorTransports builds each connection's outbound client over ConnectorResolver.
	// The validate endpoint lists a connection's tools through it. Absent when connectors are
	// off, in which case no connection can be validated.
	ConnectorTransports *core.Transports
	// ConnectorLimiter holds a connection's direct calls after its provider answered 429, until
	// the Retry-After it asked for (core.Limiter). Absent, which it is with connectors off or
	// without Redis, nothing is held and the provider limits alone.
	ConnectorLimiter *core.Limiter
	// ConnectorEventSecrets finds the secret a connector's events are verified with
	// (ConnectorEventSecrets reads the operator's from the environment). Absent, the
	// endpoint takes no events.
	ConnectorEventSecrets EventSecretLookup
	// ChannelBridge takes the messages a verified provider event carries
	// (internal/channelbridge). Absent, they are logged and dropped.
	ChannelBridge ChannelBridge
	// EventForwarder forwards a provider app's verified deliveries to the customer's event
	// destinations, and serves the endpoints that manage them (internal/eventforward). Absent,
	// nothing is forwarded and the destination endpoints say forwarding is not enabled.
	EventForwarder *eventforward.Forwarder
	// MCPEvents subscribes connector bindings to their connection's MCP events and answers
	// the deliveries (internal/mcpevents). Absent, which it is with connectors off, a validate
	// subscribes to nothing and the deliveries route answers 410.
	MCPEvents *mcpevents.Service
	// Episodes closes the episodes of a call when the call.session_ended hook says it ended,
	// and summarizes them (internal/omnichannel, T55). Absent, a call's episodes stay in
	// progress, as before T55.
	Episodes *omnichannel.Closer
	// SlackApps creates, updates and deletes the Slack app the router keeps for a customer
	// (managed, T54). Absent, the provider app paths say connectors are not enabled.
	SlackApps *slackapps.Client
	// OperatorProviderApps finds this deployment's own provider app for a built-in connector
	// (ConnectorOperatorApps reads it from the environment), which Stream staff make one
	// customer's. Absent, the staff paths say it is not configured.
	OperatorProviderApps OperatorAppLookup
	// TrustedProxies are the ranges this deployment's own proxies sit in, and they decide
	// how much of X-Forwarded-For is believed when working out who a request is from.
	// Empty means none of it is, and the connection's own address is used.
	TrustedProxies []netip.Prefix
	// PluginHTTP reaches plugins' MCP servers and their auth servers: a login, its callback,
	// and a config's MCP servers describing themselves. Absent reaches only public hosts.
	PluginHTTP *http.Client
	Logger     *slog.Logger
}

// Server serves the router's HTTP API.
type Server struct {
	routers   map[routing.Modality]routing.Inspector
	store     *store.Store
	configs   *appconfig.Store
	users     *users.Recorder
	live      *live.Client
	phone     *phone.Service
	sessions  *session.Manager
	relayed   *relayed
	directory *node.Directory
	forwarder *node.Forwarder
	streams   *Streams
	stream    *streamapp.Clients
	// touched is when each key was last recorded as signing a hook.
	touched           sync.Map
	proxyDeclaresKind bool
	trustAPIKeyHeader bool
	denyRegistration  []string
	hookSecret        string
	campaigns         *campaign.Runner
	simulations       *simulation.Runner
	knowledge         knowledge.Writer
	pages             *urls.Service
	voices            *voices.Service
	library           *voices.Catalogue
	dispatch          *dispatch.Pool
	secrets           *auth.Sealer
	corsOrigins       []string
	publicURL         string
	dashboardURL      string
	oauth             *plugins.Auth
	pluginEvents      *pluginevents.Service
	channels          *channels.Service
	dlc               *dlc.Service
	gate              *dlc.Gate
	opsKey            string
	authenticator     auth.Authenticator
	authMode          auth.Mode
	dataRetention     time.Duration
	quota             *quota.Limiter
	policies          *policy.Enforcer
	connectors        core.Registry
	// connectorSecrets seals consent attempts; credentials stores what a consent got. Both
	// are nil when connectors are off.
	connectorSecrets *auth.Sealer
	credentials      core.CredentialStore
	// connectorResolver, eventSecrets and channelBridge serve the connector events endpoint.
	connectorResolver core.Resolver
	eventSecrets      EventSecretLookup
	channelBridge     ChannelBridge
	eventForwarder    *eventforward.Forwarder
	mcpEvents         *mcpevents.Service
	// episodes ends a call's episodes on call.session_ended.
	episodes *omnichannel.Closer
	// slackApps and operatorApps serve the provider app paths.
	slackApps    *slackapps.Client
	operatorApps OperatorAppLookup
	trusted      []netip.Prefix

	// connectorTransports is what the validate endpoint reaches a connection's tools through.
	connectorTransports *core.Transports
	// connectorLimiter holds the proxy's calls after a provider's 429; nil holds none.
	connectorLimiter *core.Limiter

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
	recorder := options.Users
	if recorder == nil && options.Store != nil {
		var err error
		if recorder, err = users.New(users.Options{Store: options.Store, Logger: logger}); err != nil {
			return nil, err
		}
	}
	server := &Server{
		routers:           options.Routers,
		store:             options.Store,
		configs:           configs,
		users:             recorder,
		live:              options.Live,
		phone:             options.Phone,
		sessions:          options.Sessions,
		directory:         options.Directory,
		streams:           options.Streams,
		stream:            options.Stream,
		proxyDeclaresKind: options.ProxyDeclaresKind,
		trustAPIKeyHeader: options.TrustAPIKeyHeader,
		denyRegistration:  options.DenyRegistration,
		hookSecret:        options.HookSecret,
		campaigns:         options.Campaigns,
		simulations:       options.Simulations,
		knowledge:         options.Knowledge,
		pages:             options.KnowledgeURLs,
		voices:            options.Voices,
		library:           options.VoiceLibrary,
		dispatch:          options.Dispatch,
		secrets:           options.Secrets,
		corsOrigins:       options.CORSOrigins,
		publicURL:         options.PublicURL,
		dashboardURL:      options.DashboardURL,
		authenticator:     authenticator,
		authMode:          authMode,
		dataRetention:     retention,
		quota:             options.Quota,
		policies:          options.Policies,
		connectors:        options.Connectors,
		connectorResolver: options.ConnectorResolver,
		eventSecrets:      options.ConnectorEventSecrets,
		channelBridge:     options.ChannelBridge,
		eventForwarder:    options.EventForwarder,
		mcpEvents:         options.MCPEvents,
		episodes:          options.Episodes,
		slackApps:         options.SlackApps,
		operatorApps:      options.OperatorProviderApps,
		trusted:           options.TrustedProxies,
		upgrader:          newUpgrader(options.CORSOrigins),
		oauth: &plugins.Auth{
			HTTP:         options.PluginHTTP,
			PublicURL:    options.PublicURL,
			DashboardURL: options.DashboardURL,
			Clients:      session.PluginClients(options.Store, options.Secrets),
		},
		pluginEvents: options.PluginEvents,
		channels:     options.Channels,
		dlc:          options.DLC,
		gate:         options.Gate,
		opsKey:       options.OpsKey,
		popularity:   newPopularity(options.Store, logger),
		logger:       logger,
	}
	if options.Store != nil && options.ConnectorSecrets != nil {
		credentials, err := pgsealed.New(options.Store, options.ConnectorSecrets)
		if err != nil {
			return nil, err
		}
		server.connectorSecrets, server.credentials = options.ConnectorSecrets, credentials
	}
	server.connectorTransports = options.ConnectorTransports
	server.connectorLimiter = options.ConnectorLimiter
	if server.channelBridge == nil {
		server.channelBridge = droppingBridge{logger: logger}
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
// The routes served by hand are registered first, on the router the Huma operations are then
// added to. The sockets are written by hand because an operation returns a response and an
// upgrade returns a connection, and the logs, exports and imports because they stream;
// documentHandWritten declares them in the spec. The answer host serves a vendor's XML rather
// than this API's JSON, and the call and message hooks are reached by somebody other than a
// customer, a telephony vendor and Stream, so those three are not in the spec at all.
func (s *Server) Handler() http.Handler {
	mux := chi.NewRouter()
	mux.NotFound(func(w http.ResponseWriter, _ *http.Request) {
		writeError(w, notFound("no such route"))
	})
	mux.MethodNotAllowed(func(w http.ResponseWriter, r *http.Request) {
		writeError(w, newAPIError(ErrorTypeMethodNotAllowed, r.Method+" is not served on this route"))
	})
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
	mux.HandleFunc("POST "+phone.CallHookPath+"/{app}", s.receiveCallEvent)
	mux.HandleFunc("POST "+chat.MessageHookPath, s.receiveMessageEvent)
	mux.HandleFunc("POST "+chat.MessageHookPath+"/{app}", s.receiveMessageEvent)
	mux.HandleFunc("GET "+plugins.CallbackPath, s.finishPluginLogin)
	mux.HandleFunc("GET "+connectorLaunchPath+"{id}", s.serveConnectorLaunch)
	mux.HandleFunc("POST "+connectorLaunchPath+"{id}", s.handOffConnectorLaunch)
	mux.HandleFunc("GET "+ConnectorCallbackPath, s.finishConnectorConsent)
	mux.HandleFunc("GET "+ConnectorClientMetadataPath, s.serveConnectorClientMetadata)
	mux.HandleFunc("POST "+connectorEventsPath+"{connector_id}", s.receiveConnectorEvent)
	mux.HandleFunc("POST "+providerAppEventsPath+"{connector_id}/{provider_app_id}", s.receiveProviderAppEvent)
	// With connectors off (no transports) the proxy is no route at all, as before it existed.
	if s.connectorTransports != nil {
		for _, method := range proxyMethods {
			mux.HandleFunc(method+" "+connectionProxyPath+"*", s.proxyConnection)
		}
	}
	mux.HandleFunc("GET /v1/agents/plugins/{plugin_id}/logo", s.servePluginLogo)
	mux.HandleFunc("POST "+plugins.EventsPath+"{token}", s.receivePluginEvent)
	mux.HandleFunc("POST "+mcpevents.Path+"{token}", s.receiveConnectionEvent)
	mux.HandleFunc("GET "+channels.HookPath+"{token}", s.receiveChannelMessage)
	mux.HandleFunc("POST "+channels.HookPath+"{token}", s.receiveChannelMessage)
	mux.HandleFunc("POST "+dlc.HookPath, s.receiveDLCReport)
	s.newAPI(mux)
	var handler http.Handler = mux
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
	served := withSentry(withRequestID(withTrace(withTiming(withCORS(s.corsOrigins,
		s.onOwningNode(s.withCustomer(s.withRequestLog(s.withQuota(s.withServerSide(handler))))))))))
	if s.directory == nil {
		return served
	}
	// Peers reach this node on the same port its callers do, so what they forward is
	// served beside everything else rather than on a listener of its own.
	return node.Serve(served)
}

// withSentry reports what goes wrong serving a request to Sentry, except a request
// carrying an app's Stream secrets: Sentry copies the body it is handed, and a secret in an
// error report is a secret leaked.
func withSentry(handler http.Handler) http.Handler {
	instrumented := sentryhttp.New(sentryhttp.Options{
		Repanic:         false,
		WaitForDelivery: false,
	}).Handle(handler)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, streamCredentialsPath) {
			handler.ServeHTTP(w, r)
			return
		}
		instrumented.ServeHTTP(w, r)
	})
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
		timed := &timedResponse{ResponseWriter: w, started: time.Now()}
		next.ServeHTTP(timed, r.WithContext(context.WithValue(r.Context(), timedResponseKey{}, timed)))
	})
}

// timedResponseKey holds the request's timedResponse, for leaveUntimed.
type timedResponseKey struct{}

// leaveUntimed has the answer written as the handler writes it, with no Server-Timing and no
// duration field: for an answer that is somebody else's, as the connection proxy's is.
func leaveUntimed(ctx context.Context) {
	if timed, ok := ctx.Value(timedResponseKey{}).(*timedResponse); ok {
		timed.stamped, timed.opened = true, true
	}
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
// The error a handler returned is logged on the same line, with its stack, since its
// answer names only the request id.
//
// A panic is logged with its stack and answered with a 500 here, then panicked again so
// Sentry still reports it. Sentry recovers without writing a status, which net/http sends
// as an empty 200, and a request that panicked would otherwise leave no line at all.
func (s *Server) withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		recorder := &loggedResponse{ResponseWriter: w}
		failure := &requestFailure{}
		r = r.WithContext(context.WithValue(r.Context(), requestFailureContextKey{}, failure))
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
					"request_id", RequestIDFrom(r.Context()),
					"panic", fmt.Sprint(recovered), "stack", string(debug.Stack()))
				if recorder.code == 0 && recorder.written == 0 && !recorder.hijacked {
					writeError(recorder, internalError())
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
			"request_id", RequestIDFrom(r.Context()),
		}
		if recorder.written > 0 {
			fields = append(fields, "bytes", recorder.written)
		}
		if failure.err != nil {
			fields = append(fields, "error", failure.err.Error())
		}
		if failure.trace != "" {
			fields = append(fields, "stack", failure.trace)
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
	operations := specifiedOperations(document)

	routes := http.NewServeMux()
	nothing := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	for _, operation := range operations {
		// The connection proxy refuses a client-side caller itself (proxyConnection), on every
		// path; with connectors off it is no route, and a 403 here would answer for it.
		if operation.public || operation.open || strings.HasPrefix(operation.path, connectionProxyPath) {
			continue
		}
		routes.Handle(operation.method+" "+operation.path, nothing)
	}
	return routes, nil
}

// withServerSide refuses the generated operations only a backend may reach.
//
// It sits after withCustomer, because refusing a caller for what it is means having worked
// out what it is first. The four sockets are left out of the embedded spec by being left
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
	writeError(w, errServerSideOnly)
	return true
}

var errServerSideOnly = APIError{
	Type: ErrorTypePermission, Code: codeServerSideOnly,
	Message: "this operation is server-side only: it needs " + auth.AuthTypeHeader + ": " +
		auth.AuthTypeServer + " and a token carrying server: true",
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
		// Stream is not a customer, and a delivery is authenticated by its signature. Who a
		// request to a hook claims to be is not read, so it cannot name a tenant or record
		// one under an organization.
		if isHook(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		ctx, span := tracer.Start(r.Context(), "auth.authenticate")
		principal, err := s.authenticator.Authenticate(ctx, r)
		span.End()
		if errors.Is(err, auth.ErrLevelRefused) {
			s.logger.Debug("refused a level of user this app turns away",
				"method", r.Method, "path", r.URL.Path, "kind", principal.Kind)
			writeError(w, forbidden("this app does not accept "+
				"requests from this level of user"))
			return
		}
		if err == nil && principal.AppID != "" {
			ctx := context.WithValue(r.Context(), customerContextKey{}, principal.AppID)
			ctx = context.WithValue(ctx, organizationContextKey{}, principal.OrganizationID)
			ctx = context.WithValue(ctx, serverSideContextKey{}, principal.ServerSide)
			ctx = context.WithValue(ctx, kindContextKey{}, principal.Kind)
			if s.trustAPIKeyHeader {
				ctx = context.WithValue(ctx, mintingKeyContextKey{}, strings.TrimSpace(r.Header.Get(mintingKeyHeader)))
			}
			ctx = context.WithValue(ctx, callerContextKey{}, routing.Caller{
				UserID: principal.UserID,
				IP:     clientIP(r, s.trusted),
			})
			ctx = context.WithValue(ctx, actorContextKey{}, actorOf(r, principal.ServerSide))
			r = r.WithContext(ctx)
			s.policies.Join(principal.AppID, principal.OrganizationID)
			s.recordUser(ctx, principal)
		}
		next.ServeHTTP(w, r)
	})
}

// recordUser writes down the end user a request is for, so an app can ask who its users
// are rather than only which of them it minted as guests.
//
// Only a verified caller is recorded. An anonymous one goes by a name nobody checked, so
// a table filled with those names would be a table of what callers asked to be called. A
// backend naming one of its users is recorded as an authenticated user, because a caller
// holding the secret could mint that user a token and so has nothing to gain by lying.
//
// A failure is logged rather than returned. Writing down who called is not what the
// caller asked for, and refusing the request they did ask for because of it would be the
// wrong trade.
func (s *Server) recordUser(ctx context.Context, principal auth.Principal) {
	if s.users == nil || principal.UserID == "" {
		return
	}
	kind := store.UserKindAuthenticated
	switch principal.Kind {
	case auth.KindGuest:
		kind = store.UserKindGuest
	case auth.KindAnonymous:
		return
	}
	if err := s.users.Seen(ctx, principal.AppID, principal.UserID, kind); err != nil {
		s.logger.Error("could not record an end user",
			"customer", principal.AppID, "user", principal.UserID, "error", err)
	}
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
//
// The two actor headers are here because the dashboard is a browser app: it is the client
// that knows which person clicked save, and the audit is only worth reading if that name
// reaches the router.
const corsRequestHeaders = "Authorization, " + auth.AuthTypeHeader + ", " + auth.APIKeyHeader +
	", " + clientHeader + ", " + actorIDHeader + ", " + actorNameHeader +
	", " + auth.UserHeader + ", " + CustomerHeader + ", Content-Type"

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
			w.Header().Set("Access-Control-Expose-Headers", "Server-Timing, "+RequestIDHeader)
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

// listProviders returns the providers configured for a modality and their live health.
func (s *Server) listProviders(ctx context.Context, request *listProvidersRequest) (*listProvidersResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return nil, unknownModality(request.Modality)
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
	return &listProvidersResponse{Body: providers}, nil
}

// listRoutes returns the shortcuts offered as a choice and what each resolves to now.
func (s *Server) listRoutes(ctx context.Context, request *listRoutesRequest) (*listRoutesResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return nil, unknownModality(request.Modality)
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
	return &listRoutesResponse{Body: routes}, nil
}

// resolveTarget explains which providers would serve a target, best first.
func (s *Server) resolveTarget(ctx context.Context, request *resolveTargetRequest) (*resolveTargetResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return nil, unknownModality(request.Modality)
	}

	var languageHints []string
	if request.Language.ptr() != nil {
		languageHints = *request.Language.ptr()
	}

	candidates, err := router.Resolve(ctx, request.Target, languageHints)
	if err != nil {
		return nil, notFound(err.Error())
	}

	resolved := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		resolved = append(resolved, Candidate{
			Provider: candidate.Config.Provider,
			Model:    candidate.Config.Model,
			Health:   providerHealth(candidate.Health),
		})
	}
	return &resolveTargetResponse{Body: resolved}, nil
}

// getStats returns the calling customer's aggregated usage for one modality.
func (s *Server) getStats(ctx context.Context, request *getStatsRequest) (*getStatsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	// Statistics are not limited to the routed modalities: memory and phone are recorded
	// the same way and cost the same customer money.
	if !request.To.After(request.From) {
		return nil, invalidRequest("to must be after from")
	}
	tags, err := parseTagFilter(request.Tag.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if s.store == nil {
		return nil, invalidRequest("statistics are not available: no database configured")
	}

	granularity := granularityOf(request.Granularity.ptr())
	buckets, err := s.store.CustomerStats(
		ctx, string(request.Modality), customerID, granularity, request.From, request.To, tags)
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
	return &getStatsResponse{Body: stats}, nil
}

// getTagStats returns the calling customer's usage broken down by one cost label.
func (s *Server) getTagStats(ctx context.Context, request *getTagStatsRequest) (*getTagStatsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if !request.To.After(request.From) {
		return nil, invalidRequest("to must be after from")
	}
	if s.store == nil {
		return nil, invalidRequest("statistics are not available: no database configured")
	}

	granularity := granularityOf(request.Granularity.ptr())
	buckets, err := s.store.CustomerTagStats(ctx, string(request.Modality), customerID,
		request.Key, granularity, request.From, request.To)
	if err != nil {
		return nil, invalidRequest(err.Error())
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
	return &getTagStatsResponse{Body: stats}, nil
}

// getTurnStats returns the calling customer's conversational latency.
func (s *Server) getTurnStats(ctx context.Context, request *getTurnStatsRequest) (*getTurnStatsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if !request.To.After(request.From) {
		return nil, invalidRequest("to must be after from")
	}
	if s.store == nil {
		return nil, invalidRequest("statistics are not available: no database configured")
	}

	var agentID string
	if request.AgentId.ptr() != nil {
		agentID = *request.AgentId.ptr()
	}

	granularity := granularityOf(request.Granularity.ptr())
	buckets, err := s.store.CustomerTurnStats(
		ctx, customerID, agentID, granularity, request.From, request.To)
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
	return &getTurnStatsResponse{Body: stats}, nil
}

// getSpend returns what the calling customer spent, grouped.
func (s *Server) getSpend(ctx context.Context, request *getSpendRequest) (*getSpendResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if !request.To.After(request.From) {
		return nil, invalidRequest("to must be after from")
	}
	tags, err := parseTagFilter(request.Tag.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if s.store == nil {
		return nil, invalidRequest("statistics are not available: no database configured")
	}

	groupBy := defaultSpendGroupBy
	if request.GroupBy.ptr() != nil && *request.GroupBy.ptr() != "" {
		groupBy = *request.GroupBy.ptr()
	}
	limit := defaultSpendGroups
	if request.Limit.ptr() != nil {
		limit = *request.Limit.ptr()
	}

	buckets, err := s.store.CustomerSpend(ctx, customerID, groupBy,
		granularityOf(request.Granularity.ptr()), request.From, request.To, limit, tags)
	if err != nil {
		return nil, invalidRequest(err.Error())
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
	return &getSpendResponse{Body: spend}, nil
}

// getTagKeys returns which cost labels the calling customer's spend carries.
func (s *Server) getTagKeys(ctx context.Context, request *getTagKeysRequest) (*getTagKeysResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if !request.To.After(request.From) {
		return nil, invalidRequest("to must be after from")
	}
	tags, err := parseTagFilter(request.Tag.ptr())
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	if s.store == nil {
		return nil, invalidRequest("statistics are not available: no database configured")
	}

	found, err := s.store.CustomerTagKeys(ctx, customerID, request.From, request.To, tags)
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
	return &getTagKeysResponse{Body: keys}, nil
}

// getActivity returns who used the calling customer's agents, and how much.
func (s *Server) getActivity(ctx context.Context, request *getActivityRequest) (*getActivityResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if !request.To.After(request.From) {
		return nil, invalidRequest("to must be after from")
	}
	if s.store == nil {
		return nil, invalidRequest("statistics are not available: no database configured")
	}

	buckets, err := s.store.CustomerActivity(ctx, customerID,
		activityGranularityOf(request.Granularity.ptr()), request.From, request.To)
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
	return &getActivityResponse{Body: activity}, nil
}

// runRollup aggregates request rows into a rollup table.
func (s *Server) runRollup(ctx context.Context, request *runRollupRequest) (*runRollupResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, errMissingCustomer
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if !request.Body.To.After(request.Body.From) {
		return nil, invalidRequest("to must be after from")
	}
	if s.store == nil {
		return nil, invalidRequest("rollups are not available: no database configured")
	}

	granularity := granularityOf(request.Body.Granularity)
	written, err := s.store.Rollup(ctx, granularity, request.Body.From, request.Body.To)
	if err != nil {
		return nil, err
	}

	return &runRollupResponse{Body: RollupResult{Granularity: Granularity(granularity),
		BucketsWritten: written}}, nil
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
			return nil, stack.Wrap(fmt.Errorf("tag %q must be written key:value", entry))
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

var errMissingCustomer = APIError{
	Type: ErrorTypeAuthentication, Code: codeMissingCustomer,
	Message: "the " + CustomerHeader + " header is required",
}

func unknownModality(modality Modality) APIError {
	return APIError{
		Type: ErrorTypeNotFound, Code: codeModalityNotRouted,
		Message: "this deployment does not route " + string(modality),
	}
}

// registerServer declares the operations served in server.go.
func (s *Server) registerServer(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listProviders",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/providers",
		Summary:     "List the providers configured for a modality and their live health",
		Responses: map[string]*huma.Response{
			"200": {Description: "The configured providers"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listProviders)
	huma.Register(api, huma.Operation{
		OperationID: "listRoutes",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/routes",
		Summary:     "List the capability shortcuts offered as a choice, each with the models it resolves to",
		Description: "The shortcuts a person picking a model is shown, in the order the deployment offers " +
			"them, so the first is the one a conversation gets by default. Shortcuts that exist only " +
			"for the router's own use are left out, though they can still be named as a target.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The offered shortcuts, default first"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listRoutes)
	huma.Register(api, huma.Operation{
		OperationID: "resolveTarget",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/routes/{target}",
		Summary:     "Resolve a provider name or capability shortcut to a ranked candidate list",
		Responses: map[string]*huma.Response{
			"200": {Description: "Candidates in preference order, best first"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.resolveTarget)
	huma.Register(api, huma.Operation{
		OperationID: "getStats",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/stats",
		Summary:     "Aggregated usage for the calling customer",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket, provider and model"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getStats)
	huma.Register(api, huma.Operation{
		OperationID: "getTagStats",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/stats/tags",
		Summary:     "Aggregated usage broken down by the values of one cost label",
		Description: "What drives the spend. Requests are labelled with whatever keys the customer chooses, " +
			"so asking for key=project returns one row per project per bucket.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket and label value"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getTagStats)
	huma.Register(api, huma.Operation{
		OperationID: "getTurnStats",
		Method:      http.MethodGet,
		Path:        "/v1/turns/stats",
		Summary:     "Conversational latency for the calling customer",
		Description: "One row per bucket and agent. A request row measures one provider call; a turn measures " +
			"what the caller felt, from finishing a sentence to hearing the answer start, with the " +
			"transcription, model and voice legs kept apart so a slow conversation can be attributed.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket and agent"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getTurnStats)
	huma.Register(api, huma.Operation{
		OperationID: "getSpend",
		Method:      http.MethodGet,
		Path:        "/v1/stats/spend",
		Summary:     "What the calling customer spent, grouped",
		Description: "Spend across every modality at once, which is what a bill is. group_by decides what the " +
			"series are: \"modality\" for where the money went, or a cost label for what it was spent " +
			"on.\n" +
			"Only the biggest values keep a series of their own, because a label such as customer_id " +
			"has as many values as the customer has customers. The rest are summed into \"other\", and " +
			"spend carrying no such label at all into the empty value, so the rows still add up to " +
			"the total.\n" +
			"Reads the request rows rather than the rollups, so today's spend is there without a " +
			"rollup having run.",
		// Declared rather than read off the input, so the default is documented without
		// being filled in: the handler tells a parameter left out from one sent.
		Parameters: []*huma.Param{
			{Name: "group_by", In: "query", Description: "\"modality\", or the cost label to group by.", Schema: &huma.Schema{Type: huma.TypeString, Default: "modality"}},
			{Name: "limit", In: "query", Description: "How many values keep a series of their own.", Schema: &huma.Schema{Type: huma.TypeInteger, Minimum: bound(1), Maximum: bound(50), Default: 6}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket and group, oldest bucket first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getSpend)
	huma.Register(api, huma.Operation{
		OperationID: "getTagKeys",
		Method:      http.MethodGet,
		Path:        "/v1/stats/tags/keys",
		Summary:     "Which cost labels the calling customer's spend carries",
		Description: "Cost labels are the customer's own, so nothing here knows in advance whether spend is " +
			"broken down by product, by environment or by the end customer it was incurred for. This " +
			"reports the keys in use and what each covers, so a reader can be shown the breakdown " +
			"that means something rather than a list to guess from.\n" +
			"A key every request carries with a single value -- environment: production and nothing " +
			"else -- is context rather than a breakdown, and value_count says so.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per label key, largest spend first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getTagKeys)
	huma.Register(api, huma.Operation{
		OperationID: "getActivity",
		Method:      http.MethodGet,
		Path:        "/v1/stats/activity",
		Summary:     "Who used the calling customer's agents, and how much",
		Description: "Sessions opened, responses produced and calls held, counted per bucket, alongside how " +
			"many distinct people were behind them.\n" +
			"Distinct users are counted rather than summed, which is why the granularity here is " +
			"days or months rather than the hours the spend paths take: a month's active users are " +
			"the people who came back, not the sum of its days, so a month has to be asked for as a " +
			"month.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getActivity)
	huma.Register(api, huma.Operation{
		OperationID: "runRollup",
		Method:      http.MethodPost,
		Path:        "/v1/stats/rollup",
		Summary:     "Aggregate request rows into a rollup table",
		Description: "Covers every modality and customer in the window. Idempotent: re-running it over the " +
			"same window recomputes those buckets, so a missed run is fixed by running it again.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The rollup completed"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.runRollup)
}

type listProvidersRequest struct {
	Modality Modality `path:"modality" doc:"Which kind of model to route."`
}

type listProvidersResponse struct {
	Body []Provider `nullable:"false"`
}

type listRoutesRequest struct {
	Modality Modality `path:"modality" doc:"Which kind of model to route."`
}

type listRoutesResponse struct {
	Body []Route `nullable:"false"`
}

type resolveTargetRequest struct {
	Modality Modality                `path:"modality" doc:"Which kind of model to route."`
	Target   string                  `path:"target" doc:"A \"provider/model\" name or a capability shortcut such as en-low-latency."`
	Language optionalParam[[]string] `query:"language,explode" doc:"Language hints that candidates must cover. Repeat for several."`
}

type resolveTargetResponse struct {
	Body []Candidate `nullable:"false"`
}

type getStatsRequest struct {
	Modality    Modality                   `path:"modality" doc:"Which kind of model to route."`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" doc:"Start of the window, inclusive." required:"true"`
	To          time.Time                  `query:"to" doc:"End of the window, exclusive." required:"true"`
	Tag         optionalParam[[]string]    `query:"tag,explode" doc:"Only count requests carrying every one of these cost labels, each written \"key:value\". Repeat for several. Filtering reads the request rows rather than the rollups, since a rollup bucket no longer knows which labels its requests carried."`
}

type getStatsResponse struct {
	Body []StatsBucket `nullable:"false"`
}

type getTagStatsRequest struct {
	Modality    Modality                   `path:"modality" doc:"Which kind of model to route."`
	Key         string                     `query:"key" doc:"The cost label to group by." required:"true"`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" doc:"Start of the window, inclusive." required:"true"`
	To          time.Time                  `query:"to" doc:"End of the window, exclusive." required:"true"`
}

type getTagStatsResponse struct {
	Body []TagStatsBucket `nullable:"false"`
}

type getTurnStatsRequest struct {
	AgentId     optionalParam[string]      `query:"agent_id" doc:"Narrow to one agent. Omit for every agent the customer runs."`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" doc:"Start of the window, inclusive." required:"true"`
	To          time.Time                  `query:"to" doc:"End of the window, exclusive." required:"true"`
}

type getTurnStatsResponse struct {
	Body []TurnStatsBucket `nullable:"false"`
}

type getSpendRequest struct {
	GroupBy     optionalParam[string]      `query:"group_by" doc:"\"modality\", or the cost label to group by."`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" doc:"Start of the window, inclusive." required:"true"`
	To          time.Time                  `query:"to" doc:"End of the window, exclusive." required:"true"`
	Limit       optionalParam[int]         `query:"limit" doc:"How many values keep a series of their own." minimum:"1" maximum:"50"`
	Tag         optionalParam[[]string]    `query:"tag,explode" doc:"Only count requests carrying every one of these cost labels, each written \"key:value\". Repeat for several."`
}

type getSpendResponse struct {
	Body []SpendBucket `nullable:"false"`
}

type getTagKeysRequest struct {
	From time.Time               `query:"from" doc:"Start of the window, inclusive." required:"true"`
	To   time.Time               `query:"to" doc:"End of the window, exclusive." required:"true"`
	Tag  optionalParam[[]string] `query:"tag,explode" doc:"Only consider requests carrying every one of these cost labels, each written \"key:value\". Repeat for several."`
}

type getTagKeysResponse struct {
	Body []TagKeySummary `nullable:"false"`
}

type getActivityRequest struct {
	Granularity optionalParam[ActivityGranularity] `query:"granularity"`
	From        time.Time                          `query:"from" doc:"Start of the window, inclusive." required:"true"`
	To          time.Time                          `query:"to" doc:"End of the window, exclusive." required:"true"`
}

type getActivityResponse struct {
	Body []ActivityBucket `nullable:"false"`
}

type runRollupRequest struct {
	Body *RollupRequest `required:"true"`
}

type runRollupResponse struct {
	Body RollupResult
}

// ActivityBucket is the ActivityBucket schema.
type ActivityBucket struct {
	ActiveUsers  int64     `json:"active_users" doc:"Distinct end users who opened a session or asked something of an agent in the bucket. A guest who later turned out to be a known user counts as that user.\nA caller that named nobody is not counted, and neither is an anonymous one: an anonymous name is a claim nothing verified, so counting it would make guessing a name enough to inflate this."`
	Bucket       time.Time `json:"bucket"`
	Calls        int64     `json:"calls"`
	Messages     int64     `json:"messages" doc:"Responses the agents produced, which is one per thing asked of them."`
	PhoneMinutes float64   `json:"phone_minutes" doc:"The part of voice_minutes that arrived over a phone number."`
	Sessions     int64     `json:"sessions"`
	VoiceMinutes float64   `json:"voice_minutes" doc:"How long those calls lasted. One still running counts up to now."`
}

// ActivityGranularity Separate from Granularity, and coarser, because distinct users cannot be summed: a month of them is who came back rather than the sum of its days.
type ActivityGranularity string

// Defines values for ActivityGranularity.
const (
	ActivityGranularityDaily   ActivityGranularity = "daily"
	ActivityGranularityMonthly ActivityGranularity = "monthly"
)

// Valid indicates whether the value is a known member of the ActivityGranularity enum.
func (e ActivityGranularity) Valid() bool {
	switch e {
	case ActivityGranularityDaily:
		return true
	case ActivityGranularityMonthly:
		return true
	default:
		return false
	}
}

func (ActivityGranularity) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "ActivityGranularity", "Separate from Granularity, and coarser, because distinct users cannot be summed: a month of them is who came back rather than the sum of its days.", "daily", "monthly")
	registry.Map()["ActivityGranularity"].Default = "daily"
	return ref
}

// Candidate is the Candidate schema.
type Candidate struct {
	Health   ProviderHealth `json:"health"`
	Model    string         `json:"model"`
	Provider string         `json:"provider"`
}

// Granularity is the Granularity schema.
type Granularity string

// Defines values for Granularity.
const (
	GranularityDaily  Granularity = "daily"
	GranularityHourly Granularity = "hourly"
)

// Valid indicates whether the value is a known member of the Granularity enum.
func (e Granularity) Valid() bool {
	switch e {
	case GranularityDaily:
		return true
	case GranularityHourly:
		return true
	default:
		return false
	}
}

func (Granularity) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "Granularity", "", "hourly", "daily")
	registry.Map()["Granularity"].Default = "hourly"
	return ref
}

// ProviderBenchmark What Artificial Analysis measured for this model, refreshed by hand rather than live. A field is absent when the model was not measured on it.
type ProviderBenchmark struct {
	CharactersPerSecond   *float64 `json:"characters_per_second,omitempty" doc:"Characters a text-to-speech model synthesises per second on the vendor's API." example:"115"`
	CostPerTask           *float64 `json:"cost_per_task,omitempty" doc:"US dollars one task of the search benchmark cost, searches and the answering model's tokens together." example:"0.127"`
	Elo                   *int     `json:"elo,omitempty" doc:"Speech arena Elo rating of a text-to-speech model." example:"1273"`
	IntelligenceIndex     *int     `json:"intelligence_index,omitempty" doc:"Artificial Analysis Intelligence Index of a text model at the reasoning effort the router asks for." example:"33"`
	LatencyMs             *int     `json:"latency_ms,omitempty" doc:"Milliseconds a speech-to-text model takes to its final transcript after speech ends." example:"490"`
	OutputTokensPerSecond *float64 `json:"output_tokens_per_second,omitempty" doc:"Tokens a text model writes per second on the host the router calls." example:"330"`
	SearchIndex           *int     `json:"search_index,omitempty" doc:"Artificial Analysis Search Index of a search provider, from 0 to 100." example:"74"`
	WordErrorRate         *float64 `json:"word_error_rate,omitempty" doc:"Streaming AA-WER of a speech-to-text model, from 0 to 1." example:"0.027"`
}

func (*ProviderBenchmark) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["elo"].Format = ""
	schema.Properties["intelligence_index"].Format = ""
	schema.Properties["latency_ms"].Format = ""
	schema.Properties["search_index"].Format = ""
	schema.Description = "What Artificial Analysis measured for this model, refreshed by hand rather than live. A field is absent when the model was not measured on it."
	return schema
}

// ProviderHealth is the ProviderHealth schema.
type ProviderHealth struct {
	Available    bool    `json:"available" doc:"False once the error rate crosses the configured threshold."`
	ErrorRate    float64 `json:"error_rate"`
	Errors       int64   `json:"errors"`
	LatencyMsAvg float64 `json:"latency_ms_avg"`
	Requests     int64   `json:"requests" doc:"Requests seen in the current health window."`
}

// ProviderPrice What this deployment is billed for the model, in US dollars. A rate is absent when the model is not billed by that unit.
type ProviderPrice struct {
	PerMillionInputTokens  *float64 `json:"per_million_input_tokens,omitempty" example:"0.75"`
	PerMillionOutputTokens *float64 `json:"per_million_output_tokens,omitempty" example:"3.75"`
}

func (*ProviderPrice) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What this deployment is billed for the model, in US dollars. A rate is absent when the model is not billed by that unit."
	return schema
}

// RollupRequest is the RollupRequest schema.
type RollupRequest struct {
	From        time.Time    `json:"from"`
	Granularity *Granularity `json:"granularity,omitempty"`
	To          time.Time    `json:"to"`
}

// RollupResult is the RollupResult schema.
type RollupResult struct {
	BucketsWritten int64       `json:"buckets_written"`
	Granularity    Granularity `json:"granularity"`
}

// SpendBucket is the SpendBucket schema.
type SpendBucket struct {
	Bucket          time.Time `json:"bucket"`
	CostMicrosTotal int64     `json:"cost_micros_total" doc:"Millionths of a dollar, priced from the configured rates."`
	RequestCount    int64     `json:"request_count"`
	Value           string    `json:"value" doc:"The modality or label value this row is for. \"other\" is everything outside the biggest few, and the empty string is spend carrying no such label at all, so a customer that labels only part of its traffic can see which part." example:"support"`
}

// StatsBucket is the StatsBucket schema.
type StatsBucket struct {
	AudioMsTotal           int64     `json:"audio_ms_total" doc:"Billable audio, transcribed or produced."`
	Bucket                 time.Time `json:"bucket"`
	CachedInputTokensTotal int64     `json:"cached_input_tokens_total" doc:"The part of the prompt served from the provider's cache."`
	CharactersTotal        int64     `json:"characters_total" doc:"Billable text. Zero for providers that bill by audio."`
	CostMicrosTotal        int64     `json:"cost_micros_total" doc:"Millionths of a dollar, priced from the configured rates."`
	ErrorCount             int64     `json:"error_count"`
	ImagesTotal            int64     `json:"images_total" doc:"Pictures drawn. Zero outside image."`
	InputTokensTotal       int64     `json:"input_tokens_total" doc:"Prompt tokens read, cached ones included. Zero outside llm."`
	LatencyP50Ms           *float64  `json:"latency_p50_ms,omitempty" nullable:"true"`
	LatencyP95Ms           *float64  `json:"latency_p95_ms,omitempty" nullable:"true"`
	Model                  string    `json:"model"`
	OutputTokensTotal      int64     `json:"output_tokens_total" doc:"Generated tokens, reasoning included. Zero outside llm."`
	Provider               string    `json:"provider"`
	RequestCount           int64     `json:"request_count"`
	Uptime                 *float64  `json:"uptime,omitempty" doc:"Successes over total requests in the bucket." nullable:"true"`
}

// TagKeySummary is the TagKeySummary schema.
type TagKeySummary struct {
	CostMicrosTotal int64             `json:"cost_micros_total"`
	Coverage        float64           `json:"coverage" doc:"The share of the window's requests that carry this key, from 0 to 1. A key on half the traffic breaks down half the bill, which is worth knowing before it is read as the whole of it."`
	Key             string            `json:"key" example:"product"`
	RequestCount    int64             `json:"request_count"`
	TopValues       []TagValueSummary `json:"top_values" doc:"The ten largest values, biggest spend first." nullable:"false"`
	ValueCount      int64             `json:"value_count" doc:"How many distinct values the key was used with. One means it is context rather than a breakdown; hundreds mean it identifies something, such as an end customer, and only its largest values are worth a chart."`
}

// TagStatsBucket is the TagStatsBucket schema.
type TagStatsBucket struct {
	AudioMsTotal           int64     `json:"audio_ms_total"`
	Bucket                 time.Time `json:"bucket"`
	CachedInputTokensTotal int64     `json:"cached_input_tokens_total"`
	CharactersTotal        int64     `json:"characters_total"`
	CostMicrosTotal        int64     `json:"cost_micros_total" doc:"Millionths of a dollar, priced from the configured rates."`
	ErrorCount             int64     `json:"error_count"`
	ImagesTotal            int64     `json:"images_total"`
	InputTokensTotal       int64     `json:"input_tokens_total"`
	LatencyP50Ms           *float64  `json:"latency_p50_ms,omitempty" nullable:"true"`
	LatencyP95Ms           *float64  `json:"latency_p95_ms,omitempty" nullable:"true"`
	OutputTokensTotal      int64     `json:"output_tokens_total"`
	RequestCount           int64     `json:"request_count"`
	TagKey                 string    `json:"tag_key" example:"project"`
	TagValue               string    `json:"tag_value" example:"moderation"`
	Uptime                 *float64  `json:"uptime,omitempty" nullable:"true"`
}

// TagValueSummary is the TagValueSummary schema.
type TagValueSummary struct {
	CostMicrosTotal int64   `json:"cost_micros_total"`
	RequestCount    int64   `json:"request_count"`
	Share           float64 `json:"share" doc:"This value's share of what the key covers, from 0 to 1."`
	Value           string  `json:"value" example:"support"`
}

// Tier What the model optimises for.
type Tier string

// Defines values for Tier.
const (
	HighQuality Tier = "high-quality"
	LowLatency  Tier = "low-latency"
)

// Valid indicates whether the value is a known member of the Tier enum.
func (e Tier) Valid() bool {
	switch e {
	case HighQuality:
		return true
	case LowLatency:
		return true
	default:
		return false
	}
}

func (Tier) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Tier", "What the model optimises for.", "low-latency", "high-quality")
}

// TurnStatsBucket is the TurnStatsBucket schema.
type TurnStatsBucket struct {
	AgentId          string    `json:"agent_id"`
	AudioOutMsTotal  float64   `json:"audio_out_ms_total" doc:"How much speech the agent published in the bucket."`
	Bucket           time.Time `json:"bucket"`
	InterruptedCount int64     `json:"interrupted_count" doc:"Turns a participant talked over before they finished."`
	LlmTtftP50Ms     *float64  `json:"llm_ttft_p50_ms,omitempty" nullable:"true"`
	LlmTtftP95Ms     *float64  `json:"llm_ttft_p95_ms,omitempty" nullable:"true"`
	RoundtripP50Ms   *float64  `json:"roundtrip_p50_ms,omitempty" doc:"Settled transcript to first audio published." nullable:"true"`
	RoundtripP95Ms   *float64  `json:"roundtrip_p95_ms,omitempty" nullable:"true"`
	RoundtripP99Ms   *float64  `json:"roundtrip_p99_ms,omitempty" nullable:"true"`
	SttLatencyP50Ms  *float64  `json:"stt_latency_p50_ms,omitempty" nullable:"true"`
	SttLatencyP95Ms  *float64  `json:"stt_latency_p95_ms,omitempty" nullable:"true"`
	TtsTtfbP50Ms     *float64  `json:"tts_ttfb_p50_ms,omitempty" nullable:"true"`
	TtsTtfbP95Ms     *float64  `json:"tts_ttfb_p95_ms,omitempty" nullable:"true"`
	TurnCount        int64     `json:"turn_count"`
}
