//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/tls"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/hibiken/asynq"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/siptrunk"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// suiteConnections caps each suite's pool, so ten suites at once stay well inside the
// hundred connections Postgres allows.
const suiteConnections = 5

// settleFor is how long a test waits for something crossing a socket or a queue to arrive.
const settleFor = 5 * time.Second

// migrated runs the migrations once for every suite in the run: they set goose's globals
// and attach triggers, neither of which two suites may do at once.
var migrated struct {
	sync.Once
	err error
}

// redisDB hands each suite a Redis database of its own, so the queues one suite works
// through are not read by another. Redis serves sixteen; the first is left to whatever
// else is on the machine.
var redisDB atomic.Int32

// runSuite runs a suite beside the others. Every suite has its own router, sessions,
// queues and connections, and shares only Postgres, where every row a test makes has an id
// of its own.
func runSuite(t *testing.T, s suite.TestingSuite) {
	t.Parallel()
	suite.Run(t, s)
}

// suiteKEK seals the secrets of the keys the suite makes. It is the same for every suite,
// so a fixture one suite loaded opens in another.
const suiteKEK = "router suite"

// The Stream app the suite mints call tokens for. Minting signs a token rather than
// fetching one, so a made-up app is enough to exercise the join paths.
const (
	suiteStreamKey    = "suite-stream-key"
	suiteStreamSecret = "suite-stream-secret"
	// suiteStreamApp is that app's id, which the router knows, so work pinned to any other
	// app is recognised as somebody else's.
	suiteStreamApp = 1
)

// suiteOpsKey is what Stream staff's review paths are reached with.
const suiteOpsKey = "suite-ops-key"

// RouterSuite runs the whole router against Postgres and Redis with real API key auth, for
// suites to embed. A suite picks the app its clients call in its SetupTest:
// s.useFixture("standard") for the app most tests share, or s.useApp(s.data.createApp())
// for one of its own.
//
// Nothing is cleaned up. Every row a test makes has a fresh UUID, so nothing an earlier
// test or run left in Postgres gets in its way.
type RouterSuite struct {
	suite.Suite

	store   *store.Store
	configs *appconfig.Store
	live    *live.Client
	sealer  *auth.Sealer
	// manager runs the suite's sessions, for a test about what ends them from inside.
	manager *session.Manager

	// appMode runs the suite's Stream through app mode's own source, over the suite's
	// database and keyring, with the deployment's app as the fallback. A suite sets it, and
	// the two after it, before SetupSuite runs.
	appMode bool
	// appRefuses turns app mode's fallback off, so a customer with no app of its own is
	// written nowhere.
	appRefuses bool
	// trustAPIKeyHeader lets X-Stream-Api-Key choose the minting key.
	trustAPIKeyHeader bool
	// denied are the app ids the suite refuses registration to.
	denied []string
	server *httptest.Server
	app    testApp

	// streams, modalities and conversations are what the suite's router was built from,
	// kept so that otherNode can build a second one over the same routing.
	streams       *Streams
	modalities    map[routing.Modality]routing.Inspector
	conversations *conversation.Service
	// relayPrefix names this suite's relay channels and the keys saying which of its
	// nodes runs what. Pub/sub ignores the database number the rest of a suite's keys are
	// kept apart by, so this is the only thing stopping one suite's nodes from reading
	// another's sessions.
	relayPrefix string

	// unauthenticatedClient sends no credentials. The rest hold the app's key:
	// anonymousClient goes by a name nothing proves, guestClient and client are signed-in
	// end users, a guest and a permanent one, and serverClient is the app's own backend.
	unauthenticatedClient *testClient
	anonymousClient       *testClient
	guestClient           *testClient
	client                *testClient
	serverClient          *testClient

	// The stubs standing in for providers, for a test to read back what the router asked
	// them for. model answers questions, vision is the one that can see, voice keeps what
	// it was told to say, knowledge keeps the passages written to it. holding answers once a
	// test lets it, for a summary that has to still be in the writing.
	model     *scriptedLLM
	vision    *scriptedLLM
	holding   *scriptedLLM
	voice     *recordingTTS
	ears      *quietSTT
	knowledge *knowledgeBase
	memories  *keptMemories
	// noted are the "noted" models opened so far, one per session, newest last, for a test
	// whose sessions each need a model of their own to read back.
	notedMu sync.Mutex
	noted   []*scriptedLLM

	// chat is the Stream Chat the conversations are written to and transcripts read from:
	// the deployment's own app. apps gives a customer an app of its own instead, and
	// stream is what the API resolves through.
	chat   *chattest.Server
	apps   *suiteApps
	stream *streamapp.Clients

	// dispatch is the pool the hooks hand an arriving call or message to, for a test to
	// register a worker in and read back what it was given.
	dispatch *dispatch.Pool

	// messagesPerDay is what one end user may spend, for a suite about the cap to set
	// before it starts the harness. Zero is a cap nothing reaches.
	messagesPerDay int64

	// memoryStore is where sessions remember, for a suite about memory to set before it
	// starts the harness. Nil keeps them in memories.
	memoryStore memory.Store

	// connectors is what connector adapters the router has, for a suite about connectors to
	// set before it starts the harness. Empty has none.
	connectors core.Registry
	// connectorsOff gives the router no connector keyring, transports or limiter, as a deployment with
	// ROUTER_CONNECTORS_ENABLED unset has, for a control suite to set before it starts the
	// harness. The suite's store, resolver and sealer are still built.
	connectorsOff bool
	// eventSecrets and bridge are the connector events endpoint's secrets and channel
	// bridge, for a suite about connector events to set before it starts the harness. Nil
	// takes no events and drops messages, as a deployment without them does.
	eventSecrets EventSecretLookup
	bridge       ChannelBridge
	// slackApps and operatorApps are what the provider app paths reach Slack with and find
	// the operator's app by, for a suite about provider apps to set before it starts the
	// harness. Nil leaves those paths unconfigured, as a deployment without connectors has.
	slackApps    *slackapps.Client
	operatorApps OperatorAppLookup
	// channelProvider, set by a suite about the channel bridge before it starts the harness,
	// gives the router the real bridge (internal/channelbridge), whose replies dial the address
	// it returns at the time, whatever host a manifest's reply names: the test's own fake
	// provider. Nil leaves bridge as the suite set it.
	channelProvider func() string
	// logs, set by a suite before it starts the harness, is where the router logs, as text,
	// for a suite about what it warns of. Nil discards them.
	logs io.Writer
	// episodes is the router's closer, for a suite to sweep the idle thread episodes with.
	episodes *omnichannel.Closer
	// transcripts, set by a suite before it starts the harness, opens each voice session's
	// transcript, as cmd/router's chatlog does. Nil keeps none, as the other suites do.
	transcripts session.TranscriptFactory
	// forwardHTTP, set by a suite about event forwarding before it starts the harness, gives
	// the router an event forwarder (internal/eventforward) whose sends go through it, to the
	// test's own destinations on loopback, which egress refuses; a destination URL on loopback
	// is let through at create, any other is checked by egress. Nil leaves forwarding off, as
	// a deployment without connectors has, and runs no forwarder worker beside the other
	// suites'.
	forwardHTTP *http.Client
	forwarder   *eventforward.Forwarder
	// mcpEventsOn, set by a suite about MCP events before it starts the harness, gives the
	// router an MCP events service (internal/mcpevents) whose callbacks are the suite's own
	// router, reached on loopback over http. Off leaves it absent, as a deployment with
	// connectors off has, and runs no worker beside the other suites'.
	mcpEventsOn bool
	mcpEvents   *mcpevents.Service
	// transports is the connections' outbound clients the router was built with.
	transports *core.Transports
	// resolver is the router's connector resolver over the suite's store and sealer, with
	// connectors' schemes, set by SetupSuite.
	resolver *resolver.Resolver
	// connectorHTTP is what each connection's client sends through, for a suite whose
	// providers listen on loopback, which egress refuses, to set before it starts the harness.
	// Nil is egress's, as in the router.
	connectorHTTP *http.Client
	// publicURL and dashboardURL are the router's ROUTER_PUBLIC_URL and DASHBOARD_BASE_URL,
	// for a suite about connector consents to set before it starts the harness. Empty leaves
	// them unset, as a deployment that never set them has.
	publicURL    string
	dashboardURL string

	// pluginMCP is where every plugin's MCP server is reached, for a suite about plugin
	// events to set before it starts the harness. Nil reaches the real ones.
	pluginMCP *httptest.Server
	// pluginHTTP is what the API reaches plugins with, for a suite whose stand-ins listen
	// on loopback in each test. Nil is mcpTransport.
	pluginHTTP *http.Client
	// events subscribes configs to their plugins' events.
	events *pluginevents.Service

	// channelAPI is where every channel provider is reached, for a suite about channels to
	// set before it starts the harness. Nil reaches the real WhatsApp, Telnyx and Linq.
	channelAPI *httptest.Server
	// inbound answers what arrives on the app's channels.
	inbound *channels.Service

	// sandbox is what an app with no approved use case may do, for a suite about the
	// sandbox to set before it starts the harness. The zero value is off.
	sandbox dlc.Sandbox
	// registrar is who registers 10DLC campaigns, for a suite about registration to set
	// before it starts the harness. Nil makes Stream's approval final.
	registrar dlc.Registrar
	// gate is what every text and call passes.
	gate *dlc.Gate

	utils testUtils
	data  testData
}

// testDatabase is the suites' own database, whatever a local router is pointed at. These
// tests write apps, sessions and change rows by the hundred, and a running router reading
// the same tables is a router answering with a test's data.
func testDatabase(t *testing.T, dsn string) string {
	parsed, err := url.Parse(dsn)
	require.NoError(t, err)
	name := strings.TrimPrefix(parsed.Path, "/")
	if strings.HasSuffix(name, "_test") {
		return dsn
	}
	parsed.Path = "/" + name + "_test"
	own := parsed.String()

	parsed.Path = "/postgres"
	admin := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(parsed.String())))
	defer admin.Close() //nolint:errcheck // the error on the way out says nothing
	_, err = admin.ExecContext(context.Background(), `CREATE DATABASE "`+name+`_test"`)
	if err != nil && !strings.Contains(err.Error(), "already exists") {
		require.NoError(t, err)
	}
	return own
}

// testApp is an app with an organization and an API key to call it with.
type testApp struct {
	organization store.Organization
	app          store.App
	key, secret  string
}

func (s *RouterSuite) SetupSuite() {
	dsn, redisAddr := os.Getenv("ROUTER_POSTGRES_DSN"), os.Getenv("ROUTER_REDIS_ADDR")
	if dsn == "" || redisAddr == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN and ROUTER_REDIS_ADDR must be set")
	}
	ctx := context.Background()
	logger := slog.New(slog.DiscardHandler)
	if s.logs != nil {
		logger = slog.New(slog.NewTextHandler(s.logs, nil))
	}

	pgStore, err := store.Open(testDatabase(s.T(), dsn))
	s.Require().NoError(err)
	pgStore.DB().SetMaxOpenConns(suiteConnections)
	migrated.Do(func() { migrated.err = pgStore.Migrate(ctx) })
	s.Require().NoError(migrated.err)
	s.store = pgStore
	s.T().Cleanup(func() { s.Require().NoError(pgStore.Close()) })

	liveClient, err := live.New(live.Options{Address: redisAddr})
	s.Require().NoError(err)
	s.live = liveClient
	s.T().Cleanup(liveClient.Close)

	s.sealer, err = auth.NewSealer(suiteKEK)
	s.Require().NoError(err)
	s.data = testData{suite: s}

	s.chat = chattest.NewServer(s.T())
	s.apps = &suiteApps{own: map[string]streamapp.Identity{}, readOnly: map[string]bool{}, waiting: map[string]bool{}, nowhere: map[string]bool{}, deployment: streamapp.NewDeployment(streamapp.DeploymentOptions{
		APIKey: suiteStreamKey, Secret: suiteStreamSecret, BaseURL: s.chat.URL, App: suiteStreamApp,
	})}
	s.stream = streamapp.NewClients(s.apps, streamapp.ClientsOptions{})
	if s.appMode {
		stored, err := streamapp.NewStored(streamapp.StoredOptions{
			Store: pgStore, Sealer: s.sealer, Deployment: s.apps.deployment, FallbackToDeployment: !s.appRefuses, Logger: logger,
		})
		s.Require().NoError(err)
		s.stream = streamapp.NewClients(stored, streamapp.ClientsOptions{})
	}
	s.store.SetStreamPins(store.StreamPins{Deployment: s.apps.deployment.App, For: s.stream.Pin, Knowable: s.stream.DeploymentAppKnowable})
	s.configs, err = appconfig.New(appconfig.Options{
		Store: pgStore, Address: redisAddr, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(s.configs.Close)
	s.relayPrefix = uuid.NewString()

	limiter := s.quota(liveClient, logger)
	policies := s.policies(logger)
	streams := s.routers(limiter, policies, logger)
	// Built before it is served on, because a node says where it is before it has
	// anything to say it is running.
	listener := httptest.NewUnstartedServer(nil)
	directory := s.nodeDirectory(listener, logger)
	// Before the sessions, whose dispatcher shares the transports with the validate
	// endpoint, as cmd/router builds them.
	credentials, err := pgsealed.New(pgStore, s.sealer)
	s.Require().NoError(err)
	s.resolver, err = resolver.New(resolver.Config{Store: pgStore, Credentials: credentials, Schemes: s.connectors.Schemes})
	s.Require().NoError(err)
	transports, err := core.NewTransports(core.TransportsConfig{Resolver: s.resolver, Timeout: suiteConnectorTimeout,
		NewClient: loopbackClients(s.connectorHTTP)})
	s.Require().NoError(err)
	s.transports = transports
	sessions := s.sessionManager(streams, directory, session.Connectors{Registry: s.connectors, Transports: transports,
		Consents: ConnectorConsents(pgStore, s.connectors, s.sealer, s.publicURL)}, logger)
	// A nil client reaches public hosts alone, and every auth server here is a local one.
	public := &plugins.Auth{HTTP: http.DefaultClient}
	s.events = s.pluginEvents(sessions, public, logger)
	s.gate = dlc.NewGate(pgStore, liveClient.Redis(), s.sandbox, logger)
	s.inbound = s.channels(sessions, logger)
	s.dispatch = dispatch.NewPool()
	s.streams = streams
	s.modalities = map[routing.Modality]routing.Inspector{
		routing.LLM:    streams.LLM,
		routing.STT:    streams.STT,
		routing.TTS:    streams.TTS,
		routing.STS:    streams.STS,
		routing.Search: streams.Search,
		routing.LCM:    streams.LCM,
		routing.Image:  streams.Image,
	}

	if s.channelProvider != nil {
		s.bridge = s.channelBridge(logger)
	}
	if s.forwardHTTP != nil {
		s.forwarder, err = eventforward.New(eventforward.Options{
			Store: pgStore, Secrets: s.sealer, HTTP: s.forwardHTTP, PublicURL: loopbackOrPublic, Logger: logger,
			// A forward sent again waits milliseconds here, not the production seconds.
			Retries: []time.Duration{10 * time.Millisecond, 10 * time.Millisecond}, Poll: 10 * time.Millisecond,
		})
		s.Require().NoError(err)
		s.forwarder.Start()
		s.T().Cleanup(s.forwarder.Close)
	}

	if s.mcpEventsOn {
		s.mcpEvents, err = mcpevents.New(mcpevents.Options{
			Store: pgStore, Sessions: sessions, Registry: s.connectors, Transports: transports, Secrets: s.sealer,
			PublicURL: "http://" + listener.Listener.Addr().String(), Logger: logger,
			// A second, not the production minute, so a test's look by this worker comes and
			// saves within its window, and is longer than an attempt against the fake.
			Lease: time.Second,
		})
		s.Require().NoError(err)
		s.mcpEvents.Start()
		s.T().Cleanup(s.mcpEvents.Close)
	}

	// The closer cmd/router builds wherever there is a store and an LLM router, which the call
	// hook ends a call's episodes with. Its idle sweeper is not started: cmd/router starts it
	// only with connectors on or a config with episode_cards (startEpisodeSweeper).
	episodes, err := omnichannel.NewCloser(omnichannel.CloserOptions{
		Store: pgStore, Stream: s.stream, LLM: streams.LLM, IdleAfter: time.Hour, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(episodes.Close)
	s.episodes = episodes

	// cmd/router passes no connector keyring with connectors off (main.go, connectorSecrets).
	connectorSecrets := s.sealer
	// cmd/router builds no transports and no limiter with connectors off either
	// (newConnectorTransports, newConnectorLimiter).
	serverTransports, connectorLimiter := transports, core.NewLimiter(liveClient.Redis(), nil)
	if s.connectorsOff {
		connectorSecrets, serverTransports, connectorLimiter = nil, nil, nil
	}
	server, err := NewServer(Options{
		Routers:       s.modalities,
		Streams:       streams,
		Relay:         s.relayBus(logger),
		Directory:     directory,
		Sessions:      sessions,
		Store:         pgStore,
		Configs:       s.configs,
		Live:          liveClient,
		Auth:          s.authenticator(),
		AuthMode:      auth.APIKey,
		Phone:         s.telephony(logger),
		Campaigns:     s.campaigns(sessions, logger),
		Simulations:   s.simulations(sessions, streams, logger),
		PluginEvents:  s.events,
		Channels:      s.inbound,
		DLC:           s.registrations(listener, logger),
		Gate:          s.gate,
		OpsKey:        suiteOpsKey,
		Secrets:       s.sealer,
		Knowledge:     s.knowledgeWriter(),
		KnowledgeURLs: s.pages(redisAddr),
		Voices:        s.voiceService(),
		VoiceLibrary:  voices.NewCatalogue(),
		Dispatch:      s.dispatch,
		Policies:      policies,
		Connectors:    s.connectors,
		// Connector consents and credentials seal under the suite's key.
		ConnectorSecrets:  connectorSecrets,
		PublicURL:         s.publicURL,
		DashboardURL:      s.dashboardURL,
		Quota:             limiter,
		DataRetention:     time.Hour,
		PluginHTTP:        s.mcpTransport(),
		Stream:            s.stream,
		HookSecret:        suiteStreamSecret,
		TrustAPIKeyHeader: s.trustAPIKeyHeader,
		DenyRegistration:  s.denied,
		Logger:            logger,
		// The connector events endpoint revokes through the suite's resolver.
		ConnectorResolver:     s.resolver,
		ConnectorTransports:   serverTransports,
		ConnectorLimiter:      connectorLimiter,
		ConnectorEventSecrets: s.eventSecrets,
		ChannelBridge:         s.bridge,
		EventForwarder:        s.forwarder,
		MCPEvents:             s.mcpEvents,
		Episodes:              episodes,
		SlackApps:             s.slackApps,
		OperatorProviderApps:  s.operatorApps,
	})
	s.Require().NoError(err)
	listener.Config.Handler = server.Handler()
	listener.Start()
	public.PublicURL = listener.URL
	s.server = listener
	s.T().Cleanup(s.server.Close)
}

// channelBridge is the router's channel bridge over the suite's store, Stream Chat and
// resolver, whose replies leave through core.Transports, as cmd/router builds it, with a
// client that dials channelProvider instead of the egress client, which refuses loopback.
func (s *RouterSuite) channelBridge(logger *slog.Logger) *channelbridge.Bridge {
	transports, err := core.NewTransports(core.TransportsConfig{
		Resolver: s.resolver,
		NewClient: func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
			base := &http.Transport{
				DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, network, s.channelProvider())
				},
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // the test's own fake provider
			}
			return &http.Client{Timeout: timeout, Transport: wrap(base)}
		},
	})
	s.Require().NoError(err)
	bridge, err := channelbridge.New(channelbridge.Options{
		Store: s.store, Stream: s.stream, Schemes: s.connectors.Schemes, Transports: transports, Resolver: s.resolver,
		Gate: s.gate, Logger: logger,
		// A reply sent again waits milliseconds here, not the production seconds.
		RetryBackoff: []time.Duration{10 * time.Millisecond, 10 * time.Millisecond},
	})
	s.Require().NoError(err)
	s.T().Cleanup(bridge.Close)
	// As cmd/router hands it the finished replies of the conversations on thread channels.
	s.conversations.OnFinishedReply(bridge.Reply)
	return bridge
}

// pluginEvents subscribes against pluginMCP, whatever host a catalog plugin names, so a
// suite's own server stands in for Sentry's.
func (s *RouterSuite) pluginEvents(sessions *session.Manager, public *plugins.Auth, logger *slog.Logger) *pluginevents.Service {
	var transport *http.Client
	if s.pluginMCP != nil {
		transport = s.mcpTransport()
	}
	service, err := pluginevents.New(pluginevents.Options{
		Store: s.store, Sessions: sessions, Auth: public, Transport: transport, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(service.Close)
	return service
}

// channels answers deliveries against channelAPI, whatever host a provider's API is at, so a
// test's reply is sent to the suite's own server rather than to Meta.
func (s *RouterSuite) channels(sessions *session.Manager, logger *slog.Logger) *channels.Service {
	inbound, err := channels.New(channels.Options{
		Store:     s.store,
		Sessions:  sessions,
		Secrets:   s.sealer,
		Transport: redirected(s.channelAPI),
		Gate:      s.gate,
		Logger:    logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(inbound.Close)
	return inbound
}

// registrations reviews use cases and registers them with the suite's registrar, which
// reports back to the suite's own router.
func (s *RouterSuite) registrations(listener *httptest.Server, logger *slog.Logger) *dlc.Service {
	service, err := dlc.NewService(dlc.Options{
		Store: s.store, Registrar: s.registrar, PublicURL: "http://" + listener.Listener.Addr().String(), Logger: logger,
	})
	s.Require().NoError(err)
	return service
}

// redirected dials one server whatever host is asked for, or nothing when there is none.
func redirected(to *httptest.Server) *http.Client {
	if to == nil {
		return &http.Client{Transport: &http.Transport{
			DialContext: func(context.Context, string, string) (net.Conn, error) {
				return nil, errors.New("tests reach no real provider")
			},
		}}
	}
	address := to.Listener.Addr().String()
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, address)
		},
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // the suite's own server
	}}
}

// mcpTransport reaches pluginMCP whatever host is asked for, or nothing when a suite has
// none, so saving a config never asks a real MCP server about itself.
func (s *RouterSuite) mcpTransport() *http.Client {
	if s.pluginHTTP != nil {
		return s.pluginHTTP
	}
	if s.pluginMCP == nil {
		return &http.Client{Transport: &http.Transport{
			DialContext: func(context.Context, string, string) (net.Conn, error) {
				return nil, errors.New("tests reach no real MCP server")
			},
		}}
	}
	address := s.pluginMCP.Listener.Addr().String()
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, address)
		},
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // the suite's own server
	}}
}

// routers builds every modality against stubs that answer in process. What a real vendor
// makes of real audio is that provider package's own suite; what is under test here is the
// HTTP surface in front of it.
func (s *RouterSuite) routers(limiter *quota.Limiter, gate routing.Gate, logger *slog.Logger) *Streams {
	s.ears = &quietSTT{emitter: stt.NewEmitter(64)}
	hearing := sttrouter.NewRegistry()
	hearing.Register("stub", func(routing.Spec) (stt.STT, error) { return s.ears, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: routableConfig(), Registry: hearing, Store: s.store, Live: s.live, Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	// A session opens a model for the conversation and another for its flow controller,
	// and two sessions sharing one stub would each consume the other's turns. The first
	// open is the one a test reads back.
	s.model = &scriptedLLM{reply: "Hello."}
	var opened int
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		defer func() { opened++ }()
		if opened == 0 {
			return s.model, nil
		}
		return &scriptedLLM{}, nil
	})
	s.vision = &scriptedLLM{reply: "Two roses.", sees: true}
	reasoning.Register("vision", func(routing.Spec) (llmrouter.Provider, error) { return s.vision, nil })
	reasoning.Register("echo", func(routing.Spec) (llmrouter.Provider, error) { return &scriptedLLM{echoes: true}, nil })
	reasoning.Register("recites", func(routing.Spec) (llmrouter.Provider, error) { return &scriptedLLM{recites: true}, nil })
	s.holding = &scriptedLLM{reply: "Held."}
	reasoning.Register("holding", func(routing.Spec) (llmrouter.Provider, error) { return s.holding, nil })
	reasoning.Register("noted", func(routing.Spec) (llmrouter.Provider, error) {
		opened := &scriptedLLM{reply: "Noted."}
		s.notedMu.Lock()
		defer s.notedMu.Unlock()
		s.noted = append(s.noted, opened)
		return opened, nil
	})
	reasoning.Register("summarising", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Noted.", summarises: true}, nil
	})
	reasoning.Register("counted", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Counted.", usage: llm.Usage{InputTokens: 1000, InputTokensDetails: llm.InputTokensDetails{CachedTokens: 400}, OutputTokens: 20}}, nil
	})

	// A model that is a while in the writing, for a command that has to still be running
	// when the test asks it to stop.
	reasoning.Register("slow", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Here it is.", takes: time.Second}, nil
	})

	// A model that reaches for the caller's own tool on its first turn. One of these per
	// session rather than one for the suite, so a test reads back its own turn.
	reasoning.Register("tooling", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Let me check.", calls: []llm.ToolCall{{
			ID: store.NewID(), Name: lookupOrder, Arguments: `{"order":"12"}`,
		}}}, nil
	})
	// A model that reaches for a connector's tool on its first turn: echo of the binding
	// called crm, which says back what it is given.
	reasoning.Register("connecting", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Let me ask.", calls: []llm.ToolCall{{
			ID: store.NewID(), Name: connectorEcho, Arguments: `{"text":"` + connectorEchoText + `"}`,
		}}}, nil
	})
	// A model that runs crm's echo through a waiting binding's call_tool whenever somebody
	// asks it something, a follow-up after a login included (chat_logins_test.go).
	reasoning.Register("logging-in", func(routing.Spec) (llmrouter.Provider, error) { return &loggingInLLM{}, nil })
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: reasoningConfig(), Registry: reasoning, Store: s.store, Live: s.live,
		Quota: limiter, Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	s.voice = &recordingTTS{emitter: tts.NewEmitter(64)}
	speaking := ttsrouter.NewRegistry()
	speaking.Register("stub", func(routing.Spec) (tts.TTS, error) { return s.voice, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: routableConfig(), Registry: speaking, Store: s.store, Live: s.live, Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	transcriptions, err := sttrouter.NewRecordings(sttrouter.Options{
		Config:       routableConfig(),
		Transcribers: transcriberRegistry(),
		Store:        s.store,
		Live:         s.live,
		Logger:       logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriptions.Close)

	recordings, err := ttsrouter.NewRecordings(ttsrouter.Options{
		Config:    routableConfig(),
		Recorders: recorderRegistry(),
		Store:     s.store,
		Live:      s.live,
		Logger:    logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)

	finding, err := searchrouter.New(searchrouter.Options{
		Config: routableConfig(), Registry: searchRegistry(), Store: s.store, Live: s.live, Gate: gate,
		Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(finding.Close)

	judging, err := lcmrouter.New(lcmrouter.Options{
		Config: classifyConfig(), Registry: classifierRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(judging.Close)

	imaging, err := imagerouter.New(imagerouter.Options{
		Config: imageConfig(), Registry: painterRegistry(), Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(imaging.Close)

	conversing, err := stsrouterStub()
	s.Require().NoError(err)
	s.T().Cleanup(conversing.Close)

	return &Streams{
		STT: transcriber, TTS: speaker, LLM: reasoner, STS: conversing,
		Search: finding, LCM: judging, Image: imaging,
		Transcriptions: transcriptions, Speech: recordings,
	}
}

// sessionManager runs conversations, writing them down in Stream Chat through a stand-in
// for the chat backend: what a session records is under test, what Stream does with it is
// not.
func (s *RouterSuite) sessionManager(
	streams *Streams,
	directory *node.Directory,
	connectors session.Connectors,
	logger *slog.Logger,
) *session.Manager {
	conversations := conversation.NewForChats(conversation.StreamApps(s.stream))
	s.T().Cleanup(conversations.Close)
	s.conversations = conversations

	s.memories = &keptMemories{}
	var remembering memory.Store = s.memories
	if s.memoryStore != nil {
		remembering = s.memoryStore
	}
	sessions, err := session.NewManager(session.ManagerOptions{
		LLM:           streams.LLM,
		STT:           streams.STT,
		TTS:           streams.TTS,
		Memory:        remembering,
		Transcript:    s.transcripts,
		Conversations: conversations,
		Stream:        s.stream,
		Store:         s.store,
		Configs:       s.configs,
		Directory:     directory,
		Connectors:    connectors,
		Logger:        logger,
		Edge: func(context.Context, session.Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = sessions.Shutdown() })
	s.manager = sessions
	return sessions
}

// authenticator resolves a key the way a deployment in api_key mode does: the row is read
// from Postgres and its secret unsealed, so the credentials a test holds are real ones.
func (s *RouterSuite) authenticator() auth.Authenticator {
	authenticator, err := auth.New(auth.APIKey, s.configs.Lookup(s.sealer))
	s.Require().NoError(err)
	return authenticator
}

// telephony serves the phone paths against the real vendor registry with no credentials
// behind it, which is what a deployment that has bought no numbers yet looks like.
func (s *RouterSuite) telephony(logger *slog.Logger) *phone.Service {
	// Every vendor the router knows is declared, and none of them has credentials: what a
	// vendor does with a real number is its own package's suite, and what is under test
	// here is the HTTP surface in front of them.
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)

	sealer, err := auth.NewSealer("router suite sip trunks")
	s.Require().NoError(err)

	service, err := phone.NewService(phone.ServiceOptions{
		Registry:  vendors.Registry(config),
		Store:     s.store,
		Recorder:  routing.NewRecorder(routing.Phone, s.store, s.live, logger),
		Gate:      s.gate,
		Sealer:    sealer,
		SIPTrunks: siptrunk.New(siptrunk.Options{Logger: logger}),
		Logger:    logger,
	})
	s.Require().NoError(err)
	return service
}

func (s *RouterSuite) campaigns(sessions *session.Manager, logger *slog.Logger) *campaign.Runner {
	runner, err := campaign.New(campaign.Options{
		Store: s.store, Phone: s.telephony(logger), Sessions: sessions, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	return runner
}

func (s *RouterSuite) simulations(sessions *session.Manager, streams *Streams, logger *slog.Logger) *simulation.Runner {
	runner, err := simulation.New(simulation.Options{
		Store: s.store, Sessions: sessions, LLM: streams.LLM,
		TTS: streams.TTS, STT: streams.STT, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	return runner
}

func (s *RouterSuite) knowledgeWriter() *knowledgeBase {
	s.knowledge = newKnowledgeBase()
	return s.knowledge
}

// pages keeps the knowledge bases filled from pages published elsewhere. Its queue is in a
// Redis database of its own, so neither another suite nor a router running against the
// same Redis takes the jobs for itself.
func (s *RouterSuite) pages(redisAddr string) *urls.Service {
	service, err := urls.New(urls.Options{
		Store:         s.store,
		Redis:         asynq.RedisClientOpt{Addr: redisAddr, DB: int(redisDB.Add(1))%15 + 1},
		Reader:        pageReader{},
		Writer:        newKnowledgeBase(),
		CheckInterval: 10 * time.Millisecond,
	})
	s.Require().NoError(err)
	s.Require().NoError(service.Start())
	s.T().Cleanup(func() { s.Require().NoError(service.Close()) })
	return service
}

// voiceService keeps the voices a customer brought with them in a directory of its own,
// and clones them at a provider answering in process.
func (s *RouterSuite) voiceService() *voices.Service {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/v1/text-to-speech/") {
			_, _ = w.Write([]byte("spoken"))
			return
		}
		_, _ = w.Write([]byte(`{"voice_id":"el-cloned"}`))
	}))
	s.T().Cleanup(provider.Close)

	bucket, err := blob.Open(context.Background(), "file://"+s.T().TempDir())
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(bucket.Close()) })

	cloner, err := voices.NewElevenLabs(voices.ElevenLabsOptions{APIKey: "secret", BaseURL: provider.URL})
	s.Require().NoError(err)
	cloners := voices.NewRegistry()
	cloners.Register("elevenlabs", cloner)

	service, err := voices.NewService(voices.Options{Store: s.configs, Bucket: bucket, Cloners: cloners})
	s.Require().NoError(err)
	return service
}

// otherNode is a second router over the same Postgres, Redis and relay channels as the
// suite's own, with sessions of its own.
//
// It is what a deployment of more than one process looks like: a session opened on one
// node is in the other's memory nowhere, so a socket that lands on the wrong one has to
// be served over the relay, and anything else asked of it carried to the node running it.
// The routing is shared because what differs between two nodes is which conversations
// they are holding, not what they route to.
func (s *RouterSuite) otherNode() *httptest.Server {
	logger := slog.New(slog.DiscardHandler)
	other := httptest.NewUnstartedServer(nil)
	directory := s.nodeDirectory(other, logger)
	sessions, err := session.NewManager(session.ManagerOptions{
		LLM:           s.streams.LLM,
		STT:           s.streams.STT,
		TTS:           s.streams.TTS,
		Memory:        &keptMemories{},
		Conversations: s.conversations,
		Stream:        s.stream,
		Store:         s.store,
		Configs:       s.configs,
		Directory:     directory,
		Logger:        logger,
		Edge: func(context.Context, session.Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = sessions.Shutdown() })

	server, err := NewServer(Options{
		Routers:    s.modalities,
		Streams:    s.streams,
		Sessions:   sessions,
		Relay:      s.relayBus(logger),
		Directory:  directory,
		Store:      s.store,
		Configs:    s.configs,
		Live:       s.live,
		Auth:       s.authenticator(),
		AuthMode:   auth.APIKey,
		Stream:     s.stream,
		HookSecret: suiteStreamSecret,
		Logger:     logger,
	})
	s.Require().NoError(err)
	other.Config.Handler = server.Handler()
	other.Start()
	s.T().Cleanup(other.Close)
	return other
}

// nodeDirectory is one node's end of the register of which node is running which session.
//
// The address is the listener's, which is known before anything is served on it, and the
// prefix is the suite's, because every suite in the run shares one Redis and a node of
// another suite is a node that is not there.
func (s *RouterSuite) nodeDirectory(listener *httptest.Server, logger *slog.Logger) *node.Directory {
	directory, err := node.NewDirectory(node.DirectoryOptions{
		Redis:   s.live.Redis(),
		Address: listener.Listener.Addr().String(),
		Prefix:  s.relayPrefix,
		Logger:  logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(directory.Close)
	return directory
}

// relayBus is one node's end of the suite's relay.
func (s *RouterSuite) relayBus(logger *slog.Logger) *relay.Bus {
	bus, err := relay.New(relay.Options{
		Redis: s.live.Redis(), Prefix: s.relayPrefix, Logger: logger,
	})
	s.Require().NoError(err)
	return bus
}

func (s *RouterSuite) policies(logger *slog.Logger) *policy.Enforcer {
	enforcer, err := policy.New(s.configs, logger)
	s.Require().NoError(err)
	return enforcer
}

// quota caps what one end user may spend in a day. The default is high enough that nothing
// a test does reaches it; a suite about the cap sets messagesPerDay before it starts the
// harness.
func (s *RouterSuite) quota(liveClient *live.Client, logger *slog.Logger) *quota.Limiter {
	allowed := s.messagesPerDay
	if allowed == 0 {
		allowed = 1_000_000
	}
	limiter, err := quota.New(liveClient.Redis(), quota.Limits{MessagesPerDay: allowed}, logger)
	s.Require().NoError(err)
	return limiter
}

// useFixture points the clients at the app of the fixture called name, with client signed
// in as the fixture's user.
func (s *RouterSuite) useFixture(name string) {
	loaded := s.requireFixture(name)
	s.useApp(loaded.testApp)
	s.client = s.data.signedInAs(loaded.userID)
}

// useApp points the clients at an app, each as a new caller of its kind.
func (s *RouterSuite) useApp(app testApp) {
	s.app = app
	s.unauthenticatedClient = &testClient{suite: s, header: http.Header{}, kind: noCredential}
	s.anonymousClient = s.data.createAnonymous()
	s.guestClient = s.data.createGuest()
	s.client = s.data.createUser()
	s.serverClient = s.signedIn(jwt.MapClaims{"server": true}, auth.AuthTypeServer, server, "")
}

// customerID is the tenant the suite's clients are, which is the app an API key belongs
// to. It is what the rows a test writes straight into Postgres have to be filed under to
// be the ones an endpoint reads back.
func (s *RouterSuite) customerID() string { return s.app.app.ID }

// signedFor is a token for an end user signed with secret, for a test presenting one that
// was signed with the wrong one.
func (s *RouterSuite) signedFor(secret string) string {
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id": s.utils.uuid(), "exp": time.Now().Add(time.Hour).Unix(),
	}).SignedString([]byte(secret))
	s.Require().NoError(err)
	return token
}

// signedIn is a client holding the app's key and a token signed with its secret.
func (s *RouterSuite) signedIn(claims jwt.MapClaims, authType string, kind callerKind, userID string) *testClient {
	claims["exp"] = time.Now().Add(time.Hour).Unix()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(s.app.secret))
	s.Require().NoError(err)

	header := http.Header{}
	header.Set(auth.APIKeyHeader, s.app.key)
	header.Set("Authorization", "Bearer "+token)
	header.Set(auth.AuthTypeHeader, authType)
	return &testClient{suite: s, header: header, kind: kind, userID: userID, token: token}
}

// testClient calls the router with one set of credentials. Its helpers expect the call to
// succeed; do is for asserting on a status.
type testClient struct {
	suite  *RouterSuite
	header http.Header
	kind   callerKind
	// userID is who the client acts for, empty for the backend and the unauthenticated.
	userID string
	// token is what signs for the client, which a socket sends in its query string.
	token string
	// address is the node the client sends to, empty for the suite's own.
	address string
}

// actingFor is the backend naming user as who it acts for, so what it opens is theirs.
func (c *testClient) actingFor(user *testClient) *testClient {
	named := *c
	named.header = c.header.Clone()
	named.header.Set(auth.UserHeader, user.userID)
	named.userID = user.userID
	return &named
}

// from is the same caller saying which client it is and who is at the keyboard, which is
// what the audit records a change as having been made with.
func (c *testClient) from(client, actorID, actorName string) *testClient {
	named := *c
	named.header = c.header.Clone()
	named.header.Set(clientHeader, client)
	named.header.Set(actorIDHeader, actorID)
	named.header.Set(actorNameHeader, actorName)
	return &named
}

// on is the same caller sending to another node of the same deployment, for a test about
// what a caller reaches when their request did not land where the session is.
func (c *testClient) on(other *httptest.Server) *testClient {
	elsewhere := *c
	elsewhere.header = c.header.Clone()
	elsewhere.address = other.URL
	return &elsewhere
}

// base is the node the client sends to.
func (c *testClient) base() string {
	if c.address != "" {
		return c.address
	}
	return c.suite.server.URL
}

// do sends body as JSON and decodes the answer into into, when there is one to decode.
func (c *testClient) do(method, path string, body, into any) int {
	status, payload := c.call(method, path, body)
	if into != nil && status < http.StatusBadRequest {
		c.suite.Require().NoError(json.Unmarshal(payload, into), string(payload))
	}
	return status
}

// call is do without decoding, for a test reading the error it was answered with.
// raw is a request's whole answer, headers and all, its body already read.
func (c *testClient) raw(method, path string, body any) *http.Response {
	require := c.suite.Require()
	payload := bytes.NewReader(nil)
	if body != nil {
		encoded, err := encode(body)
		require.NoError(err)
		payload = bytes.NewReader(encoded)
	}
	request, err := http.NewRequest(method, c.suite.server.URL+path, payload)
	require.NoError(err)
	request.Header = c.header.Clone()
	request.Header.Set("Content-Type", "application/json")
	response, err := c.suite.server.Client().Do(request)
	require.NoError(err)
	defer response.Body.Close()
	_, err = readAll(response)
	require.NoError(err)
	return response
}

func (c *testClient) call(method, path string, body any) (int, []byte) {
	require := c.suite.Require()
	payload := bytes.NewReader(nil)
	if body != nil {
		encoded, err := encode(body)
		require.NoError(err)
		payload = bytes.NewReader(encoded)
	}
	request, err := http.NewRequest(method, c.base()+path, payload)
	require.NoError(err)
	request.Header = c.header.Clone()
	request.Header.Set("Content-Type", "application/json")

	response, err := c.suite.server.Client().Do(request)
	require.NoError(err)
	defer response.Body.Close()
	answered, err := readAll(response)
	require.NoError(err)
	return response.StatusCode, answered
}

// failure is what the router said was wrong, for a test asserting on the message as well
// as the status.
func (c *testClient) failure(method, path string, body any) (int, string) {
	status, payload := c.call(method, path, body)
	var answered ErrorResponse
	if err := json.Unmarshal(payload, &answered); err != nil || answered.Error.Message == "" {
		return status, string(payload)
	}
	return status, answered.Error.Message
}

// watch opens a socket, with the client's credentials in the query string as well as in
// the headers: a browser cannot set headers on one and an SDK can, and a backend has to,
// since nothing in a query string may say a caller is one.
//
// It returns the status when the handshake is refused.
func (c *testClient) watch(path string) (*websocket.Conn, int) {
	address := "ws" + strings.TrimPrefix(c.base(), "http") + path
	query := url.Values{}
	if key := c.header.Get(auth.APIKeyHeader); key != "" {
		query.Set(auth.APIKeyParam, key)
		query.Set(auth.TokenParam, c.token)
	}
	if c.kind == anonymous && c.userID != "" {
		query.Set(auth.UserParam, c.userID)
	}
	if len(query) > 0 {
		if strings.Contains(address, "?") {
			address += "&" + query.Encode()
		} else {
			address += "?" + query.Encode()
		}
	}

	connection, response, err := websocket.DefaultDialer.Dial(address, c.header.Clone())
	if err != nil {
		c.suite.Require().NotNil(response, "dialling %s: %v", path, err)
		return nil, response.StatusCode
	}
	c.suite.T().Cleanup(func() { _ = connection.Close() })
	return connection, response.StatusCode
}

// opens a socket the handshake must accept.
func (c *testClient) opens(path string) *websocket.Conn {
	connection, status := c.watch(path)
	c.suite.Require().NotNil(connection, "the handshake for %s answered %d", path, status)
	return connection
}

// await reads frames off a socket until one of the wanted type arrives.
func (s *RouterSuite) await(connection *websocket.Conn, wanted string) map[string]any {
	s.Require().NoError(connection.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var received map[string]any
		if err := connection.ReadJSON(&received); err != nil {
			s.Require().FailNow("the socket closed before " + wanted + " arrived: " + err.Error())
		}
		if received["type"] == wanted {
			return received
		}
	}
}

// createSession opens a session the router must accept.
func (c *testClient) createSession(request CreateSessionRequest) Session {
	var created Session
	c.suite.Require().Equal(http.StatusCreated,
		c.do(http.MethodPost, "/v1/agents/sessions", request, &created))
	return created
}

// getSession reads one back.
func (c *testClient) getSession(id string) Session {
	var read Session
	c.suite.Require().Equal(http.StatusOK, c.do(http.MethodGet, "/v1/agents/sessions/"+id, nil, &read))
	return read
}

// updateSession renames or re-describes one.
func (c *testClient) updateSession(id string, request UpdateSessionRequest) Session {
	var updated Session
	c.suite.Require().Equal(http.StatusOK,
		c.do(http.MethodPatch, "/v1/agents/sessions/"+id, request, &updated))
	return updated
}

// stopSession ends one, and waits for it to be gone from the live set, because stopping is
// what frees its id for another session to take.
func (c *testClient) stopSession(id string) {
	c.suite.Require().Equal(http.StatusNoContent,
		c.do(http.MethodPost, "/v1/agents/sessions/"+id+"/stop", nil, nil))
}

// deleteSession deletes one, with its turns and what it remembered.
func (c *testClient) deleteSession(id string) {
	c.suite.Require().Equal(http.StatusNoContent,
		c.do(http.MethodDelete, "/v1/agents/sessions/"+id, nil, nil))
}

// querySessions lists what the caller may see.
func (c *testClient) querySessions(query SessionQuery) SessionPage {
	var page SessionPage
	c.suite.Require().Equal(http.StatusOK,
		c.do(http.MethodPost, "/v1/agents/sessions/query", query, &page))
	return page
}

// testUtils makes the values a test needs to be unique.
type testUtils struct{}

// uuid is a fresh UUIDv7, the kind the router gives its own rows.
func (testUtils) uuid() string {
	return uuid.Must(uuid.NewV7()).String()
}

// callID is a call nobody else is on.
func (u testUtils) callID() string {
	return "call-" + u.uuid()
}

// number is an E.164 number nobody else holds. It is random rather than read off the clock,
// which ticks in microseconds on some machines and so repeats every ten thousand numbers,
// and nothing a suite holds is ever released.
func (testUtils) number() string {
	return fmt.Sprintf("+1512%07d", rand.IntN(10_000_000))
}

// testData makes what a test runs against. Nothing it makes is cleaned up.
type testData struct {
	suite *RouterSuite
}

// createApp makes an organization and an app with a key, none of which any other test sees.
func (d testData) createApp() testApp {
	s, ctx := d.suite, context.Background()
	organization := store.Organization{Name: "organization-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateOrganization(ctx, &organization))
	app := store.App{OrganizationID: organization.ID, Name: "app-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateApp(ctx, &app))
	return d.keyed(organization, app)
}

// createAppAdmitting is an app of its own that takes only some levels of end user, for a
// test about a caller an app turns away.
func (d testData) createAppAdmitting(settings store.AppSettings) testApp {
	s, ctx := d.suite, context.Background()
	organization := store.Organization{Name: "organization-" + s.utils.uuid()}
	s.Require().NoError(s.store.CreateOrganization(ctx, &organization))
	app := store.App{OrganizationID: organization.ID, Name: "app-" + s.utils.uuid(), Settings: settings}
	s.Require().NoError(s.store.CreateApp(ctx, &app))
	return d.keyed(organization, app)
}

// keyed mints a credential for an app and stores it sealed, the way the router's own keys
// are held.
func (d testData) keyed(organization store.Organization, app store.App) testApp {
	s, ctx := d.suite, context.Background()
	key, secret, err := auth.NewCredential(auth.Test)
	s.Require().NoError(err)
	sealed, err := s.sealer.Seal(secret)
	s.Require().NoError(err)
	s.Require().NoError(s.store.CreateAPIKey(ctx, &store.APIKey{
		ID: key, AppID: app.ID, Name: "router suite", Env: string(auth.Test),
		Sealed: sealed, KEKVersion: auth.KEKVersion, Last4: auth.Last4(secret), CreatedBy: "router suite",
	}))
	return testApp{organization: organization, app: app, key: key, secret: secret}
}

// createAgentConfig is an agent of the suite's app, for the endpoints that need one to
// name. The models are the stub the suite routes to.
func (d testData) createAgentConfig() AgentConfig {
	var created AgentConfig
	d.suite.Require().Equal(http.StatusCreated, d.suite.serverClient.do(
		http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
			Name: "agent-" + d.suite.utils.uuid(),
			Llm:  pointerTo("llm-flow"), Instructions: pointerTo("be brief"),
		}, &created))
	return created
}

// createUser is a client signed in as a new end user of the suite's app.
func (d testData) createUser() *testClient {
	return d.signedInAs(d.suite.utils.uuid())
}

// signedInAs is a client signed in as the end user id.
func (d testData) signedInAs(id string) *testClient {
	return d.suite.signedIn(jwt.MapClaims{"user_id": id}, auth.AuthTypeJWT, user, id)
}

// createGuest is a client signed in as a new guest, a user Stream issued a temporary account.
func (d testData) createGuest() *testClient {
	id := d.suite.utils.uuid()
	return d.suite.signedIn(jwt.MapClaims{"user_id": id, "role": "guest"}, auth.AuthTypeJWT, guest, id)
}

// createAnonymous is a client going by a new name, with a token that proves nothing about it.
func (d testData) createAnonymous() *testClient {
	return d.claiming(d.suite.utils.uuid())
}

// claiming is an anonymous client going by name, whoever else goes by it.
func (d testData) claiming(name string) *testClient {
	client := d.suite.signedIn(jwt.MapClaims{}, auth.AuthTypeAnonymous, anonymous, name)
	client.header.Set(auth.UserHeader, name)
	client.userID = name
	return client
}

// backendOfAnotherApp is the server of an app this test has nothing to do with, for
// checking that one customer's rows do not reach another's.
func (d testData) backendOfAnotherApp() *testClient {
	s := d.suite
	mine := s.app
	s.app = d.createApp()
	stranger := s.signedIn(jwt.MapClaims{"server": true}, auth.AuthTypeServer, server, "")
	s.app = mine
	return stranger
}

// pointerTo is an optional field set to a value, which the generated requests take as a
// pointer because the absent key means something other than the zero one.
func pointerTo[T any](value T) *T { return &value }

// textSession asks for a conversation in writing, which needs no call.
func textSession(id *string) CreateSessionRequest {
	target, text := "en-low-latency", true
	return CreateSessionRequest{Id: id, Text: &text, Llm: &target}
}

// inProject lists the sessions in one project, which is how a test sharing a fixture's app
// finds its own.
func inProject(project string) SessionQuery {
	equals := Equals(project)
	return SessionQuery{Filter: &SessionFilter{ProjectID: &equals}}
}

// ids is the ids of a list of sessions, in order.
func ids(sessions []Session) []string {
	listed := make([]string, 0, len(sessions))
	for _, one := range sessions {
		listed = append(listed, one.Id)
	}
	return listed
}

// encode marshals a body, passing a string through as the JSON it already is so a test can
// send something the generated types cannot hold.
func encode(body any) ([]byte, error) {
	if raw, ok := body.(string); ok {
		return []byte(raw), nil
	}
	return json.Marshal(body)
}

func readAll(response *http.Response) ([]byte, error) {
	var buffer bytes.Buffer
	_, err := buffer.ReadFrom(response.Body)
	return buffer.Bytes(), err
}

// suiteApps is the Stream apps the suite's customers act in. A customer given none acts in
// the deployment's app, which is the suite's chattest.
type suiteApps struct {
	mu         sync.Mutex
	own        map[string]streamapp.Identity
	deployment *streamapp.Deployment
	// readOnly are customers whose work in the deployment's app may be read and not
	// added to, as app mode leaves it once the fallback is off.
	readOnly map[string]bool
	// waiting are customers whose app cannot be told until the deployment's own is known.
	waiting map[string]bool
	// nowhere are customers with no app to act in at all.
	nowhere map[string]bool
	// perApp answers as app mode does, where customers act in apps of their own.
	perApp bool
}

func (a *suiteApps) PerApp() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.perApp
}

func (a *suiteApps) For(ctx context.Context, customer string) (streamapp.Identity, error) {
	a.mu.Lock()
	identity, ok := a.own[customer]
	waiting, nowhere := a.waiting[customer], a.nowhere[customer]
	a.mu.Unlock()
	switch {
	case waiting:
		return streamapp.Identity{}, streamapp.ErrDeploymentAppUnknown
	case nowhere:
		return streamapp.Identity{}, streamapp.ErrNoIdentity
	case ok:
		return identity, nil
	}
	return a.deployment.For(ctx, customer)
}

func (a *suiteApps) ForApp(ctx context.Context, customer string, app int64) (streamapp.Identity, error) {
	a.mu.Lock()
	readOnly := a.readOnly[customer]
	a.mu.Unlock()
	if readOnly && (app == 0 || app == a.deployment.App()) {
		return streamapp.Identity{}, streamapp.ErrReadOnly
	}
	return a.ForAppReading(ctx, customer, app)
}

func (a *suiteApps) ForAppReading(ctx context.Context, customer string, app int64) (streamapp.Identity, error) {
	a.mu.Lock()
	identity, ok := a.own[customer]
	waiting := a.waiting[customer]
	a.mu.Unlock()
	switch {
	case waiting:
		return streamapp.Identity{}, streamapp.ErrDeploymentAppUnknown
	case ok && identity.StreamApp == app:
		return identity, nil
	}
	return a.deployment.ForApp(ctx, customer, app)
}

// set changes how the suite's apps answer for a customer, and forgets what they said.
func (s *RouterSuite) setApps(customer string, change func(*suiteApps)) {
	s.apps.mu.Lock()
	change(s.apps)
	s.apps.mu.Unlock()
	s.stream.Invalidate(customer)
}

// giveApp makes the customer act in an app of its own, a chattest of its own, from now on.
func (s *RouterSuite) giveApp(customer string, app int64, key string) *chattest.Server {
	own := chattest.NewServer(s.T())
	s.apps.mu.Lock()
	s.apps.own[customer] = streamapp.Identity{
		CustomerID: customer, StreamApp: app, APIKey: key,
		Secret: streamapp.NewSecret(key + "-secret"), BaseURL: own.URL,
		Registered: true, AllowGuests: true,
	}
	s.apps.mu.Unlock()
	s.stream.Invalidate(customer)
	return own
}
