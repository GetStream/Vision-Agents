//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/ed25519"
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
	"net/http/httputil"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/hibiken/asynq"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/blob"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detectionrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/ingest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/omnichannel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/siptrunk"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/vendors"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pluginevents"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// suiteConnections caps each suite's pool, so ten suites at once stay well inside the
// hundred connections Postgres allows.
const suiteConnections = 5

// settleFor is how long a test waits for something crossing a socket or a queue to arrive.
const settleFor = 5 * time.Second

// migrated runs the migrations once per database for every suite in the run: they set
// goose's globals and attach triggers, neither of which two suites may do at once.
var migrated = struct {
	sync.Mutex
	databases map[string]bool
}{databases: map[string]bool{}}

// migrate applies the migrations to the suite's database, unless a suite before it did.
func migrate(ctx context.Context, db *store.Store, dsn string) error {
	migrated.Lock()
	defer migrated.Unlock()
	if migrated.databases[dsn] {
		return nil
	}
	if err := db.Migrate(ctx); err != nil {
		return err
	}
	migrated.databases[dsn] = true
	return nil
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

// BeforeTest, in a suite with a database of its own, requires that no scheduled task is left
// and empties the table once the test is done, whoever's tasks they are, so none a test leaves
// due is claimed by the next test's worker. It runs after SetupTest, which may start tasks of its own.
func (s *RouterSuite) BeforeTest(_, _ string) {
	if s.database == "" || s.store == nil {
		return
	}
	t := s.T()
	count, err := s.store.DB().NewSelect().Model((*store.ScheduledTask)(nil)).Count(context.Background())
	require.NoError(t, err)
	require.Zero(t, count, "a task is left from an earlier test or run")
	t.Cleanup(func() {
		_, err := s.store.DB().NewDelete().Model((*store.ScheduledTask)(nil)).Where("true").Exec(context.Background())
		require.NoError(t, err)
	})
}

// suiteKEK seals the secrets of the keys the suite makes. It is the same for every suite,
// so a fixture one suite loaded opens in another.
const suiteKEK = "router suite"

// suiteTrunkKEK seals the trunk passwords of the numbers the phone service attaches.
const suiteTrunkKEK = "router suite sip trunks"

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

// suiteManageKey is what the management API, /v1/manage, is reached with.
const suiteManageKey = "suite-manage-key"

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
	// database and keyring, with the deployment's app as the fallback: behind a gateway,
	// whose mint endpoint is the suite's, that forwards each app's server token and vouches
	// for its hooks. A suite sets it, and the one after it, before SetupSuite runs.
	appMode bool
	// appRefuses turns app mode's fallback off, so a customer with no app of its own is
	// written nowhere.
	appRefuses bool
	// proxied authenticates as a router behind Stream's proxy does, believing the headers it
	// writes, so a test names its caller with proxiedClient rather than a key.
	proxied bool
	// shipping ships the conversation model's rows to a usage service of the suite's own,
	// usage, as cmd/router's writer does.
	shipping bool
	usage    *usagetest.Service
	// limited reads every organization's money from that usage service, started with its
	// management API before the policies, through usageFront, which a test can take down; and
	// lets the rows the service acks go from what the policies hold. It ships, too.
	limited    bool
	usageFront *usageFront
	enforcer   *policy.Enforcer
	// holds is the Redis the suite's policies share holds through, as the routers of a
	// deployment do; nil keeps them in the process.
	holds rueidis.Client
	// limitsMode is the suite's ROUTER_LIMITS_MODE, for a suite to set before it starts the
	// harness. Empty is enforce.
	limitsMode config.LimitsMode
	// owned are the apps the suite made or used, whose rows must all name a key.
	ownedMu sync.Mutex
	owned   []string
	// deploymentKnown has the suite's Stream say which app is the deployment's own, as a
	// deployment-mode router that was given its app id does.
	deploymentKnown bool
	// database gives the suite a database of its own, named after it, in place of the one
	// the suites share, for a suite whose tests act on rows other tests made: a
	// scheduled-task worker claims every task due in its database, whoever made it. A
	// suite sets it before SetupSuite runs.
	database string
	server   *httptest.Server
	// router is the suite's router itself, for what cmd/router calls on it besides serving.
	router *Server
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
	// shard and otherShard are the shards of the suite's node and of otherNode, for a suite
	// about shards to set before it starts the harness. Nodes in two shards have a relay
	// each. Empty is one deployment of one shard.
	shard, otherShard string

	// unauthenticatedClient sends no credentials. The rest hold the app's key:
	// anonymousClient goes by a name nothing proves, guestClient and client are signed-in
	// end users, a guest and a permanent one, and serverClient is the app's own backend.
	unauthenticatedClient *testClient
	anonymousClient       *testClient
	guestClient           *testClient
	client                *testClient
	serverClient          *testClient

	// The stubs standing in for providers, for a test to read back what the router asked
	// them for. model answers questions, vision is the one that can see, knowledge keeps the
	// passages written to it. holding answers once a test lets it, for a summary that has to
	// still be in the writing.
	model     *scriptedLLM
	vision    *scriptedLLM
	holding   *scriptedLLM
	knowledge *knowledgeBase
	memories  *keptMemories
	// voice keeps what every stub voice was told to say. Each socket is given a stub voice
	// and transcriber of its own, so one a test left open cannot read what the next test
	// sends; newestVoice and newestEars are the last the router started, read through
	// speaking and ears.
	voice       *saidLog
	speechMu    sync.Mutex
	newestVoice *recordingTTS
	newestEars  *quietSTT
	// unconfiguredBuilds counts what the "unconfigured" transcriber was asked to build.
	unconfiguredBuilds atomic.Int64
	// noted are the "noted" models opened so far, one per session, newest last, for a test
	// whose sessions each need a model of their own to read back.
	notedMu sync.Mutex
	noted   []*scriptedLLM
	// scheduled are the "scheduling" models opened so far, one per session, newest last.
	// They run the tool calls schedulingScript queues, so a test of the scheduling tools
	// scripts its own sessions' turns and reads back what each model was offered.
	scheduledMu      sync.Mutex
	scheduled        []*schedulingModel
	schedulingScript schedulingScript

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
	// limiter is the router's daily limit, for a suite to spend a user's day with.
	limiter *quota.Limiter

	// memoryStore is where sessions remember, for a suite about memory to set before it
	// starts the harness. Nil keeps them in memories.
	memoryStore memory.Backend
	// metering is what memory, knowledge and voice cloning record through, as the process
	// writer is in cmd/router.
	metering *routing.Recorder
	// memoryOff gives the router no memory store, as a deployment with no memory provider
	// has, for a control suite to set before it starts the harness.
	memoryOff bool

	// connectors is what connector adapters the router has, for a suite about connectors to
	// set before it starts the harness. Empty has none.
	connectors core.Registry
	// connectorsOff gives the router no connector keyring, transports or limiter, as a deployment with
	// ROUTER_CONNECTORS_ENABLED unset has, for a control suite to set before it starts the
	// harness. The suite's store, resolver and sealer are still built.
	connectorsOff bool
	// ttsVoices resolves the voice of every text-to-speech request, live and recorded, for a
	// suite about voices to set before it starts the harness. Nil passes each one through.
	ttsVoices routing.VoiceResolver
	// ttsConfig is what the text-to-speech routers, live and recorded, route over, for a
	// suite about voices to set before it starts the harness. Empty is routableConfig.
	ttsConfig routing.ModalityConfig
	// proxyCallsPerMinute caps one customer's direct calls to one connector, for a suite about
	// the connection proxy to set before it starts the harness. Zero caps nothing.
	proxyCallsPerMinute int64
	// runsPerApp caps one app's runs in flight, for a suite about runs to set before it starts
	// the harness. Zero caps nothing.
	runsPerApp int64
	// knowledgeLimits is how documents are cut, for a suite about knowledge limits to set
	// before it starts the harness. Zero is ingest.DefaultLimits.
	knowledgeLimits ingest.Limits
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
	// logs, set by a suite before it starts the harness, is where the router logs, as text at
	// debug and up, for a suite about what it warns of. Nil discards them.
	logs io.Writer
	// telnyxPublicKey, set by a suite about Telnyx's call webhook before it starts the
	// harness, is the key Telnyx's requests are checked against. Nil leaves it unconfigured.
	telnyxPublicKey ed25519.PublicKey
	// telnyxConnectionID is the TeXML application the router's Telnyx numbers are on, for the
	// same suite to set. Empty is a router with none, which answers no Telnyx call.
	telnyxConnectionID string
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
	// approvalWait is how long a connector call held for approval waits, for a suite to set
	// before it starts the harness. Zero is the router's ten minutes.
	approvalWait time.Duration
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
	// imageHTTP is what runs fetch their images with, for a suite whose images are served on
	// loopback. Nil reaches only public hosts.
	imageHTTP *http.Client
	// events subscribes configs to their plugins' events.
	events *pluginevents.Service

	// channelAPI is where every channel provider is reached, for a suite about channels to
	// set before it starts the harness. Nil reaches the real WhatsApp, Telnyx and Linq.
	channelAPI *httptest.Server
	// inbound answers what arrives on the app's channels.
	inbound *channels.Service

	// daytona is where the sessions' sandboxes are made, for a suite whose agents run code to
	// set before it starts the harness. Nil makes them on the Daytona the environment names.
	daytona *daytonatest.Server
	// sandbox is what an app with no approved use case may do, for a suite about the
	// sandbox to set before it starts the harness. The zero value is off.
	sandbox dlc.Sandbox
	// registrar is who registers 10DLC campaigns, for a suite about registration to set
	// before it starts the harness. Nil makes Stream's approval final.
	registrar dlc.Registrar
	// gate is what every text and call passes.
	gate *dlc.Gate
	// phoneRecorder is what texts, call minutes, numbers and 10DLC fees record through, priced
	// from the vendor list as a router's first start prices them.
	phoneRecorder *routing.Recorder
	// phoneNow, set by a suite before it starts the harness, is the clock the phone service
	// times call legs and months by. Nil is the wall clock.
	phoneNow func() time.Time
	// registrar10DLC is the router's 10DLC service, for a suite to poll and bill with.
	registrar10DLC *dlc.Service
	// manageOff builds the router with no manage key, as a deployment that never set
	// ROUTER_AUTH_MANAGE_KEY has, for a suite about that to set before it starts the harness.
	manageOff bool
	// pricedFromDatabase seeds the price list from the embedded vendor list, as a router's
	// first start does, and prices the LLM router's rows and the API from it, read every
	// priceReload, for a suite about the price list to set before it starts the harness.
	// prices is that list.
	pricedFromDatabase bool
	prices             *pricing.Active

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
	s.forgetPhoneVendors()
	ctx := context.Background()
	logger := slog.New(slog.DiscardHandler)
	if s.logs != nil {
		logger = slog.New(slog.NewTextHandler(s.logs, &slog.HandlerOptions{Level: slog.LevelDebug}))
	}

	// A shard database and Redis run the suite as a hosted shard runs: the shared rows in one
	// place and the calls, sessions and requests in another.
	options := store.Options{DSN: testDatabase(s.T(), dsn)}
	if s.database != "" {
		options.DSN = testenv.Database(options.DSN, s.database)
	}
	if shardDSN := os.Getenv("ROUTER_POSTGRES_SHARD_DSN"); shardDSN != "" {
		options.ShardDSN = testDatabase(s.T(), shardDSN)
	}
	pgStore, err := store.OpenShards(options)
	s.Require().NoError(err)
	pgStore.DB().SetMaxOpenConns(suiteConnections)
	s.Require().NoError(migrate(ctx, pgStore, options.DSN))
	s.store = pgStore
	s.T().Cleanup(func() { s.Require().NoError(pgStore.Close()) })
	// Registered before everything that records, so it runs once all of it has closed and
	// written what it held.
	s.T().Cleanup(s.assertEveryRowNamesAKey)
	if s.database != "" {
		// Only a suite's own database is emptied, never the shared one. A run killed by
		// -timeout skips its cleanups, and the database outlives it.
		_, err := pgStore.DB().NewDelete().Model((*store.ScheduledTask)(nil)).Where("true").Exec(ctx)
		s.Require().NoError(err)
	}

	liveClient, err := live.New(live.Options{Address: redisAddr, Shard: live.Shard{Address: os.Getenv("ROUTER_REDIS_SHARD_ADDR")}})
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
	if s.deploymentKnown {
		s.stream = streamapp.NewClients(knownApps{s.apps}, streamapp.ClientsOptions{})
	}
	if s.appMode {
		mint, err := streamapp.NewMint(s.gatewayMint().URL, suiteMintBearer)
		s.Require().NoError(err)
		forwarded, err := streamapp.NewForwarded(streamapp.ForwardedOptions{
			Store: pgStore, Sealer: s.sealer, Mint: mint, BaseURL: s.chat.URL, Logger: logger,
		})
		s.Require().NoError(err)
		stored, err := streamapp.NewStored(streamapp.StoredOptions{
			Store: pgStore, Deployment: s.apps.deployment, FallbackToDeployment: !s.appRefuses, Forwarded: forwarded, Logger: logger,
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

	if s.pricedFromDatabase {
		s.prices = s.databasePrices(logger)
	}
	s.metering = s.meteringRecorder(logger)
	policies := s.policies(logger)
	s.limiter = s.quota(liveClient, policies, logger)
	streams := s.routers(s.limiter, policies, logger)
	// Built before it is served on, because a node says where it is before it has
	// anything to say it is running.
	listener := httptest.NewUnstartedServer(nil)
	directory := s.nodeDirectory(listener, s.shard, logger)
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
		Consents: ConnectorConsents(pgStore, s.connectors, s.sealer, s.publicURL), ApprovalWait: s.approvalWait}, policies, logger)
	// A nil client reaches public hosts alone, and every auth server here is a local one.
	public := &plugins.Auth{HTTP: http.DefaultClient}
	s.events = s.pluginEvents(sessions, public, logger)
	// Every text, call and number asks the same admission as a model, as in cmd/router.
	s.gate = dlc.NewGate(pgStore, liveClient.Redis(), s.sandbox, logger).Charging(policies).Limiting(s.limitsMode)
	s.phoneRecorder = s.telephonyRecorder(logger)
	s.inbound = s.channels(sessions, logger)
	s.dispatch = dispatch.NewPool()
	s.streams = streams
	s.modalities = map[routing.Modality]routing.Inspector{
		routing.LLM:           streams.LLM,
		routing.STT:           streams.STT,
		routing.TTS:           streams.TTS,
		routing.STS:           streams.STS,
		routing.Search:        streams.Search,
		routing.DecisionModel: streams.DecisionModel,
		routing.Image:         streams.Image,
		routing.Detection:     streams.Detection,
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
	manageKey := suiteManageKey
	if s.manageOff {
		manageKey = ""
	}
	server, err := NewServer(Options{
		Routers:   s.modalities,
		Prices:    s.prices,
		Streams:   streams,
		Relay:     s.relayBus(s.shard, logger),
		Directory: directory,
		Sessions:  sessions,
		Store:     pgStore,
		Configs:   s.configs,
		Live:      liveClient,
		Auth:      s.authenticator(),
		AuthMode:  s.authMode(),
		// App mode is served only behind a gateway that declares kinds.
		ProxyDeclaresKind: s.appMode,
		Phone:             s.telephony(logger),
		Campaigns:         s.campaigns(sessions, logger),
		Simulations:       s.simulations(sessions, streams, logger),
		PluginEvents:      s.events,
		Channels:          s.inbound,
		DLC:               s.registrations(listener, logger),
		Gate:              s.gate,
		OpsKey:            suiteOpsKey,
		ManageKey:         manageKey,
		Secrets:           s.sealer,
		Knowledge:         s.knowledgeWriter(),
		KnowledgeLimits:   s.knowledgeLimits,
		KnowledgeURLs:     s.pages(redisAddr),
		Voices:            s.voiceService(),
		VoiceLibrary:      voices.NewCatalogue(),
		Dispatch:          s.dispatch,
		Policies:          policies,
		Connectors:        s.connectors,
		// Connector consents and credentials seal under the suite's key.
		ConnectorSecrets:   connectorSecrets,
		PublicURL:          s.publicURL,
		DashboardURL:       s.dashboardURL,
		Quota:              s.limiter,
		DataRetention:      time.Hour,
		PluginHTTP:         s.mcpTransport(),
		Stream:             s.stream,
		HookSecret:         suiteStreamSecret,
		TelnyxPublicKey:    s.telnyxPublicKey,
		TelnyxConnectionID: s.telnyxConnectionID,
		Logger:             logger,
		// The connector events endpoint revokes through the suite's resolver.
		ConnectorResolver:     s.resolver,
		ConnectorTransports:   serverTransports,
		ConnectorLimiter:      connectorLimiter,
		ProxyCallsPerMinute:   s.proxyCallsPerMinute,
		RunsPerApp:            s.runsPerApp,
		ImageHTTP:             s.imageHTTP,
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
	s.router = server
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
		Recorder:  s.phoneRecorder,
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
		Store: s.store, Registrar: s.registrar, PublicURL: "http://" + listener.Listener.Addr().String(),
		Recorder: s.phoneRecorder, Logger: logger,
	})
	s.Require().NoError(err)
	s.registrar10DLC = service
	return service
}

// telephonyRecorder records telephony through a writer of its own, priced at the vendor list's
// rates, and writes what it holds when the suite ends.
func (s *RouterSuite) telephonyRecorder(logger *slog.Logger) *routing.Recorder {
	list, err := pricing.NewList("vendor", pricing.AllAtCost(prices.Vendor().Entries), logger)
	s.Require().NoError(err)
	recorder := routing.NewRecorder(routing.Phone, s.store, s.live, list, logger)
	s.T().Cleanup(recorder.Close)
	return recorder
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

// speaking is the stub voice the router started last, the one a test's newest socket speaks
// in.
func (s *RouterSuite) speaking() *recordingTTS {
	s.speechMu.Lock()
	defer s.speechMu.Unlock()
	return s.newestVoice
}

// ears is the stub transcriber the router started last, the one a test's newest socket
// hears through.
func (s *RouterSuite) ears() *quietSTT {
	s.speechMu.Lock()
	defer s.speechMu.Unlock()
	return s.newestEars
}

func (s *RouterSuite) spoke(voice *recordingTTS) {
	s.speechMu.Lock()
	defer s.speechMu.Unlock()
	s.newestVoice = voice
}

func (s *RouterSuite) heard(ears *quietSTT) {
	s.speechMu.Lock()
	defer s.speechMu.Unlock()
	s.newestEars = ears
}

// routers builds every modality against stubs that answer in process. What a real vendor
// makes of real audio is that provider package's own suite; what is under test here is the
// HTTP surface in front of it.
func (s *RouterSuite) routers(limiter *quota.Limiter, gate routing.Gate, logger *slog.Logger) *Streams {
	hearing := sttrouter.NewRegistry()
	hearing.Register("stub", func(routing.Spec) (stt.STT, error) {
		return &quietSTT{emitter: stt.NewEmitter(64), started: s.heard}, nil
	})
	hearing.Register("refusing", func(routing.Spec) (stt.STT, error) {
		return &refusingSTT{emitter: stt.NewEmitter(64)}, nil
	})
	hearing.Register("echoing", func(routing.Spec) (stt.STT, error) {
		return &echoingSTT{emitter: stt.NewEmitter(64)}, nil
	})
	hearing.Register("batch", func(routing.Spec) (stt.STT, error) { return batchOnlySTT{}, nil })
	// A live model whose vendor is down: its socket fails to open the way batchOnlySTT's does.
	hearing.Register("down", func(routing.Spec) (stt.STT, error) { return batchOnlySTT{}, nil })
	hearing.Register("unconfigured", func(routing.Spec) (stt.STT, error) {
		s.unconfiguredBuilds.Add(1)
		return nil, errUnconfigured
	})
	// Reached by name only: no shortcut asks for Latin.
	heard := routableConfig()
	// A title is what makes a shortcut a listed route (speech_recorders_test.go reads the list).
	heard.Aliases["en-low-latency"] = routing.Alias{Title: "Low latency", Languages: []string{"en"}, RequireRealtime: true}
	heard.Providers = append(heard.Providers, routing.ProviderConfig{
		Provider: "refusing", Model: refusingModel, Languages: []string{"la"}, Realtime: true,
	}, routing.ProviderConfig{Provider: "refusing", Model: handingOverModel, Languages: []string{"la"}, Realtime: true},
		routing.ProviderConfig{Provider: "echoing", Model: "echoing-model", Languages: []string{"la"}, Realtime: true},
		routing.ProviderConfig{Provider: "batch", Model: batchModel, Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "down", Model: downModel, Languages: []string{"la"}, Realtime: true},
		routing.ProviderConfig{Provider: "unconfigured", Model: unconfiguredModel, Languages: []string{"la"}, Realtime: true})
	// As the deployment's own registries do: what cannot be built is listed unavailable.
	hearing.Probe()
	transcriber, err := sttrouter.New(sttrouter.Options{
		Config: heard, Registry: hearing, Store: s.store, Live: s.live, Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)

	// A session opens a model for the conversation and another for its flow controller,
	// and two sessions sharing one stub would each consume the other's turns. The first
	// open is the one a test reads back.
	s.model = &scriptedLLM{reply: "Hello."}
	var opened atomic.Int64
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		if opened.Add(1) == 1 {
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
	reasoning.Register("analysing", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Noted.", analyses: true}, nil
	})
	// A guardrail judge that finds every turn against its policy.
	reasoning.Register("judging", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: `{"violates": 1, "reason": "it is against the policy"}`}, nil
	})
	reasoning.Register("counted", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Counted.", usage: llm.Usage{InputTokens: 1000, InputTokensDetails: llm.InputTokensDetails{CachedTokens: 400}, OutputTokens: 20}}, nil
	})

	// A model that loads the skill called policy on its first turn and answers once it is
	// back, the way a moderator's run reads a skill before it judges.
	reasoning.Register("checking", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Checking.", then: "Judged.", calls: []llm.ToolCall{{
			ID: store.NewID(), Name: harness.LoadSkillTool, Arguments: `{"name":"policy"}`,
		}}}, nil
	})

	reasoning.Register("failing", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{refuses: errors.New("failing: the provider is down")}, nil
	})

	// A model that is a while in the writing, for a command that has to still be running
	// when the test asks it to stop.
	reasoning.Register("slow", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Here it is.", takes: time.Second}, nil
	})
	// A model that outlasts the moment a closing session gives its reply to finish, for a
	// reply that is still being written when the router stops.
	reasoning.Register("stuck", func(routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: "Too late.", takes: time.Minute}, nil
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
	// A model that runs the tool calls its test queued in the suite's schedulingScript,
	// one call per person who speaks, for the scheduling tools' tests.
	reasoning.Register("scheduling", func(routing.Spec) (llmrouter.Provider, error) {
		model := &schedulingModel{script: &s.schedulingScript, failing: s.schedulingScript.failingTurns()}
		s.scheduledMu.Lock()
		s.scheduled = append(s.scheduled, model)
		s.scheduledMu.Unlock()
		return model, nil
	})
	// A model that runs crm's echo through a waiting binding's call_tool whenever somebody
	// asks it something, a follow-up after a login included (chat_logins_test.go).
	reasoning.Register("logging-in", func(routing.Spec) (llmrouter.Provider, error) { return &loggingInLLM{}, nil })
	// A customer's own model, answering with where it was told to go, so a test reads back
	// that the stored endpoint and key reached the provider.
	reasoning.Register(routing.CustomProvider, func(spec routing.Spec) (llmrouter.Provider, error) {
		return &scriptedLLM{reply: spec.Endpoint.BaseURL + " " + spec.Endpoint.Model + " " + spec.Endpoint.APIKey}, nil
	})
	// Nil unless the suite prices from the database: a nil *pricing.Active is no nil Pricer.
	var pricer pricing.Pricer
	if s.prices != nil {
		pricer = s.prices
	}
	reasoner, err := llmrouter.New(llmrouter.Options{
		Config: reasoningConfig(), Registry: reasoning, Store: s.store, Live: s.live,
		Quota: limiter, Gate: gate, Pricer: pricer, Logger: logger,
		Models: llmrouter.NewCustomModels(s.configs, s.sealer, false),
		Writer: s.shippingWriter(gate, logger),
	})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)

	s.voice = &saidLog{}
	speaking := ttsrouter.NewRegistry()
	speaking.Register("stub", func(spec routing.Spec) (tts.TTS, error) {
		return &recordingTTS{emitter: tts.NewEmitter(64), said: s.voice, voice: spec.Voice, started: s.spoke}, nil
	})
	speaking.Register("plain", func(routing.Spec) (tts.TTS, error) {
		return &recordingTTS{emitter: tts.NewEmitter(64), said: s.voice, started: s.spoke}, nil
	})
	ttsConfig := s.ttsConfig
	if len(ttsConfig.Providers) == 0 {
		ttsConfig = routableConfig()
	}
	speaker, err := ttsrouter.New(ttsrouter.Options{
		Config: ttsConfig, Registry: speaking, Store: s.store, Live: s.live, Voices: s.ttsVoices, Gate: gate,
		Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	transcriptions, err := sttrouter.NewRecordings(sttrouter.Options{
		Config:       recordingConfig(),
		Transcribers: transcriberRegistry(),
		Store:        s.store,
		Live:         s.live,
		Gate:         gate,
		Logger:       logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(transcriptions.Close)

	recordings, err := ttsrouter.NewRecordings(ttsrouter.Options{
		Config:    ttsConfig,
		Recorders: recorderRegistry(),
		Store:     s.store,
		Live:      s.live,
		Voices:    s.ttsVoices,
		Gate:      gate,
		Logger:    logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(recordings.Close)

	finding, err := searchrouter.New(searchrouter.Options{
		Config: searchConfig(), Registry: searchRegistry(), Store: s.store, Live: s.live, Gate: gate,
		Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(finding.Close)

	judging, err := decisionrouter.New(decisionrouter.Options{
		Config: classifyConfig(), Registry: classifierRegistry(), Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(judging.Close)

	imaging, err := imagerouter.New(imagerouter.Options{
		Config: imageConfig(), Registry: painterRegistry(), Store: s.store, Live: s.live, Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(imaging.Close)

	detecting, err := detectionrouter.New(detectionrouter.Options{
		Config: detectionConfig(), Registry: detectorRegistry(), Gate: gate, Logger: logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(detecting.Close)

	conversing, err := stsrouterStub()
	s.Require().NoError(err)
	s.T().Cleanup(conversing.Close)

	return &Streams{
		STT: transcriber, TTS: speaker, LLM: reasoner, STS: conversing,
		Search: finding, DecisionModel: judging, Image: imaging, Detection: detecting,
		Transcriptions: transcriptions, Speech: recordings,
	}
}

// suitePriceReload is how often a suite pricing from the database reads its list again: a
// tenth of a second rather than a router's 30, so a change made in a test prices within it.
const suitePriceReload = 100 * time.Millisecond

// databasePrices seeds the suite's price list from the embedded vendor list, as a router's
// first start does (once: every later start finds it seeded and adds nothing it already has),
// and reads it every suitePriceReload until the suite ends.
func (s *RouterSuite) databasePrices(logger *slog.Logger) *pricing.Active {
	ctx, cancel := context.WithCancel(context.Background())
	s.T().Cleanup(cancel)
	_, _, _, err := s.store.SyncPrices(ctx, store.PriceLoad{Entries: prices.Vendor().Entries, Source: prices.VendorFile, Release: "suite"})
	s.Require().NoError(err)
	fallback, err := prices.Embedded(routing.LocalItems(routing.Config{routing.LLM: reasoningConfig()}), "suite", logger)
	s.Require().NoError(err)
	return pricing.NewActive(ctx, pricing.ActiveOptions{
		Source: s.store.PriceSource(), Fallback: fallback, Interval: suitePriceReload, RequireEntries: true, Logger: logger,
	})
}

// sessionManager runs conversations, writing them down in Stream Chat through a stand-in
// for the chat backend: what a session records is under test, what Stream does with it is
// not.
func (s *RouterSuite) sessionManager(
	streams *Streams,
	directory *node.Directory,
	connectors session.Connectors,
	policies *policy.Enforcer,
	logger *slog.Logger,
) *session.Manager {
	conversations := conversation.NewForChats(conversation.StreamApps(s.stream))
	s.T().Cleanup(conversations.Close)
	s.conversations = conversations

	s.memories = &keptMemories{}
	var remembering memory.Backend = s.memories
	if s.memoryStore != nil {
		remembering = s.memoryStore
	}
	var metering *metered.Memory
	if !s.memoryOff {
		metering = metered.NewMemory(remembering, s.metering)
	}
	var sandboxes *metered.Sandboxes
	if s.daytona != nil {
		sandboxes = metered.NewSandboxes(metered.SandboxesOptions{APIKey: "daytona-key", APIURL: s.daytona.APIURL(), Logger: logger})
	}
	sessions, err := session.NewManager(session.ManagerOptions{
		Sandboxes:     sandboxes,
		LLM:           streams.LLM,
		STT:           streams.STT,
		TTS:           streams.TTS,
		Memory:        metering,
		Transcript:    s.transcripts,
		Conversations: conversations,
		Stream:        s.stream,
		Store:         s.store,
		Policies:      policies,
		Live:          s.live,
		Configs:       s.configs,
		Directory:     directory,
		Connectors:    connectors,
		Logger:        logger,
		// The policies a session's resume_history falls back to, as cmd/router passes them.
		ResumeHistories: policies,
		Edge: func(context.Context, session.Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = sessions.Shutdown() })
	s.manager = sessions
	return sessions
}

// own notes an app the suite made or used.
func (s *RouterSuite) own(appID string) {
	s.ownedMu.Lock()
	defer s.ownedMu.Unlock()
	if !slices.Contains(s.owned, appID) {
		s.owned = append(s.owned, appID)
	}
}

// assertEveryRowNamesAKey fails the suite when anything it drove recorded usage with neither a
// key nor an origin: the owner site that wrote it was missed.
func (s *RouterSuite) assertEveryRowNamesAKey() {
	s.ownedMu.Lock()
	owned := slices.Clone(s.owned)
	s.ownedMu.Unlock()
	if len(owned) == 0 {
		return
	}
	var rows []store.Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&rows).
		Where("customer_id IN (?)", bun.In(owned)).Where("api_key = ?", routing.Unattributed).
		Scan(context.Background()))
	for _, row := range rows {
		s.Failf("a usage row names no key", "customer %s, %s %s/%s, purpose %q, session %q, agent %q",
			row.CustomerID, row.Modality, row.Provider, row.Model, row.Purpose, row.SessionID, row.AgentID)
	}
}

// shippingWriter is the writer the conversation model records through when the suite ships,
// and nil, a writer of the router's own, when it does not.
func (s *RouterSuite) shippingWriter(gate routing.Gate, logger *slog.Logger) *routing.Writer {
	if !s.shipping {
		return nil
	}
	if s.usage == nil {
		s.usage = usagetest.Start(s.T(), usagetest.Options{Logger: logger})
	}
	var shipped func([]routing.Acked)
	if s.limited {
		shipped = s.enforcer.Acked
	}
	writer := routing.NewWriter(routing.WriterOptions{
		Store: s.store, Live: s.live, Logger: logger, Gate: gate, Usage: s.usage.Client, Source: "router/suite/" + s.utils.uuid(),
		Shipped: shipped,
	})
	s.T().Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		writer.Close(ctx)
	})
	return writer
}

// authenticator resolves a key the way a deployment in api_key mode does: the row is read
// from Postgres and its secret unsealed, so the credentials a test holds are real ones.
func (s *RouterSuite) authenticator() auth.Authenticator {
	if s.proxied {
		return auth.NewProxy(auth.ProxyOptions{DeclaresKind: true})
	}
	authenticator, err := auth.New(auth.APIKey, s.configs.Lookup(s.sealer))
	s.Require().NoError(err)
	return authenticator
}

// forgetPhoneVendors unsets every credential a phone vendor is opened with, so no suite
// searches or buys at a real vendor whatever keys testenv loaded from the repo's .env. The
// suites share one process, and a parallel suite cannot use t.Setenv, so they go for all.
func (s *RouterSuite) forgetPhoneVendors() {
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)
	for _, vendor := range config.Vendors {
		for _, name := range vendor.Credentials {
			s.Require().NoError(os.Unsetenv(name))
		}
	}
}

// telephony serves the phone paths against the real vendor registry with no credentials
// behind it, which is what a deployment that has bought no numbers yet looks like.
func (s *RouterSuite) telephony(logger *slog.Logger) *phone.Service {
	sealer, err := auth.NewSealer(suiteTrunkKEK)
	s.Require().NoError(err)
	return s.telephonySealedBy(sealer, logger)
}

// telephonySealedBy is telephony with trunk passwords sealed under sealer; nil is a
// deployment with no keyring.
func (s *RouterSuite) telephonySealedBy(sealer *auth.Sealer, logger *slog.Logger) *phone.Service {
	// Every vendor the router knows is declared, and none of them has credentials: what a
	// vendor does with a real number is its own package's suite, and what is under test
	// here is the HTTP surface in front of them.
	config, err := phone.DefaultConfig()
	s.Require().NoError(err)

	service, err := phone.NewService(phone.ServiceOptions{
		Registry:  vendors.Registry(config),
		Store:     s.store,
		Recorder:  s.phoneRecorder,
		Now:       s.phoneNow,
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

// meteringRecorder is the recorder the suite's memory, knowledge and voice cloning record
// through, priced as the suite's routers are, and drained when the suite ends.
func (s *RouterSuite) meteringRecorder(logger *slog.Logger) *routing.Recorder {
	var pricer pricing.Pricer
	if s.prices != nil {
		pricer = s.prices
	}
	recorder := routing.NewRecorder(routing.Knowledge, s.store, s.live, pricer, logger)
	s.T().Cleanup(recorder.Close)
	return recorder
}

func (s *RouterSuite) knowledgeWriter() *metered.Knowledge {
	s.knowledge = newKnowledgeBase()
	return metered.NewKnowledge(s.knowledge, s.metering)
}

// pages keeps the knowledge bases filled from pages published elsewhere. Its queue is in a
// Redis database of its own, so neither another suite nor a router running against the
// same Redis takes the jobs for itself.
func (s *RouterSuite) pages(redisAddr string) *urls.Service {
	// Sixteen Redis databases do not go round every suite running at once, so the queue is
	// also named for the suite: one suite's worker taking another's read loses it.
	n := redisDB.Add(1)
	service, err := urls.New(urls.Options{
		Store:         s.store,
		Redis:         asynq.RedisClientOpt{Addr: redisAddr, DB: int(n)%15 + 1},
		Queue:         fmt.Sprintf("knowledge-%d", n),
		Reader:        pageReader{},
		Knowledge:     metered.NewKnowledge(newKnowledgeBase(), s.metering),
		Limits:        s.knowledgeLimits,
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

	service, err := voices.NewService(voices.Options{Store: s.configs, Bucket: bucket, Cloners: cloners, Recorder: s.metering})
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
	return s.otherNodeDispatching(nil)
}

// otherNodeDispatching is otherNode with dispatch workers of its own, waiting in workers.
func (s *RouterSuite) otherNodeDispatching(workers *dispatch.Pool) *httptest.Server {
	other, _ := s.otherRouter(workers)
	return other
}

// otherRouter is otherNodeDispatching with the router behind it, for a test that stops it.
func (s *RouterSuite) otherRouter(workers *dispatch.Pool) (*httptest.Server, *Server) {
	return s.otherRouterSealedBy(workers, s.sealer, slog.New(slog.DiscardHandler))
}

// otherRouterSealedBy is otherRouter holding keyring rather than the suite's, which is
// what its peers' credentials are checked against, and logging to logger. Nil is a node
// with no keyring.
func (s *RouterSuite) otherRouterSealedBy(workers *dispatch.Pool, keyring *auth.Sealer, logger *slog.Logger) (*httptest.Server, *Server) {
	other, server, _ := s.otherNodeHolding(workers, keyring, logger)
	return other, server
}

// otherNodeHolding is otherRouterSealedBy with the sessions it runs, for a test that stops
// them the way a router going down does.
func (s *RouterSuite) otherNodeHolding(workers *dispatch.Pool, keyring *auth.Sealer, logger *slog.Logger) (*httptest.Server, *Server, *session.Manager) {
	other := httptest.NewUnstartedServer(nil)
	directory := s.nodeDirectory(other, s.otherShard, logger)
	sessions, err := session.NewManager(session.ManagerOptions{
		LLM:           s.streams.LLM,
		STT:           s.streams.STT,
		TTS:           s.streams.TTS,
		Memory:        metered.NewMemory(&keptMemories{}, s.metering),
		Conversations: s.conversations,
		Stream:        s.stream,
		Store:         s.store,
		Live:          s.live,
		Configs:       s.configs,
		Directory:     directory,
		Logger:        logger,
		Edge: func(context.Context, session.Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return &silentEdge{inbound: make(chan agent.InboundAudio, 4)}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = sessions.Shutdown() })

	bus := s.relayBus(s.otherShard, logger)
	var everyWorker *node.Workers
	if workers != nil {
		everyWorker = s.deploymentWorkers(workers, directory, s.workBus(logger))
	}
	server, err := NewServer(Options{
		Routers:           s.modalities,
		Streams:           s.streams,
		Sessions:          sessions,
		Relay:             bus,
		Directory:         directory,
		Dispatch:          workers,
		Workers:           everyWorker,
		Store:             s.store,
		Configs:           s.configs,
		Live:              s.live,
		Auth:              s.authenticator(),
		AuthMode:          s.authMode(),
		ProxyDeclaresKind: s.appMode,
		Stream:            s.stream,
		HookSecret:        suiteStreamSecret,
		Secrets:           keyring,
		// The node's phone service is the suite's: the call hook a peer hands over answers
		// the same database.
		Phone:  s.router.phone,
		Logger: logger,
	})
	s.Require().NoError(err)
	other.Config.Handler = server.Handler()
	other.Start()
	s.T().Cleanup(other.Close)
	return other, server, sessions
}

// deploymentWorkers is one node's view of the dispatch workers across the deployment.
func (s *RouterSuite) deploymentWorkers(workers *dispatch.Pool, directory *node.Directory, bus *relay.Bus) *node.Workers {
	ctx, cancel := context.WithCancel(context.Background())
	s.T().Cleanup(cancel)
	everyWorker, err := node.NewWorkers(ctx, workers, directory, bus, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	return everyWorker
}

// nodeDirectory is one node's end of the register of which node is running which session.
//
// The address is the listener's, which is known before anything is served on it, and the
// prefix is the suite's, because every suite in the run shares one Redis and a node of
// another suite is a node that is not there.
func (s *RouterSuite) nodeDirectory(listener *httptest.Server, shard string, logger *slog.Logger) *node.Directory {
	directory, err := node.NewDirectory(node.DirectoryOptions{
		Redis:   s.live.Redis(),
		Address: listener.Listener.Addr().String(),
		Prefix:  s.relayPrefix,
		Shard:   shard,
		Logger:  logger,
	})
	s.Require().NoError(err)
	s.T().Cleanup(directory.Close)
	return directory
}

// relayBus is one node's end of its shard's relay.
func (s *RouterSuite) relayBus(shard string, logger *slog.Logger) *relay.Bus {
	bus, err := relay.New(relay.Options{
		Redis: s.live.ShardRedis(), Prefix: s.relayPrefix + shard, Logger: logger,
	})
	s.Require().NoError(err)
	return bus
}

// workBus is one node's end of the bus its workers take work on, which every shard shares.
func (s *RouterSuite) workBus(logger *slog.Logger) *relay.Bus {
	bus, err := relay.New(relay.Options{
		Redis: s.live.Redis(), Prefix: s.relayPrefix, Logger: logger,
	})
	s.Require().NoError(err)
	return bus
}

// suiteFresh is how often a suite's policies read money and their decisions again: a tenth of
// a second rather than a router's five, so a limit a test spends is measured within it.
const suiteFresh = 100 * time.Millisecond

// suiteFlushMaxLag is how long a suite sharing holds lets one wait unshared before billable
// work is refused, rather than a router's ten seconds.
const suiteFlushMaxLag = 300 * time.Millisecond

// limitApp gives the suite's app a spend limit of its own, as the limits API writes one.
func (s *RouterSuite) limitApp(interval usage.Interval, micros int64) store.Limit {
	limit := store.Limit{
		Subject: usage.SubjectApp, Level: store.LevelApp, OrganizationID: s.app.organization.ID, AppID: s.customerID(),
		Interval: interval, LimitMicros: micros, UpdatedBy: "test",
	}
	s.Require().NoError(s.store.CreateLimit(s.T().Context(), &limit, store.LimitAudit{CustomerID: s.customerID(), Source: store.AuditSourceAPI}))
	return limit
}

func (s *RouterSuite) policies(logger *slog.Logger) *policy.Enforcer {
	settings := policy.Settings{Mode: s.limitsMode, Fresh: suiteFresh, SpendEvery: suiteFresh, SumEvery: suiteFresh}
	if s.holds != nil {
		settings.Holds = s.holds
		settings.FlushEvery, settings.HoldRefresh, settings.FlushMaxLag = 10*time.Millisecond, 50*time.Millisecond, suiteFlushMaxLag
	}
	if s.limited {
		s.shipping = true
		// A database of the suite's own: suites run in parallel, and starting one empties it.
		database := "usage_" + strings.ToLower(strings.TrimPrefix(s.T().Name(), "Test"))
		s.usage = usagetest.Start(s.T(), usagetest.Options{
			DSN: usagetest.Database(s.T(), database), ManageKey: "manage-" + s.utils.uuid(), Logger: logger,
		})
		s.usageFront = newUsageFront(s.T(), s.usage)
		client, err := usage.NewClient(usage.ClientOptions{URL: s.usageFront.server.URL, Token: s.usage.Token})
		s.Require().NoError(err)
		settings.Usage, settings.MaxStale, settings.PendingTTL = client, time.Minute, time.Minute
	}
	enforcer, err := policy.New(s.configs, settings, logger)
	s.Require().NoError(err)
	s.T().Cleanup(enforcer.Close)
	s.enforcer = enforcer
	return enforcer
}

// usageFront is the usage service as the suite's policies reach it: the real one, through a
// proxy a test can take down, as the service going away does.
type usageFront struct {
	server *httptest.Server
	down   atomic.Bool
}

func newUsageFront(t *testing.T, service *usagetest.Service) *usageFront {
	target, err := url.Parse(service.URL)
	require.NoError(t, err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	front := &usageFront{}
	front.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if front.down.Load() {
			panic(http.ErrAbortHandler)
		}
		proxy.ServeHTTP(w, r)
	}))
	t.Cleanup(front.server.Close)
	return front
}

// quota caps what one end user may spend in a day, unless their app's policy says otherwise.
// The default is high enough that nothing a test does reaches it; a suite about the cap sets
// messagesPerDay before it starts the harness.
func (s *RouterSuite) quota(liveClient *live.Client, policies *policy.Enforcer, logger *slog.Logger) *quota.Limiter {
	allowed := s.messagesPerDay
	if allowed == 0 {
		allowed = 1_000_000
	}
	limiter, err := quota.New(liveClient.Redis(), quota.Limits{MessagesPerDay: allowed}, policies, logger)
	s.Require().NoError(err)
	return limiter
}

// gateOnClosedStore has the gate every text and call passes read from a store closed for the
// test, so its opt-out check fails as an unreachable database does, which is no refusal. It
// returns the way to put the gate back, which the test's end does too.
func (s *RouterSuite) gateOnClosedStore() (restore func()) {
	closed, err := store.Open(testDatabase(s.T(), os.Getenv("ROUTER_POSTGRES_DSN")))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	original := *s.gate
	*s.gate = *dlc.NewGate(closed, s.live.Redis(), s.sandbox, nil)
	restored := false
	restore = func() {
		if !restored {
			*s.gate = original
			restored = true
		}
	}
	s.T().Cleanup(restore)
	return restore
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
	s.own(app.app.ID)
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
	read, err := readAll(response)
	require.NoError(err)
	response.Body = io.NopCloser(bytes.NewReader(read))
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
	status, payload := c.call(http.MethodPost, "/v1/agents/sessions", request)
	c.suite.Require().Equal(http.StatusCreated, status, string(payload))
	c.suite.Require().NoError(json.Unmarshal(payload, &created))
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
	s.own(app.ID)
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

// instructedAgent is the id of an agent config of the suite's app told instructions.
func (d testData) instructedAgent(instructions string) *string {
	var created AgentConfig
	d.suite.Require().Equal(http.StatusCreated, d.suite.serverClient.do(
		http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
			Name: "agent-" + d.suite.utils.uuid(), Instructions: &instructions,
		}, &created))
	return &created.Id
}

// briefed is request with instructions of its own, such as the brief of a customer somebody
// practises on, returned with the brief.
func (d testData) briefed(request CreateSessionRequest) (CreateSessionRequest, string) {
	brief := "Be a customer whose order " + d.suite.utils.uuid() + " never came."
	request.Instructions = &brief
	return request, brief
}

// storedSession is the id of a session of the suite's app that is not running, its row put
// straight into the store, for an endpoint that only asks whose a session is.
func (d testData) storedSession() string {
	id := "session-" + d.suite.utils.uuid()
	d.suite.Require().NoError(d.suite.store.SaveSession(context.Background(),
		&store.AgentSession{ID: id, CustomerID: d.suite.customerID()}))
	return id
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
	target := "en-low-latency"
	return CreateSessionRequest{Id: id, Llm: &target}
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

// knownApps are the suite's apps saying which one is the deployment's own.
type knownApps struct{ *suiteApps }

func (a knownApps) App() int64 { return a.deployment.App() }

// set changes how the suite's apps answer for a customer, and forgets what they said.
func (s *RouterSuite) setApps(customer string, change func(*suiteApps)) {
	s.apps.mu.Lock()
	change(s.apps)
	s.apps.mu.Unlock()
	s.stream.Invalidate(customer)
}

// authMode is the mode the suite's router says it runs in: proxy in app mode, which only a
// gateway serves, though the suite's callers still sign with the app's key.
func (s *RouterSuite) authMode() auth.Mode {
	if s.appMode || s.proxied {
		return auth.Proxy
	}
	return auth.APIKey
}

// giveApp makes the customer act in an app of its own, a chattest of its own, from now on.
func (s *RouterSuite) giveApp(customer string, app int64, key string) *chattest.Server {
	own := chattest.NewServer(s.T())
	s.apps.mu.Lock()
	s.apps.own[customer] = streamapp.Identity{
		CustomerID: customer, StreamApp: app, APIKey: key,
		Secret: streamapp.NewSecret(key + "-secret"), BaseURL: own.URL,
		Registered: true,
	}
	s.apps.mu.Unlock()
	s.stream.Invalidate(customer)
	return own
}
