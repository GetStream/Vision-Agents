package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

type ConfigSuite struct {
	suite.Suite
}

func TestConfigSuite(t *testing.T) {
	suite.Run(t, new(ConfigSuite))
}

// clean keeps one test's variables out of the next one's, since Load both reads the
// environment and writes to it.
func (s *ConfigSuite) SetupTest() {
	for _, variable := range variables {
		s.T().Setenv(variable, "")
		s.Require().NoError(os.Unsetenv(variable))
	}
	s.T().Setenv(EnvVar, Local)
	s.T().Setenv(FileVar, "")
	s.Require().NoError(os.Unsetenv(FileVar))
}

func (s *ConfigSuite) TestEveryEnvironmentLoads() {
	for _, name := range []string{Local, Staging, Testing} {
		s.T().Setenv(EnvVar, name)
		config, from, err := Load("")
		s.Require().NoError(err, name)
		s.Equal(name, from)
		s.Equal(":8080", config.Addr, "the default fills in what no file sets")
	}
}

func (s *ConfigSuite) TestTestingUsesItsOwnDatabase() {
	s.T().Setenv(EnvVar, Local)
	local, _, err := Load("")
	s.Require().NoError(err)

	s.T().Setenv(EnvVar, Testing)
	testing, _, err := Load("")
	s.Require().NoError(err)

	s.NotEqual(local.Postgres.DSN, testing.Postgres.DSN)
	s.Contains(testing.Postgres.DSN, "_test?", "the store suite refuses any other database")
}

func (s *ConfigSuite) TestTheEnvironmentWinsOverTheEmbeddedFile() {
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://elsewhere/db")

	config, _, err := Load("")
	s.Require().NoError(err)

	s.Equal("postgres://elsewhere/db", config.Postgres.DSN)
	s.True(strings.HasPrefix(config.Redis.Addr, "localhost:"), "the file fills in what is unset")
}

func (s *ConfigSuite) TestTheTestingFileWinsOverTheEnvironment() {
	s.T().Setenv(EnvVar, Testing)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost:55432/model_router")

	config, _, err := Load("")
	s.Require().NoError(err)

	s.Contains(config.Postgres.DSN, "model_router_test")
}

func (s *ConfigSuite) TestAFileOfYourOwn() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte(`
addr: ":9000"
postgres:
  dsn: postgres://db/self_hosted
redis:
  addr: redis:6379
auth:
  mode: api_key
  kek: a-passphrase
data_move:
  retention: 48h
`), 0o600))

	config, from, err := Load(path)
	s.Require().NoError(err)

	s.Equal(path, from)
	s.Equal(":9000", config.Addr)
	s.Equal("postgres://db/self_hosted", config.Postgres.DSN)
	s.Equal("redis:6379", config.Redis.Addr)
	s.Equal("api_key", config.Auth.Mode)
	s.Equal(48*time.Hour, config.DataMove.Retention)
	s.EqualValues(200, config.RateLimit.MessagesPerDay, "a file that says nothing keeps the default")
}

func (s *ConfigSuite) TestAFileIsFoundThroughTheEnvironmentToo() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("addr: \":9100\"\n"), 0o600))
	s.T().Setenv(FileVar, path)

	config, _, err := Load("")
	s.Require().NoError(err)

	s.Equal(":9100", config.Addr)
}

func (s *ConfigSuite) TestAMissingFileIsRefused() {
	_, _, err := Load(filepath.Join(s.T().TempDir(), "absent.yaml"))
	s.ErrorContains(err, "absent.yaml")
}

func (s *ConfigSuite) TestAnUnknownEnvironmentIsRefused() {
	s.T().Setenv(EnvVar, "production")
	_, _, err := Load("")
	s.ErrorContains(err, `unknown ROUTER_ENV "production"`)
}

func (s *ConfigSuite) TestDevelopmentStillNamesTheLocalFile() {
	s.T().Setenv(EnvVar, "development")
	config, from, err := Load("")
	s.Require().NoError(err)

	s.Equal(Local, from)
	s.Contains(config.Postgres.DSN, "model_router")
}

func (s *ConfigSuite) TestSettingsAreReadableFromTheEnvironmentAfterwards() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("postgres:\n  dsn: postgres://db/exported\n"), 0o600))

	_, _, err := Load(path)
	s.Require().NoError(err)

	s.Equal("postgres://db/exported", os.Getenv("ROUTER_POSTGRES_DSN"),
		"cmd/agent and the suites find the database the same way")
}

func (s *ConfigSuite) TestSpeculativeRepliesAreOffUnlessAskedFor() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.False(config.Agent.SpeculativeReplies)

	s.T().Setenv("ROUTER_SPECULATIVE_REPLIES", "true")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.True(config.Agent.SpeculativeReplies)
}

func (s *ConfigSuite) TestConnectorsAreOffUnlessAskedFor() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.False(config.Connectors.Enabled)

	s.T().Setenv("ROUTER_CONNECTORS_ENABLED", "true")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.True(config.Connectors.Enabled)
}

// TestDirectCallsAreCappedAtSixtyAMinuteUnlessAskedOtherwise: the default, the variable that
// changes it, 0 that turns it off, and a negative cap refused (AI-958).
func (s *ConfigSuite) TestDirectCallsAreCappedAtSixtyAMinuteUnlessAskedOtherwise() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.EqualValues(60, config.Connectors.ProxyCallsPerMinute)

	s.T().Setenv("ROUTER_CONNECTORS_PROXY_CALLS_PER_MINUTE", "0")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Zero(config.Connectors.ProxyCallsPerMinute)

	s.T().Setenv("ROUTER_CONNECTORS_PROXY_CALLS_PER_MINUTE", "-1")
	_, _, err = Load("")
	s.ErrorContains(err, "connectors.proxy_calls_per_minute")
}

// TestRunsAreCappedAtTwentyPerAppUnlessAskedOtherwise: the default, the variable that changes
// it, 0 that turns it off, and a negative cap refused.
func (s *ConfigSuite) TestRunsAreCappedAtTwentyPerAppUnlessAskedOtherwise() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.EqualValues(20, config.Runs.ConcurrentPerApp)

	s.T().Setenv("ROUTER_RUNS_CONCURRENT_PER_APP", "0")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Zero(config.Runs.ConcurrentPerApp)

	s.T().Setenv("ROUTER_RUNS_CONCURRENT_PER_APP", "-1")
	_, _, err = Load("")
	s.ErrorContains(err, "runs.concurrent_per_app")
}

func (s *ConfigSuite) TestAnEpisodeIsIdleAfterAnHourUnlessAskedFor() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(time.Hour, config.Episodes.IdleAfter)

	s.T().Setenv("ROUTER_EPISODES_IDLE_AFTER", "30m")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(30*time.Minute, config.Episodes.IdleAfter)
}

// WhatsApp's window is 24 hours from the person's last message, so an episode closes inside
// it; and an episode that closes at once would summarize every message alone.
func (s *ConfigSuite) TestAnEpisodeIdlePeriodOfADayOrOfNothingIsRefused() {
	for _, idle := range []string{"24h", "0s"} {
		s.T().Setenv("ROUTER_EPISODES_IDLE_AFTER", idle)
		_, _, err := Load("")
		s.ErrorContains(err, "episodes.idle_after", idle)
	}
	s.T().Setenv("ROUTER_EPISODES_IDLE_AFTER", "23h59m")
	_, _, err := Load("")
	s.NoError(err)
}

func (s *ConfigSuite) TestTheProxyDeclaresNoKindUnlessAskedFor() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.False(config.Auth.ProxyDeclaresKind)

	s.T().Setenv("ROUTER_AUTH_PROXY_DECLARES_KIND", "true")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.True(config.Auth.ProxyDeclaresKind)
}

func (s *ConfigSuite) TestStreamTenancyDefaultsToDeployment() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(config.Stream.Tenancy)

	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyDeployment)
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(TenancyDeployment, config.Stream.Tenancy)
}

func (s *ConfigSuite) TestStreamTenancyIsDeploymentOrApp() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", "shared")
	_, _, err := Load("")
	s.ErrorContains(err, "stream.tenancy")
}

func (s *ConfigSuite) TestAppModeIsRefusedWithoutPostgres() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "")

	_, _, err := Load("")

	s.ErrorContains(err, "postgres.dsn")
}

// behindTheGateway sets what app mode needs: Postgres, a proxy that declares kinds, and the
// gateway's mint endpoint.
func (s *ConfigSuite) behindTheGateway() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost/router")
	s.T().Setenv("ROUTER_AUTH_MODE", "proxy")
	s.T().Setenv("ROUTER_AUTH_PROXY_DECLARES_KIND", "true")
	s.T().Setenv("ROUTER_STREAM_MINT_URL", "http://gateway-internal:8082")
	s.T().Setenv("ROUTER_STREAM_MINT_TOKEN", "the-routers-bearer")
}

func (s *ConfigSuite) TestAppModeRefusesAUserTokenInTheEnvironment() {
	s.behindTheGateway()
	s.T().Setenv("STREAM_USER_TOKEN", "one-apps-token")

	_, _, err := Load("")

	s.ErrorContains(err, "stream.user_token")
	s.NotContains(err.Error(), "one-apps-token")
}

func (s *ConfigSuite) TestAppModeStartsBehindTheGateway() {
	s.behindTheGateway()

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal(TenancyApp, config.Stream.Tenancy)
	s.Equal("http://gateway-internal:8082", config.Stream.MintURL)
}

func (s *ConfigSuite) TestAppModeRefusesAnythingButAProxy() {
	for _, mode := range []string{"noauth", "api_key"} {
		s.Run(mode, func() {
			s.behindTheGateway()
			s.T().Setenv("ROUTER_AUTH_MODE", mode)

			_, _, err := Load("")

			s.ErrorContains(err, "auth.mode=proxy")
		})
	}
}

func (s *ConfigSuite) TestAppModeRefusesAProxyThatDeclaresNoKind() {
	s.behindTheGateway()
	s.T().Setenv("ROUTER_AUTH_PROXY_DECLARES_KIND", "false")

	_, _, err := Load("")

	s.ErrorContains(err, "auth.proxy_declares_kind")
}

func (s *ConfigSuite) TestAppModeRefusesToStartWithoutTheMintEndpoint() {
	s.behindTheGateway()
	s.T().Setenv("ROUTER_STREAM_MINT_TOKEN", "")

	_, _, err := Load("")

	s.ErrorContains(err, "stream.mint_url")
}

func (s *ConfigSuite) TestStreamFallbackDefaultsToRefuseInAppMode() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(config.Stream.Fallback)
	s.Equal(FallbackRefuse, config.Stream.EffectiveFallback())

	s.T().Setenv("ROUTER_STREAM_FALLBACK", FallbackDeployment)
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(FallbackDeployment, config.Stream.EffectiveFallback())

	s.T().Setenv("ROUTER_STREAM_FALLBACK", "maybe")
	_, _, err = Load("")
	s.ErrorContains(err, "stream.fallback")
}

func (s *ConfigSuite) TestTheDeploymentsAppIDIsReadAsANumber() {
	s.T().Setenv("ROUTER_STREAM_APP_ID", "1234")
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(int64(1234), config.Stream.AppID)

	s.T().Setenv("ROUTER_STREAM_APP_ID", "-4")
	_, _, err = Load("")
	s.ErrorContains(err, "stream.app_id")
}

func (s *ConfigSuite) TestTheStreamBaseURLIsReadOnce() {
	s.T().Setenv("STREAM_BASE_URL", "https://chat.example.test")
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal("https://chat.example.test", config.Stream.BaseURL)
}

func (s *ConfigSuite) TestANegativeLimitIsRefused() {
	s.T().Setenv("ROUTER_RATE_LIMIT_MESSAGES_PER_DAY", "-1")
	_, _, err := Load("")
	s.ErrorContains(err, "cannot be negative")
}

func (s *ConfigSuite) TestALimitThatIsNotANumberIsRefused() {
	s.T().Setenv("ROUTER_RATE_LIMIT_TOKENS_PER_DAY", "lots")
	_, _, err := Load("")
	s.Error(err)
}

func (s *ConfigSuite) TestALateReplyIsAskedOfAnotherCandidateAfterTheHedgeUnlessTurnedOff() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(1200*time.Millisecond, config.Agent.ReplyHedge)

	s.T().Setenv("ROUTER_REPLY_HEDGE", "800ms")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(800*time.Millisecond, config.Agent.ReplyHedge)

	s.T().Setenv("ROUTER_REPLY_HEDGE", "0")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Zero(config.Agent.ReplyHedge, "zero asks once rather than falling back to the default")
	s.Equal("0s", os.Getenv("ROUTER_REPLY_HEDGE"), "what is exported says the same")

	s.T().Setenv("ROUTER_REPLY_HEDGE", "-1s")
	_, _, err = Load("")
	s.Require().Error(err)
	s.Contains(err.Error(), "agent.reply_hedge")
}

func (s *ConfigSuite) TestALateFlowDecisionIsAskedOfAnotherCandidateAfterItsOwnHedgeUnlessTurnedOff() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(400*time.Millisecond, config.Agent.FlowHedge, "a decision is hedged sooner than a reply")

	s.T().Setenv("ROUTER_FLOW_HEDGE", "250ms")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(250*time.Millisecond, config.Agent.FlowHedge)
	s.Equal(1200*time.Millisecond, config.Agent.ReplyHedge, "the reply's hedge is its own")

	s.T().Setenv("ROUTER_FLOW_HEDGE", "0")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Zero(config.Agent.FlowHedge, "zero asks once rather than falling back to the default")
	s.Equal("0s", os.Getenv("ROUTER_FLOW_HEDGE"), "what is exported says the same")

	s.T().Setenv("ROUTER_FLOW_HEDGE", "-1s")
	_, _, err = Load("")
	s.Require().Error(err)
	s.Contains(err.Error(), "agent.flow_hedge")
}

func (s *ConfigSuite) TestChatTimingsAreOffUnlessTurnedOn() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.False(config.Agent.ChatTimings)

	s.T().Setenv("ROUTER_CHAT_TIMINGS", "true")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.True(config.Agent.ChatTimings)
	s.Equal("true", os.Getenv("ROUTER_CHAT_TIMINGS"), "what is exported says the same")
}

func (s *ConfigSuite) TestTelnyxSettingsAreReadFromTheEnvironment() {
	s.T().Setenv("TELNYX_API_KEY", "KEY123")
	s.T().Setenv("TELNYX_CONNECTION_ID", "conn-1")
	s.T().Setenv("TELNYX_PUBLIC_KEY", "cHVibGlj")

	config, _, err := Load("")
	s.Require().NoError(err)

	s.Equal(Telnyx{APIKey: "KEY123", ConnectionID: "conn-1", PublicKey: "cHVibGlj"}, config.Telnyx)
}

// The Telnyx provider reads its own variables, so settings from a file have to reach them.
func (s *ConfigSuite) TestTelnyxSettingsInAFileReachTheEnvironment() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte(
		"telnyx:\n  api_key: KEY123\n  connection_id: conn-1\n  public_key: cHVibGlj\n"), 0o600))

	_, _, err := Load(path)
	s.Require().NoError(err)

	s.Equal("KEY123", os.Getenv("TELNYX_API_KEY"))
	s.Equal("conn-1", os.Getenv("TELNYX_CONNECTION_ID"))
	s.Equal("cHVibGlj", os.Getenv("TELNYX_PUBLIC_KEY"))
}

func (s *ConfigSuite) TestASingleNodeSetsNoShard() {
	config, _, err := Load("")
	s.Require().NoError(err)

	s.Empty(config.Shard)
	s.Empty(config.Postgres.ShardDSN)
	s.Empty(config.Postgres.Shards)
	s.Empty(config.Redis.Shard.Addr)
}

func (s *ConfigSuite) TestAShardIsReadFromAFile() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte(`
shard: us-west1
postgres:
  dsn: postgres://global/router
  shard_dsn: postgres://usw/router
  shards:
    us-east4: postgres://use/router
    us-west1: postgres://usw/router
redis:
  addr: global:6379
  shard:
    addr: usw:6379
`), 0o600))

	config, _, err := Load(path)
	s.Require().NoError(err)

	s.Equal("us-west1", config.Shard)
	s.Equal("postgres://usw/router", config.Postgres.ShardDSN)
	s.Equal(map[string]string{"us-east4": "postgres://use/router", "us-west1": "postgres://usw/router"}, config.Postgres.Shards)
	s.Equal("usw:6379", config.Redis.Shard.Addr)
}

func (s *ConfigSuite) TestAShardIsReadFromTheEnvironment() {
	s.T().Setenv("ROUTER_SHARD", "us-west1")
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://global/router")
	s.T().Setenv("ROUTER_POSTGRES_SHARD_DSN", "postgres://usw/router")
	s.T().Setenv("ROUTER_POSTGRES_SHARDS", "us-east4=postgres://use/router, us-west1=postgres://usw/router")
	s.T().Setenv("ROUTER_REDIS_SHARD_ADDR", "usw:6379")

	config, _, err := Load("")
	s.Require().NoError(err)

	s.Equal(map[string]string{"us-east4": "postgres://use/router", "us-west1": "postgres://usw/router"}, config.Postgres.Shards)
	s.Equal("usw:6379", config.Redis.Shard.Addr)
	s.Equal("us-east4=postgres://use/router,us-west1=postgres://usw/router", os.Getenv("ROUTER_POSTGRES_SHARDS"),
		"written back the way it is read")
}

func (s *ConfigSuite) TestAShardDatabaseWithoutTheSharedOneIsRefused() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("postgres:\n  dsn: \"\"\n  shard_dsn: postgres://usw/router\n"), 0o600))

	_, _, err := Load(path)
	s.ErrorContains(err, "set postgres.dsn")
}

func (s *ConfigSuite) TestAShardRedisWithoutTheSharedOneIsRefused() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("redis:\n  addr: \"\"\n  shard:\n    addr: usw:6379\n"), 0o600))

	_, _, err := Load(path)
	s.ErrorContains(err, "set redis.addr")
}

func (s *ConfigSuite) TestShardsThatLeaveThisOneOutAreRefused() {
	s.T().Setenv("ROUTER_SHARD", "us-west1")
	s.T().Setenv("ROUTER_POSTGRES_SHARD_DSN", "postgres://usw/router")
	s.T().Setenv("ROUTER_POSTGRES_SHARDS", "us-east4=postgres://use/router")

	_, _, err := Load("")
	s.ErrorContains(err, `does not list this shard, "us-west1"`)
}

func (s *ConfigSuite) TestShardsWithoutThisOnesNameAreRefused() {
	s.T().Setenv("ROUTER_POSTGRES_SHARDS", "us-east4=postgres://use/router")

	_, _, err := Load("")
	s.ErrorContains(err, "set shard")
}

func (s *ConfigSuite) TestShardsListingThisOneElsewhereAreRefused() {
	s.T().Setenv("ROUTER_SHARD", "us-west1")
	s.T().Setenv("ROUTER_POSTGRES_SHARD_DSN", "postgres://usw/router")
	s.T().Setenv("ROUTER_POSTGRES_SHARDS", "us-west1=postgres://other/router")

	_, _, err := Load("")
	s.ErrorContains(err, "another DSN")
}
