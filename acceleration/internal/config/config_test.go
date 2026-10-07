package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
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

func (s *ConfigSuite) TestAcousticEndpointDefaultsToHostedPrimaryAndRemainsEnvironmentBacked() {
	defaults, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(eotdefaults.HostedDemoEndpoint, defaults.EOT.Endpoint)
	s.Equal("primary", defaults.EOT.Mode)
	s.Equal(0.5, defaults.EOT.Threshold)

	s.T().Setenv("ROUTER_EOT_MODE", "primary")
	s.T().Setenv("ROUTER_EOT_URL", "https://eot.example.run.app")
	s.T().Setenv("ROUTER_EOT_ID_TOKEN_FILE", "/run/secrets/eot-id-token")
	s.T().Setenv("ROUTER_EOT_THRESHOLD", "0.72")
	configured, _, err := Load("")
	s.Require().NoError(err)
	s.Equal("primary", configured.EOT.Mode)
	s.Equal("https://eot.example.run.app", configured.EOT.Endpoint)
	s.Equal("/run/secrets/eot-id-token", configured.EOT.IDTokenFile)
	s.Equal(0.72, configured.EOT.Threshold)
}

func (s *ConfigSuite) TestEmptyEndpointFromEnvironmentOrYAMLStaysDisabled() {
	s.T().Setenv("ROUTER_EOT_URL", "")
	fromEnvironment, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(fromEnvironment.EOT.Endpoint)
	s.Equal("gate", fromEnvironment.EOT.Mode)
	value, present := os.LookupEnv("ROUTER_EOT_URL")
	s.True(present)
	s.Empty(value)

	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("eot:\n  endpoint: ''\n"), 0o600))
	s.Require().NoError(os.Unsetenv("ROUTER_EOT_URL"))
	fromFile, _, err := Load(path)
	s.Require().NoError(err)
	s.Empty(fromFile.EOT.Endpoint)
	s.Equal("gate", fromFile.EOT.Mode)
	value, present = os.LookupEnv("ROUTER_EOT_URL")
	s.True(present, "an explicit empty YAML endpoint must survive config export")
	s.Empty(value)

	reloaded, _, err := Load(path)
	s.Require().NoError(err)
	s.Empty(reloaded.EOT.Endpoint, "a second env-backed load must not re-enable the hosted scorer")
	s.Equal("gate", reloaded.EOT.Mode)
}

func (s *ConfigSuite) TestCustomEndpointDefaultsToGateAndPrivateAuthStaysExplicit() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("eot:\n  endpoint: https://private.example/v1/eot\n  id_token_file: /run/secrets/eot-token\n"), 0o600))
	settings, _, err := Load(path)
	s.Require().NoError(err)
	s.Equal("https://private.example/v1/eot", settings.EOT.Endpoint)
	s.Equal("gate", settings.EOT.Mode)
	s.Equal("/run/secrets/eot-token", settings.EOT.IDTokenFile)

	s.Require().NoError(os.Unsetenv("ROUTER_EOT_URL"))
	s.Require().NoError(os.Unsetenv("ROUTER_EOT_MODE"))
	s.T().Setenv("ROUTER_EOT_ID_TOKEN_FILE", "/run/secrets/eot-token")
	_, _, err = Load("")
	s.ErrorContains(err, "cannot be used with the hosted demo endpoint")

	s.T().Setenv("ROUTER_EOT_URL", eotdefaults.HostedDemoEndpoint)
	_, _, err = Load("")
	s.ErrorContains(err, "cannot be used with the hosted demo endpoint")

	for _, endpoint := range []string{
		"https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app",
		"https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app:443/v1/eot",
		"https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app./v1/eot",
	} {
		s.T().Setenv("ROUTER_EOT_URL", endpoint)
		_, _, err = Load("")
		s.ErrorContains(err, "cannot be used with the hosted demo endpoint", endpoint)
	}
}

func (s *ConfigSuite) TestExplicitModeWinsForTheHostedEndpointAndEmptyModeIsInvalid() {
	s.T().Setenv("ROUTER_EOT_URL", eotdefaults.HostedDemoEndpoint)
	s.T().Setenv("ROUTER_EOT_MODE", "gate")
	settings, _, err := Load("")
	s.Require().NoError(err)
	s.Equal("gate", settings.EOT.Mode)

	s.T().Setenv("ROUTER_EOT_MODE", "")
	_, _, err = Load("")
	s.ErrorContains(err, "eot.mode")
}

func (s *ConfigSuite) TestEOTModeMustBeGateOrPrimary() {
	for _, mode := range []string{"semantic", "PRIMARY"} {
		s.T().Setenv("ROUTER_EOT_MODE", mode)
		_, _, err := Load("")
		s.ErrorContains(err, "eot.mode")
		s.T().Setenv("ROUTER_EOT_MODE", "")
		s.Require().NoError(os.Unsetenv("ROUTER_EOT_MODE"))
	}
}

func (s *ConfigSuite) TestAcousticThresholdMustBeFiniteAndInRange() {
	for _, threshold := range []string{"-0.1", "1.1", "NaN", "+Inf"} {
		s.T().Setenv("ROUTER_EOT_THRESHOLD", threshold)
		_, _, err := Load("")
		s.ErrorContains(err, "eot.threshold", threshold)
		s.T().Setenv("ROUTER_EOT_THRESHOLD", "")
		s.Require().NoError(os.Unsetenv("ROUTER_EOT_THRESHOLD"))
	}
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
cors_origins: [https://agents.example.com]
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
	s.Equal([]string{"https://agents.example.com"}, config.CORSOrigins)
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

func (s *ConfigSuite) TestSpeculativeRepliesAreOnUnlessTurnedOff() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.True(config.Agent.SpeculativeReplies)

	s.T().Setenv("ROUTER_SPECULATIVE_REPLIES", "false")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.False(config.Agent.SpeculativeReplies)

	s.T().Setenv("ROUTER_SPECULATIVE_REPLIES", "true")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.True(config.Agent.SpeculativeReplies)
}

func (s *ConfigSuite) TestAReplyWaitsForTheCallerToBeQuietUnlessTurnedOff() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(700*time.Millisecond, config.Agent.ReplySilence)

	s.T().Setenv("ROUTER_REPLY_SILENCE", "250ms")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(250*time.Millisecond, config.Agent.ReplySilence)

	s.T().Setenv("ROUTER_REPLY_SILENCE", "0")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Zero(config.Agent.ReplySilence, "zero turns the wait off rather than falling back to the default")
	s.Equal("0s", os.Getenv("ROUTER_REPLY_SILENCE"), "what is exported says the same")

	s.T().Setenv("ROUTER_REPLY_SILENCE", "-1s")
	_, _, err = Load("")
	s.Require().Error(err)
	s.Contains(err.Error(), "agent.reply_silence")
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

func (s *ConfigSuite) TestAppModeRefusesAUserTokenInTheEnvironment() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost/router")
	s.T().Setenv("STREAM_USER_TOKEN", "one-apps-token")

	_, _, err := Load("")

	s.ErrorContains(err, "stream.user_token")
	s.NotContains(err.Error(), "one-apps-token")
}

func (s *ConfigSuite) TestAppModeStartsWithPostgresAndNoUserToken() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost/router")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal(TenancyApp, config.Stream.Tenancy)
}

func (s *ConfigSuite) TestAppModeRefusesNoAuth() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost/router")
	s.T().Setenv("ROUTER_AUTH_MODE", "noauth")

	_, _, err := Load("")

	s.ErrorContains(err, "auth.mode=noauth")
}

func (s *ConfigSuite) TestAppModeRefusesAProxyThatDeclaresNoKind() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost/router")
	s.T().Setenv("ROUTER_AUTH_MODE", "proxy")

	_, _, err := Load("")

	s.ErrorContains(err, "auth.proxy_declares_kind")
}

func (s *ConfigSuite) TestAppModeStartsBehindAProxyThatDeclaresKinds() {
	s.T().Setenv("ROUTER_STREAM_TENANCY", TenancyApp)
	s.T().Setenv("ROUTER_POSTGRES_DSN", "postgres://localhost/router")
	s.T().Setenv("ROUTER_AUTH_MODE", "proxy")
	s.T().Setenv("ROUTER_AUTH_PROXY_DECLARES_KIND", "true")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal(TenancyApp, config.Stream.Tenancy)
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

func (s *ConfigSuite) TestRegistrationSettingsAreOffAndEmptyUnlessSet() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.False(config.Stream.TrustAPIKeyHeader)
	s.Empty(config.Stream.DenyRegistration)

	s.T().Setenv("ROUTER_STREAM_TRUST_API_KEY_HEADER", "true")
	s.T().Setenv("ROUTER_STREAM_DENY_REGISTRATION", "11,22")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.True(config.Stream.TrustAPIKeyHeader)
	s.Equal([]string{"11", "22"}, config.Stream.DenyRegistration)
}
