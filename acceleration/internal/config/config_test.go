package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
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

func (s *ConfigSuite) TestTheKnowledgeProvidersCostIsConfigThatAFileReplaces() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("knowledge:\n  stream_cost:\n    bytes_written:\n      usd: 2.6\n      per: 1000000000\n      minimum: 10000\n"), 0o600))

	config, _, err := Load(path)

	s.Require().NoError(err)
	s.Equal(pricing.Rate{USD: 2.6, Per: 1_000_000_000, Minimum: 10_000}, config.Knowledge.StreamCost["bytes_written"],
		"a price change of the provider's is a line of config")
	s.Equal(pricing.Rate{USD: 0.33, Per: 730_000_000_000}, config.Knowledge.StreamCost["byte_hours"],
		"what the file does not name keeps the provider's list price")
}

func (s *ConfigSuite) TestAKnowledgeCostWithNoUnitIsRefused() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("knowledge:\n  stream_cost:\n    bytes_written:\n      usd: 2.6\n"), 0o600))

	_, _, err := Load(path)

	s.ErrorContains(err, "knowledge.stream_cost.bytes_written")
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

func (s *ConfigSuite) TestKnowledgeIsCutByTheIngestLimitsUnlessAskedOtherwise() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(500, config.Knowledge.MinChunk)
	s.Equal(128, config.Knowledge.MaxHeading)

	s.T().Setenv("ROUTER_KNOWLEDGE_MIN_CHUNK", "800")
	s.T().Setenv("ROUTER_KNOWLEDGE_MAX_HEADING", "64")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(800, config.Knowledge.MinChunk)
	s.Equal(64, config.Knowledge.MaxHeading)
}

func (s *ConfigSuite) TestAHeadingLimitNotUnderTheMinimumChunkIsRefused() {
	s.T().Setenv("ROUTER_KNOWLEDGE_MIN_CHUNK", "500")
	s.T().Setenv("ROUTER_KNOWLEDGE_MAX_HEADING", "500")

	_, _, err := Load("")

	s.ErrorContains(err, "knowledge.max_heading")
	s.ErrorContains(err, "knowledge.min_chunk")
}

func (s *ConfigSuite) TestANegativeKnowledgeLimitIsRefused() {
	s.T().Setenv("ROUTER_KNOWLEDGE_MAX_HEADING", "-1")

	_, _, err := Load("")

	s.ErrorContains(err, "knowledge.max_heading")
}

func (s *ConfigSuite) TestKnowledgeLimitsOfZeroAreAccepted() {
	s.T().Setenv("ROUTER_KNOWLEDGE_MIN_CHUNK", "0")
	s.T().Setenv("ROUTER_KNOWLEDGE_MAX_HEADING", "0")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Zero(config.Knowledge.MinChunk)
	s.Zero(config.Knowledge.MaxHeading)
}

// TestTheCodeSandboxBrokerIsDaytonaAndNamesNoEnvironmentUnlessAskedFor: the defaults, the
// variables that set them, and a broker there is none of refused.
func (s *ConfigSuite) TestTheCodeSandboxBrokerIsDaytonaAndNamesNoEnvironmentUnlessAskedFor() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(CodeSandbox{Broker: "daytona", IdleStop: 15 * time.Minute, StoppedDelete: time.Hour}, config.CodeSandbox)

	s.T().Setenv("ROUTER_CODE_SANDBOX_SECRET_PREFIX", "stg")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(CodeSandbox{Broker: "daytona", SecretPrefix: "stg", IdleStop: 15 * time.Minute, StoppedDelete: time.Hour},
		config.CodeSandbox)

	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "vault")
	_, _, err = Load("")
	s.ErrorContains(err, "code_sandbox.broker is daytona or proxy")
}

// TestASandboxStopsAfter15IdleMinutesAndIsDeletedAnHourLater: the defaults, the variables that
// set them, and intervals Daytona cannot take refused.
func (s *ConfigSuite) TestASandboxStopsAfter15IdleMinutesAndIsDeletedAnHourLater() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_IDLE_STOP", "5m")
	s.T().Setenv("ROUTER_CODE_SANDBOX_STOPPED_DELETE", "2h")
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(5*time.Minute, config.CodeSandbox.IdleStop)
	s.Equal(2*time.Hour, config.CodeSandbox.StoppedDelete)

	for _, interval := range []string{"0s", "30s", "90s"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_STOPPED_DELETE", interval)
		_, _, err = Load("")
		s.ErrorContains(err, "code_sandbox.stopped_delete (ROUTER_CODE_SANDBOX_STOPPED_DELETE) is a whole number of minutes", interval)
	}
	s.T().Setenv("ROUTER_CODE_SANDBOX_STOPPED_DELETE", "1h")
	s.T().Setenv("ROUTER_CODE_SANDBOX_IDLE_STOP", "-1m")
	_, _, err = Load("")
	s.ErrorContains(err, "code_sandbox.idle_stop (ROUTER_CODE_SANDBOX_IDLE_STOP)")
}

// TestASandboxNamesItsDeploymentOnlyWhenAskedTo: empty by default, set by its variable, and a
// value Daytona could not take as a label refused.
func (s *ConfigSuite) TestASandboxNamesItsDeploymentOnlyWhenAskedTo() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(config.CodeSandbox.Deployment)

	s.T().Setenv("ROUTER_CODE_SANDBOX_DEPLOYMENT", "eu-west.blue")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal("eu-west.blue", config.CodeSandbox.Deployment)

	for _, refused := range []string{"Blue", "-blue", "blue deployment"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_DEPLOYMENT", refused)
		_, _, err = Load("")
		s.ErrorContains(err, "code_sandbox.deployment (ROUTER_CODE_SANDBOX_DEPLOYMENT)", refused)
	}
}

// TestTheProxyBrokerNeedsTheProxysURLAndCA: a sandbox must reach the proxy and trust it, its
// own credential is the URL's user, and the router serves the proxy, with the CA's key.
func (s *ConfigSuite) TestTheProxyBrokerNeedsTheProxysURLAndCA() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "proxy")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	for _, address := range []string{"", "https://proxy.stg.example:8091", "http://sbx:secret@proxy.stg.example:8091", "http://:8091",
		"proxy.stg.example:8091", "http://203.0.113.7:8091/%zz", "http://sbx:secret@203.0.113.7:8091",
		"http://0.0.0.0:8091", "http://127.0.0.1:8091", "http://10.1.2.3:8091", "http://172.16.0.1:8091", "http://192.168.1.1:8091",
		"http://169.254.169.254:8091", "http://255.255.255.255:8091", "http://203.0.113.7", "http://203.0.113.7:0",
		"http://203.0.113.7:99999", "http://203.0.113.7:8091/", "http://203.0.113.7:8091/path", "http://203.0.113.7:8091?a=b",
		"http://203.0.113.7:8091#frag"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", address)
		_, _, err := Load("")
		s.ErrorContains(err, "code_sandbox.proxy_public_url", address)
	}

	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", "http://203.0.113.7:8091")
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(CodeSandbox{Broker: "proxy", ProxyAddr: ":8091", ProxyCACert: "cert", ProxyCAKey: "key",
		ProxyPublicURL: "http://203.0.113.7:8091", IdleStop: 15 * time.Minute, StoppedDelete: time.Hour}, config.CodeSandbox)

	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "")
	_, _, err = Load("")
	s.ErrorContains(err, "proxy_ca_key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", "")
	_, _, err = Load("")
	s.ErrorContains(err, "code_sandbox.proxy_addr")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "")
	_, _, err = Load("")
	s.ErrorContains(err, "proxy_ca_cert")
}

// TestTheProxyBrokersProxyIsAnIPv4Address: the sandbox may reach that address alone, so it
// has no DNS to look a name up with, and no IPv6.
func (s *ConfigSuite) TestTheProxyBrokersProxyIsAnIPv4Address() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "proxy")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	for _, address := range []string{"http://proxy.stg.example:8091", "http://[2001:db8::7]:8091", "http://[::ffff:203.0.113.7]:8091"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", address)
		_, _, err := Load("")
		s.ErrorContains(err, "code_sandbox.proxy_public_url (http://<public IPv4 address>:port or https://<public IPv4 address>:port", address)
	}

	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "daytona")
	_, _, err := Load("")
	s.NoError(err, "only the proxy broker sends a sandbox there")
}

// TestAnHTTPSProxyURLHasTheRulesOfAnHTTPOne: https is the proxy serving TLS, at a public IPv4
// address and a port, and with no user, path, query or fragment.
func (s *ConfigSuite) TestAnHTTPSProxyURLHasTheRulesOfAnHTTPOne() {
	for _, address := range []string{"https://proxy.stg.example:8091", "https://sbx:secret@203.0.113.7:8091",
		"https://10.1.2.3:8091", "https://127.0.0.1:8091", "https://169.254.169.254:8091", "https://[2001:db8::7]:8091",
		"https://203.0.113.7", "https://203.0.113.7:0", "https://203.0.113.7:8091/", "https://203.0.113.7:8091?a=b",
		"https://203.0.113.7:8091#frag", "ftp://203.0.113.7:8091", "httpx://203.0.113.7:8091"} {
		_, ok := ProxyURL(address)
		s.False(ok, address)
	}

	public, ok := ProxyURL("https://203.0.113.7:8091")
	s.Require().True(ok)
	s.Equal("https", public.Scheme)
	s.Equal("203.0.113.7", public.Hostname())
}

// TestTheProxyBrokerTakesAnHTTPSProxyURL: its sandboxes reach the proxy through their TLS
// tunnel, and the router serves the proxy over TLS.
func (s *ConfigSuite) TestTheProxyBrokerTakesAnHTTPSProxyURL() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", "https://203.0.113.7:8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "proxy")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal(CodeSandbox{Broker: "proxy", ProxyAddr: ":8091", ProxyCACert: "cert", ProxyCAKey: "key",
		ProxyPublicURL: "https://203.0.113.7:8091", IdleStop: 15 * time.Minute, StoppedDelete: time.Hour}, config.CodeSandbox)
}

// TestAnHTTPSProxyURLTheListenerCannotServeIsRefused: the daytona broker with a proxy address
// and an https URL a sandbox cannot use would serve plain HTTP in silence. An http URL and no
// URL load as on base.
func (s *ConfigSuite) TestAnHTTPSProxyURLTheListenerCannotServeIsRefused() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "daytona")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", "https://proxy.example:8091")

	_, _, err := Load("")

	s.ErrorContains(err, "an https URL that is not https://<public IPv4 address>:port")
	for _, address := range []string{"http://proxy.example:8091", "", "https://203.0.113.7:8091"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", address)
		_, _, err = Load("")
		s.NoError(err, address)
	}
}

// TestAnHTTPSProxyURLWithoutAProxyAddressLoadsAsOnBase: with no listener the URL is never served,
// so base loads it.
func (s *ConfigSuite) TestAnHTTPSProxyURLWithoutAProxyAddressLoadsAsOnBase() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "daytona")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", "")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", "https://proxy.example:8091")

	_, _, err := Load("")

	s.NoError(err)
}

// TestTheHTTPSSchemeIsRefusedInAnyCase: a URL scheme is case-insensitive (RFC 3986 3.1).
func (s *ConfigSuite) TestTheHTTPSSchemeIsRefusedInAnyCase() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "daytona")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", "HTTPS://proxy.example:8091")

	_, _, err := Load("")

	s.ErrorContains(err, "an https URL that is not https://<public IPv4 address>:port")
}

// TestTheRefusalOfAnHTTPSProxyURLNamesNoCredential: the URL's user is a credential, also when
// the URL does not parse (a port that is no number), where the error names no host at all.
func (s *ConfigSuite) TestTheRefusalOfAnHTTPSProxyURLNamesNoCredential() {
	for address, host := range map[string]string{"https://sbx:canary-tok@proxy.example:8091": "proxy.example:8091",
		"https://sbx:canary-tok@proxy.example:abc": "unparseable"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_BROKER", "daytona")
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
		s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL", address)

		_, _, err := Load("")

		s.Require().ErrorContains(err, `has host "`+host+`"`, address)
		s.NotContains(err.Error(), "canary-tok", address)
		s.NotContains(err.Error(), "sbx", address)
		s.NotContains(err.Error(), "proxy.example:abc", address)
	}
}

// TestTheProxysCACertificateHoldsNoKey: every sandbox is given it.
func (s *ConfigSuite) TestTheProxysCACertificateHoldsNoKey() {
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "-----BEGIN EC PRIVATE KEY-----\nkey\n-----END EC PRIVATE KEY-----\n")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "-----BEGIN CERTIFICATE-----\ncert\n-----END CERTIFICATE-----\n"+
		"-----BEGIN EC PRIVATE KEY-----\nkey\n-----END EC PRIVATE KEY-----\n")

	_, _, err := Load("")

	s.ErrorContains(err, "code_sandbox.proxy_ca_cert (ROUTER_CODE_SANDBOX_PROXY_CA_CERT) holds a private key")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "-----BEGIN CERTIFICATE-----\ncert\n-----END CERTIFICATE-----\n")
	_, _, err = Load("")
	s.NoError(err)
}

// TestTheProxysPublicURLIsReadableFromTheEnvironmentAfterwards, like every other setting.
func (s *ConfigSuite) TestTheProxysPublicURLIsReadableFromTheEnvironmentAfterwards() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("code_sandbox:\n  proxy_public_url: http://proxy.stg.example:8091\n"), 0o600))

	_, _, err := Load(path)
	s.Require().NoError(err)

	s.Equal("http://proxy.stg.example:8091", os.Getenv("ROUTER_CODE_SANDBOX_PROXY_PUBLIC_URL"))
}

// TestTheSandboxProxyIsOffUnlessAskedForAndThenNeedsItsCA: no address serves no proxy, and
// an address without the CA's certificate and key does not start.
func (s *ConfigSuite) TestTheSandboxProxyIsOffUnlessAskedForAndThenNeedsItsCA() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(config.CodeSandbox.ProxyAddr)

	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_ADDR", ":8091")
	_, _, err = Load("")
	s.ErrorContains(err, "code_sandbox.proxy_ca_cert")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	_, _, err = Load("")
	s.ErrorContains(err, "code_sandbox.proxy_ca_cert")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "")
	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_KEY", "key")
	_, _, err = Load("")
	s.ErrorContains(err, "code_sandbox.proxy_ca_cert")

	s.T().Setenv("ROUTER_CODE_SANDBOX_PROXY_CA_CERT", "cert")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(CodeSandbox{Broker: "daytona", ProxyAddr: ":8091", ProxyCACert: "cert", ProxyCAKey: "key",
		IdleStop: 15 * time.Minute, StoppedDelete: time.Hour}, config.CodeSandbox)
}

// TestTheSecretPrefixHoldsNoUnderscore: a router named stg_dev would be swept by one named stg.
func (s *ConfigSuite) TestTheSecretPrefixHoldsNoUnderscore() {
	for _, prefix := range []string{"stg_dev", "Stg", "stg-dev", "stg.dev"} {
		s.T().Setenv("ROUTER_CODE_SANDBOX_SECRET_PREFIX", prefix)
		_, _, err := Load("")
		s.ErrorContains(err, "code_sandbox.secret_prefix", prefix)
	}
	s.T().Setenv("ROUTER_CODE_SANDBOX_SECRET_PREFIX", "stg2")
	_, _, err := Load("")
	s.NoError(err)
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

func (s *ConfigSuite) TestTheManageKeyIsNeverTheOpsKey() {
	s.T().Setenv("ROUTER_AUTH_OPS_KEY", "one-staff-key")
	s.T().Setenv("ROUTER_AUTH_MANAGE_KEY", "one-staff-key")

	_, _, err := Load("")

	s.Require().ErrorContains(err, "auth.manage_key")
	s.ErrorContains(err, "auth.ops_key")
	s.NotContains(err.Error(), "one-staff-key", "the error names the settings, never the key")
}

func (s *ConfigSuite) TestAManageKeyOfItsOwnIsTaken() {
	s.T().Setenv("ROUTER_AUTH_OPS_KEY", "the-ops-key")
	s.T().Setenv("ROUTER_AUTH_MANAGE_KEY", "the-manage-key")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal("the-manage-key", config.Auth.ManageKey)
}

func (s *ConfigSuite) TestTheDeliveryKeysAreACommaSeparatedListAndExported() {
	s.T().Setenv("ROUTER_STREAM_DELIVERY_KEYS", "current-key=, previous-key=,")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal([]string{"current-key=", "previous-key="}, config.Stream.DeliveryKeys)
	s.Equal("current-key=,previous-key=", os.Getenv("ROUTER_STREAM_DELIVERY_KEYS"))
}

func (s *ConfigSuite) TestTheDeliveryKeysAreASequenceInAFile() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("stream:\n  delivery_keys: [current-key=, previous-key=]\n"), 0o600))

	config, _, err := Load(path)

	s.Require().NoError(err)
	s.Equal([]string{"current-key=", "previous-key="}, config.Stream.DeliveryKeys)
	s.Equal("current-key=,previous-key=", os.Getenv("ROUTER_STREAM_DELIVERY_KEYS"), "what a file sets reaches the environment")
}

func (s *ConfigSuite) TestNoDeliveryKeysIsTheDefault() {
	config, _, err := Load("")

	s.Require().NoError(err)
	s.Nil(config.Stream.DeliveryKeys)
}

func (s *ConfigSuite) TestAUsageServiceWithoutItsTokenIsRefused() {
	s.T().Setenv("ROUTER_USAGE_URL", "http://usage:8070")

	_, _, err := Load("")

	s.Require().ErrorContains(err, "usage.url")
	s.ErrorContains(err, "ROUTER_USAGE_TOKEN")
}

func (s *ConfigSuite) TestAUsageURLThatNamesNoServiceIsRefused() {
	s.T().Setenv("ROUTER_USAGE_TOKEN", "token")
	for _, address := range []string{"usage:8070", "http://", "ftp://usage:8070"} {
		s.T().Setenv("ROUTER_USAGE_URL", address)

		_, _, err := Load("")

		s.ErrorContains(err, "usage.url", address)
	}
}

func (s *ConfigSuite) TestAUsageTokenWithoutTheServiceIsAllowed() {
	s.T().Setenv("ROUTER_USAGE_TOKEN", "the-ingest-token")

	settings, _, err := Load("")

	s.Require().NoError(err, "unsetting the URL alone stops shipping")
	s.Empty(settings.Usage.URL)
}

func (s *ConfigSuite) TestTheUsageServicesRedisNeedsTheService() {
	s.T().Setenv("ROUTER_USAGE_REDIS_URL", "redis://redis:6379")

	_, _, err := Load("")

	s.Require().ErrorContains(err, "ROUTER_USAGE_REDIS_URL")
	s.ErrorContains(err, "ROUTER_USAGE_URL")

	s.T().Setenv("ROUTER_USAGE_URL", "http://usage:8070")
	s.T().Setenv("ROUTER_USAGE_TOKEN", "token")
	settings, _, err := Load("")
	s.Require().NoError(err)
	s.Equal("redis://redis:6379", settings.Usage.RedisURL)
}

func (s *ConfigSuite) TestTheUsageServiceIsReadAndExported() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("usage:\n  url: http://usage:8070\n  token: from-the-file\n"), 0o600))
	s.T().Setenv("ROUTER_USAGE_TOKEN", "from-the-environment")

	config, _, err := Load(path)

	s.Require().NoError(err)
	s.Equal(Usage{URL: "http://usage:8070", Token: "from-the-environment"}, config.Usage)
	s.Equal("http://usage:8070", os.Getenv("ROUTER_USAGE_URL"), "what a file sets reaches the environment")
}

func (s *ConfigSuite) TestTheUsageManageKeyIsReadAndExported() {
	s.T().Setenv("ROUTER_USAGE_URL", "http://usage:8070")
	s.T().Setenv("ROUTER_USAGE_TOKEN", "token")
	s.T().Setenv("ROUTER_USAGE_MANAGE_KEY", "the-manage-key")

	config, _, err := Load("")

	s.Require().NoError(err)
	s.Equal("the-manage-key", config.Usage.ManageKey)
	s.Equal("the-manage-key", os.Getenv("ROUTER_USAGE_MANAGE_KEY"))
}

func (s *ConfigSuite) TestNoUsageServiceIsTheDefault() {
	config, _, err := Load("")

	s.Require().NoError(err)
	s.Empty(config.Usage.URL)
}

func (s *ConfigSuite) TestLimitsModeMustBeKnown() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Equal(LimitsEnforce, config.Limits.Mode, "a deployment that says nothing enforces")

	for _, mode := range []LimitsMode{LimitsEnforce, LimitsShadow, LimitsOff} {
		s.T().Setenv("ROUTER_LIMITS_MODE", string(mode))
		config, _, err := Load("")
		s.Require().NoError(err, mode)
		s.Equal(mode, config.Limits.Mode)
		s.Equal(string(mode), os.Getenv("ROUTER_LIMITS_MODE"), "exported")
	}

	for _, mode := range []string{"Enforce", "disabled", "true"} {
		s.T().Setenv("ROUTER_LIMITS_MODE", mode)
		_, _, err := Load("")
		s.Require().ErrorContains(err, "ROUTER_LIMITS_MODE", mode)
		s.ErrorContains(err, "enforce, shadow or off")
	}
}

func (s *ConfigSuite) TestTheKMSKeyIsReadFromTheEnvironment() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(config.Auth.KMSKey, "no KMS key unless one is named")

	s.T().Setenv("ROUTER_AUTH_KMS_KEY", "projects/p/locations/l/keyRings/r/cryptoKeys/k")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal("projects/p/locations/l/keyRings/r/cryptoKeys/k", config.Auth.KMSKey)
}

func (s *ConfigSuite) TestAKMSKeyFromAFileIsExported() {
	path := filepath.Join(s.T().TempDir(), "router.yaml")
	s.Require().NoError(os.WriteFile(path, []byte("auth:\n  kms_key: projects/p/locations/l/keyRings/r/cryptoKeys/k\n"), 0o600))

	_, _, err := Load(path)
	s.Require().NoError(err)
	s.Equal("projects/p/locations/l/keyRings/r/cryptoKeys/k", os.Getenv("ROUTER_AUTH_KMS_KEY"))
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

func (s *ConfigSuite) TestMetricsAreNotServedUnlessAnAddressIsSet() {
	config, _, err := Load("")
	s.Require().NoError(err)
	s.Empty(config.MetricsAddr)

	s.T().Setenv("ROUTER_METRICS_ADDR", ":9091")
	config, _, err = Load("")
	s.Require().NoError(err)
	s.Equal(":9091", config.MetricsAddr)
}
