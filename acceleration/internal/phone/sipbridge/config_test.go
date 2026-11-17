package sipbridge

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func validConfig() Config {
	return Config{
		CustomerTrunk: CustomerTrunk{Host: "sip.carrier.test", Username: "alice", Password: "secret"},
		Stream:        StreamTrunk{URI: "sip:trunk-1@bridge.sip.example.com", Username: "u", Password: "p"},
		Call:          CallParams{From: "+15550002222", To: "+15550001111"},
	}.WithDefaults()
}

func TestDefaultsFillWhatWasLeftOut(t *testing.T) {
	cfg := validConfig()

	require.Equal(t, 5060, cfg.CustomerTrunk.Port)
	require.Equal(t, "tcp", cfg.CustomerTrunk.Transport)
	require.Equal(t, "tcp", cfg.Stream.Transport)
	require.Equal(t, []string{"PCMU", "PCMA"}, cfg.CustomerTrunk.Codecs)
	require.Equal(t, 30*time.Second, cfg.Call.RingTimeout)
	require.Equal(t, 20*time.Second, cfg.KeepaliveInterval)
	require.NotNil(t, cfg.Logger)
	require.Equal(t, "192.0.2.1", cfg.FlowB.PlaceholderAddr)
	require.Equal(t, 20000, cfg.FlowB.PlaceholderPort)
	require.NoError(t, cfg.Validate())
}

func TestValidateRefusesABadFlowBPlaceholder(t *testing.T) {
	for name, mutate := range map[string]func(*Config){
		"not an ip": func(c *Config) { c.FlowB.PlaceholderAddr = "not-an-ip" },
		"ipv6":      func(c *Config) { c.FlowB.PlaceholderAddr = "2001:db8::1" },
		"port 0":    func(c *Config) { c.FlowB.PlaceholderPort = 0 },
		"port high": func(c *Config) { c.FlowB.PlaceholderPort = 65536 },
	} {
		t.Run(name, func(t *testing.T) {
			cfg := validConfig()
			mutate(&cfg)
			err := cfg.Validate()
			require.Error(t, err)
			require.ErrorContains(t, err, "flow_b.placeholder_")
		})
	}
}

func TestValidateAllowsAnUnspecifiedPlaceholderAddress(t *testing.T) {
	cfg := validConfig()
	cfg.FlowB.PlaceholderAddr = "0.0.0.0"
	require.NoError(t, cfg.Validate())
}

func TestValidateNamesEveryMissingField(t *testing.T) {
	err := Config{}.WithDefaults().Validate()

	require.Error(t, err)
	for _, field := range []string{"customer_trunk.host", "stream.uri", "call.from", "call.to"} {
		require.ErrorContains(t, err, field)
	}
}

func TestValidateRefusesAnUnknownCodec(t *testing.T) {
	cfg := validConfig()
	cfg.CustomerTrunk.Codecs = []string{"PCMU", "OPUS"}

	require.ErrorContains(t, cfg.Validate(), `unsupported codec "OPUS"`)
}

func TestValidateRefusesAnUnknownTransport(t *testing.T) {
	cfg := validConfig()
	cfg.CustomerTrunk.Transport = "ws"

	require.ErrorContains(t, cfg.Validate(), "customer_trunk.transport")
}

func TestCustomerURIDialsTheCalledNumberOnTheTrunk(t *testing.T) {
	uri, err := validConfig().customerURI()

	require.NoError(t, err)
	require.Equal(t, "sip:+15550001111@sip.carrier.test:5060;transport=tcp", uri.String())
}

func TestStreamURIKeepsTheTrunkAddressAndAddsTransport(t *testing.T) {
	uri, err := validConfig().streamURI()

	require.NoError(t, err)
	require.Equal(t, "sip:trunk-1@bridge.sip.example.com;transport=tcp", uri.String())
}

func TestStreamURIKeepsATransportAlreadyGiven(t *testing.T) {
	cfg := validConfig()
	cfg.Stream.URI = "sip:trunk-1@bridge.sip.example.com;transport=udp"

	uri, err := cfg.streamURI()

	require.NoError(t, err)
	require.Equal(t, "sip:trunk-1@bridge.sip.example.com;transport=udp", uri.String())
}
