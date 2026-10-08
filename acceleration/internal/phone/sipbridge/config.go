package sipbridge

import (
	"errors"
	"fmt"
	"log/slog"
	"net"
	"time"

	"github.com/emiago/sipgo/sip"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Config is everything one outbound call needs. Dial takes it as a value, so a caller fills
// it from wherever its trunks are kept.
type Config struct {
	CustomerTrunk CustomerTrunk
	Stream        StreamTrunk
	Call          CallParams
	FlowB         FlowB
	// KeepaliveInterval is how often each leg gets an in-dialog OPTIONS, so its TCP
	// connection stays open through NAT and the other side's BYE can still reach us.
	KeepaliveInterval time.Duration
	Logger            *slog.Logger
	// guard checks where the customer leg's requests go. Its zero value allows only public
	// addresses; tests on loopback set their own.
	guard guard
}

// CustomerTrunk is the customer's own SIP trunk, which we dial.
type CustomerTrunk struct {
	Host      string
	Port      int
	Transport string
	Username  string
	Password  string
	// LateOffer says the trunk accepts an INVITE without SDP. It picks flow A.
	LateOffer bool
	Codecs    []string
}

// StreamTrunk is the inbound trunk Stream created for this call.
type StreamTrunk struct {
	// URI is Bridge.URI from CreateTrunk, the Stream leg's Request-URI.
	URI       string
	Username  string
	Password  string
	Transport string
}

// CallParams says who calls whom.
type CallParams struct {
	// From is the number the Stream trunk and its routing rule are set up for. Stream
	// matches the rule on it, so it goes in the Stream leg's To.
	From string
	// To is who to ring.
	To          string
	RingTimeout time.Duration
}

// FlowB tunes the flow B offer that opens the session in Stream before the customer answers.
type FlowB struct {
	// PlaceholderAddr is the address in that first SDP. See placeholderOffer.
	PlaceholderAddr string
	PlaceholderPort int
}

// WithDefaults returns c with every unset optional field filled in.
func (c Config) WithDefaults() Config {
	if c.CustomerTrunk.Port == 0 {
		c.CustomerTrunk.Port = 5060
	}
	if c.CustomerTrunk.Transport == "" {
		c.CustomerTrunk.Transport = "tcp"
	}
	if len(c.CustomerTrunk.Codecs) == 0 {
		c.CustomerTrunk.Codecs = []string{"PCMU", "PCMA"}
	}
	if c.Stream.Transport == "" {
		c.Stream.Transport = "tcp"
	}
	if c.Call.RingTimeout == 0 {
		c.Call.RingTimeout = 30 * time.Second
	}
	if c.KeepaliveInterval == 0 {
		c.KeepaliveInterval = 20 * time.Second
	}
	if c.FlowB.PlaceholderAddr == "" {
		c.FlowB.PlaceholderAddr = defaultPlaceholderAddr
	}
	if c.FlowB.PlaceholderPort == 0 {
		c.FlowB.PlaceholderPort = defaultPlaceholderPort
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
	return c
}

// Validate reports every problem at once.
func (c Config) Validate() error {
	var errs []error
	required := func(value, field string) {
		if value == "" {
			errs = append(errs, fmt.Errorf("%s is required", field))
		}
	}
	required(c.CustomerTrunk.Host, "customer_trunk.host")
	required(c.Stream.URI, "stream.uri")
	required(c.Call.From, "call.from")
	required(c.Call.To, "call.to")

	if !validTransport(c.CustomerTrunk.Transport) {
		errs = append(errs, fmt.Errorf("customer_trunk.transport %q must be udp, tcp or tls", c.CustomerTrunk.Transport))
	}
	if !validTransport(c.Stream.Transport) {
		errs = append(errs, fmt.Errorf("stream.transport %q must be udp, tcp or tls", c.Stream.Transport))
	}
	if _, err := codecsFromNames(c.CustomerTrunk.Codecs); err != nil {
		errs = append(errs, fmt.Errorf("customer_trunk.codecs: %w", err))
	}
	if c.Stream.URI != "" {
		if _, err := c.streamURI(); err != nil {
			errs = append(errs, fmt.Errorf("stream.uri: %w", err))
		}
	}
	if c.Call.RingTimeout <= 0 {
		errs = append(errs, errors.New("call.ring_timeout must be positive"))
	}
	if c.KeepaliveInterval <= 0 {
		errs = append(errs, errors.New("keepalive_interval must be positive"))
	}
	if ip := net.ParseIP(c.FlowB.PlaceholderAddr); ip == nil || ip.To4() == nil {
		errs = append(errs, fmt.Errorf("flow_b.placeholder_addr %q must be an IPv4 address", c.FlowB.PlaceholderAddr))
	}
	if c.FlowB.PlaceholderPort < 1 || c.FlowB.PlaceholderPort > 65535 {
		errs = append(errs, fmt.Errorf("flow_b.placeholder_port %d must be 1..65535", c.FlowB.PlaceholderPort))
	}
	return stack.Wrap(errors.Join(errs...))
}

func validTransport(t string) bool {
	return t == "udp" || t == "tcp" || t == "tls"
}

func (c Config) customerURI() (sip.Uri, error) {
	var uri sip.Uri
	raw := fmt.Sprintf("sip:%s@%s:%d;transport=%s",
		c.Call.To, c.CustomerTrunk.Host, c.CustomerTrunk.Port, c.CustomerTrunk.Transport)
	err := sip.ParseUri(raw, &uri)
	return uri, stack.Wrap(err)
}

func (c Config) streamURI() (sip.Uri, error) {
	var uri sip.Uri
	if err := sip.ParseUri(c.Stream.URI, &uri); err != nil {
		return uri, stack.Wrap(err)
	}
	if uri.UriParams == nil {
		uri.UriParams = sip.NewParams()
	}
	if !uri.UriParams.Has("transport") {
		uri.UriParams.Add("transport", c.Stream.Transport)
	}
	return uri, nil
}
