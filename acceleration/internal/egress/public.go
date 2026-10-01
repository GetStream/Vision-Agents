package egress

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
)

// nonPublicPrefixes are the address blocks no connector endpoint may resolve to. Each one
// is a block IANA's special-purpose registries mark as not globally reachable, or one whose
// address carries an IPv4 address inside it that could be private:
// https://www.iana.org/assignments/iana-ipv4-special-registry and
// https://www.iana.org/assignments/iana-ipv6-special-registry. The /24 and /23 of IETF
// protocol assignments are refused whole, including the few anycast services inside them
// that are reachable, since no connector endpoint lives there. Some entries repeat what
// netip's IsPrivate and IsLoopback or the 2000::/3 check below already refuse; they stay so
// the list can be read against the registries line by line.
var nonPublicPrefixes = mustParsePrefixes([]string{
	"0.0.0.0/8",       // "This network", RFC 791 section 3.2
	"10.0.0.0/8",      // Private-Use, RFC 1918
	"100.64.0.0/10",   // Shared Address Space (carrier-grade NAT), RFC 6598
	"127.0.0.0/8",     // Loopback, RFC 1122 section 3.2.1.3
	"169.254.0.0/16",  // Link Local, RFC 3927; the cloud metadata server answers on 169.254.169.254
	"172.16.0.0/12",   // Private-Use, RFC 1918
	"192.0.0.0/24",    // IETF Protocol Assignments, RFC 6890 section 2.1
	"192.0.2.0/24",    // Documentation (TEST-NET-1), RFC 5737
	"192.88.99.0/24",  // 6to4 Relay Anycast, deprecated by RFC 7526
	"192.168.0.0/16",  // Private-Use, RFC 1918
	"198.18.0.0/15",   // Benchmarking, RFC 2544
	"198.51.100.0/24", // Documentation (TEST-NET-2), RFC 5737
	"203.0.113.0/24",  // Documentation (TEST-NET-3), RFC 5737
	"224.0.0.0/4",     // Multicast, RFC 5771
	"240.0.0.0/4",     // Reserved, RFC 1112 section 4; holds limited broadcast 255.255.255.255
	"::/128",          // Unspecified Address, RFC 4291
	"::1/128",         // Loopback Address, RFC 4291
	"64:ff9b::/96",    // IPv4-IPv6 translation, RFC 6052; the low 32 bits are an IPv4 address
	"64:ff9b:1::/48",  // Local-use IPv4-IPv6 translation, RFC 8215
	"100::/64",        // Discard-Only Address Block, RFC 6666
	"2001::/23",       // IETF Protocol Assignments, RFC 2928; holds Teredo 2001::/32, RFC 4380, which wraps an IPv4 address
	"2001:db8::/32",   // Documentation, RFC 3849
	"2002::/16",       // 6to4, RFC 3056; bits 16 to 47 are an IPv4 address
	"3fff::/20",       // Documentation, RFC 9637
	"fc00::/7",        // Unique-Local, RFC 4193
	"fe80::/10",       // Link-Local Unicast, RFC 4291
	"ff00::/8",        // Multicast, RFC 4291 section 2.7
})

// ipv6GlobalUnicast is the only IPv6 space IANA allocates for global unicast (RFC 4291,
// https://www.iana.org/assignments/ipv6-address-space). Outside it sit IPv4-compatible
// (::/96) and IPv4-translated (::ffff:0:0:0/96) forms of local hosts.
var ipv6GlobalUnicast = netip.MustParsePrefix("2000::/3")

var errCrossHostRedirect = errors.New("egress: redirect to another host refused")

// ValidatePublicHTTPSURL checks a tenant-provided remote service address before it is
// stored. The dialer repeats address checks when opening each outbound connection.
func ValidatePublicHTTPSURL(ctx context.Context, raw string) error {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme != "https" || parsed.Hostname() == "" || parsed.User != nil ||
		parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return fmt.Errorf("egress: endpoint must be a public HTTPS URL without userinfo, query or fragment")
	}
	return publicAddresses(ctx, parsed.Hostname())
}

// NewPublicHTTPClient creates an HTTP client that only dials globally routable IPs and
// follows redirects only within the same scheme, host and port.
// It does not use environment proxies, because a proxy would bypass local address checks.
func NewPublicHTTPClient(timeout time.Duration) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DialContext = dialPublic
	return &http.Client{
		Transport: publicRoundTripper{transport: transport},
		Timeout:   timeout,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if request.URL.Scheme != via[0].URL.Scheme || request.URL.Host != via[0].URL.Host {
				return errCrossHostRedirect
			}
			if len(via) >= 10 {
				return errors.New("egress: stopped after 10 redirects")
			}
			return nil
		},
	}
}

type publicRoundTripper struct {
	transport *http.Transport
}

func (t publicRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL.Scheme != "https" || request.URL.User != nil {
		return nil, fmt.Errorf("egress: only HTTPS requests without userinfo are allowed")
	}
	return t.transport.RoundTrip(request)
}

func dialPublic(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, fmt.Errorf("egress: invalid outbound address")
	}
	ips, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
	if err != nil {
		return nil, fmt.Errorf("egress: resolve outbound host: %w", err)
	}
	if len(ips) == 0 {
		return nil, fmt.Errorf("egress: outbound host has no address")
	}
	for _, ip := range ips {
		if !isPublic(ip) {
			return nil, fmt.Errorf("egress: outbound host resolved to a non-public address")
		}
	}
	dialer := &net.Dialer{Timeout: 8 * time.Second, KeepAlive: 30 * time.Second}
	var lastErr error
	for _, ip := range ips {
		connection, err := dialer.DialContext(ctx, network, net.JoinHostPort(ip.String(), port))
		if err == nil {
			return connection, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("egress: connect to public endpoint: %w", lastErr)
}

func publicAddresses(ctx context.Context, host string) error {
	if ip, err := netip.ParseAddr(host); err == nil {
		if !isPublic(ip) {
			return fmt.Errorf("egress: endpoint host is not publicly routable")
		}
		return nil
	}
	if strings.Contains(host, "%") {
		return fmt.Errorf("egress: scoped IP addresses are not allowed")
	}
	ips, err := net.DefaultResolver.LookupNetIP(ctx, "ip", host)
	if err != nil || len(ips) == 0 {
		return fmt.Errorf("egress: endpoint host could not be resolved")
	}
	for _, ip := range ips {
		if !isPublic(ip) {
			return fmt.Errorf("egress: endpoint host resolved to a non-public address")
		}
	}
	return nil
}

func isPublic(ip netip.Addr) bool {
	// A zoned address never matches a netip.Prefix, so it would skip the list below.
	if !ip.IsValid() || ip.Zone() != "" {
		return false
	}
	ip = ip.Unmap()
	if ip.Is6() && !ipv6GlobalUnicast.Contains(ip) {
		return false
	}
	if !ip.IsGlobalUnicast() || ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsUnspecified() {
		return false
	}
	for _, prefix := range nonPublicPrefixes {
		if prefix.Contains(ip) {
			return false
		}
	}
	return true
}

func mustParsePrefixes(values []string) []netip.Prefix {
	prefixes := make([]netip.Prefix, 0, len(values))
	for _, value := range values {
		prefixes = append(prefixes, netip.MustParsePrefix(value))
	}
	return prefixes
}
