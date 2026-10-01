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

var nonPublicPrefixes = mustParsePrefixes([]string{
	"0.0.0.0/8",
	"10.0.0.0/8",
	"100.64.0.0/10",
	"127.0.0.0/8",
	"169.254.0.0/16",
	"172.16.0.0/12",
	"192.0.0.0/24",
	"192.0.2.0/24",
	"192.88.99.0/24",
	"192.168.0.0/16",
	"198.18.0.0/15",
	"198.51.100.0/24",
	"203.0.113.0/24",
	"224.0.0.0/4",
	"240.0.0.0/4",
	"::/128",
	"::1/128",
	"64:ff9b::/96",
	"64:ff9b:1::/48",
	"100::/64",
	"2001::/23",
	"2001:db8::/32",
	"2002::/16",
	"3fff::/20",
	"fc00::/7",
	"fe80::/10",
	"ff00::/8",
})

// ipv6GlobalUnicast is the only IPv6 space IANA allocates for global unicast. Outside it
// sit IPv4-compatible (::/96) and IPv4-translated (::ffff:0:0:0/96) forms of local hosts.
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
