package egress

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
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

// maxRedirects is net/http's own limit (defaultCheckRedirect in net/http/client.go). A
// custom CheckRedirect replaces that check, so it is repeated here.
const maxRedirects = 10

// httpsPort is what an https URL without a port connects to (RFC 9110 section 4.2.2).
const httpsPort = 443

var (
	errCrossHostRedirect = errors.New("egress: redirect to another host refused")
	errMethodRedirect    = errors.New("egress: redirect that changes the request method refused")
)

// lookup is the part of *net.Resolver the policy uses, so a test can answer a name one
// way when it is validated and another way when it is dialed.
type lookup interface {
	LookupNetIP(ctx context.Context, network, host string) ([]netip.Addr, error)
}

// policy decides where connector traffic may go. Its fields exist for tests; callers
// outside the package always get defaultPolicy.
type policy struct {
	resolver lookup
	// dial opens the socket to an address the policy has already checked.
	dial      func(ctx context.Context, network, address string) (net.Conn, error)
	tlsConfig *tls.Config
}

// defaultPolicy dials with the settings http.DefaultTransport uses (30s connect, 30s
// keep-alive, net/http/transport.go); the client timeout bounds the whole call.
var defaultPolicy = policy{
	resolver: net.DefaultResolver,
	dial:     (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
}

// ValidatePublicHTTPSURL checks a tenant-provided remote service address before it is
// stored. The client repeats the address check when opening each outbound connection.
func ValidatePublicHTTPSURL(ctx context.Context, raw string) error {
	return defaultPolicy.validate(ctx, raw)
}

// NewClient is how connector traffic leaves the router. wrap adds what a scheme puts on
// each request (Scheme.Wrap) and may be nil. It sits between the redirect policy and the
// address checks, so it can remove neither: a redirect never leaves the origin, so the
// credential wrap adds only reaches the host it was meant for, and every connection,
// redirect hops included, dials only a checked public IP. Replacing the returned client's
// Transport removes those checks, so callers must not. Environment proxies are not used,
// because a proxy would bypass the address checks.
func NewClient(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
	return defaultPolicy.client(timeout, wrap)
}

func (p policy) validate(ctx context.Context, raw string) error {
	parsed, err := url.Parse(raw)
	if err != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return stack.Wrap(fmt.Errorf("egress: endpoint must be a public HTTPS URL without userinfo, query or fragment"))
	}
	if err := checkURL(parsed); err != nil {
		return err
	}
	return p.publicAddresses(ctx, parsed.Hostname())
}

func (p policy) client(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DialContext = p.dialPublic
	if p.tlsConfig != nil {
		transport.TLSClientConfig = p.tlsConfig
	}
	var next http.RoundTripper = checkedRoundTripper{transport: transport}
	if wrap != nil {
		next = wrap(next)
	}
	return &http.Client{
		Transport:     clientRoundTripper{next: next, transport: transport},
		Timeout:       timeout,
		CheckRedirect: checkRedirect,
	}
}

// checkRedirect follows a redirect only within the origin, so a credential added to each
// request never reaches another host, and only when the method survives it: net/http
// turns a POST into a GET on 301, 302 and 303 and drops the body.
func checkRedirect(request *http.Request, via []*http.Request) error {
	if !sameOrigin(request.URL, via[0].URL) {
		return errCrossHostRedirect
	}
	if request.Method != via[len(via)-1].Method {
		return errMethodRedirect
	}
	if len(via) >= maxRedirects {
		return fmt.Errorf("egress: stopped after %d redirects", maxRedirects)
	}
	return nil
}

func sameOrigin(a, b *url.URL) bool {
	portA, errA := port(a)
	portB, errB := port(b)
	return errA == nil && errB == nil && portA == portB &&
		strings.EqualFold(a.Scheme, b.Scheme) && sameHost(a.Hostname(), b.Hostname())
}

func sameHost(a, b string) bool {
	ipA, errA := netip.ParseAddr(a)
	ipB, errB := netip.ParseAddr(b)
	if errA == nil && errB == nil {
		return ipA.Unmap() == ipB.Unmap()
	}
	return strings.EqualFold(a, b)
}

// checkURL is what every outbound URL must satisfy, at validation and on each request.
func checkURL(target *url.URL) error {
	host := target.Hostname()
	if target.Scheme != "https" || host == "" || target.User != nil {
		return stack.Wrap(fmt.Errorf("egress: only HTTPS URLs with a host and without userinfo are allowed"))
	}
	if _, err := port(target); err != nil {
		return err
	}
	if _, err := netip.ParseAddr(host); err != nil && numericHost(host) {
		return stack.Wrap(fmt.Errorf("egress: host is neither an IP address nor a domain name"))
	}
	return nil
}

func port(target *url.URL) (int, error) {
	raw := target.Port()
	if raw == "" {
		return httpsPort, nil
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 1 || value > 65535 {
		return 0, stack.Wrap(fmt.Errorf("egress: port must be between 1 and 65535"))
	}
	return value, nil
}

// numericHost reports a host whose last label is a number, such as 2130706433, 127.1,
// 0x7f.0.0.1 or 0x7f000001. No domain name ends that way (RFC 3696 section 2: a top-level
// domain is not all-numeric), but some resolvers read these as IPv4 addresses, so they are
// refused rather than resolved.
func numericHost(host string) bool {
	label := strings.TrimSuffix(host, ".")
	if i := strings.LastIndexByte(label, '.'); i >= 0 {
		label = label[i+1:]
	}
	digits := "0123456789"
	if len(label) > 2 && strings.EqualFold(label[:2], "0x") {
		label, digits = label[2:], "0123456789abcdefABCDEF"
	}
	return label != "" && strings.Trim(label, digits) == ""
}

// clientRoundTripper is the client's transport. It checks the URL before the scheme's
// wrapper sees the request, so no credential is applied to a URL that will be refused,
// and forwards CloseIdleConnections to the inner transport, which the wrapper would
// otherwise hide from Client.
type clientRoundTripper struct {
	next      http.RoundTripper
	transport *http.Transport
}

func (t clientRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	if err := checkURL(request.URL); err != nil {
		return nil, err
	}
	return t.next.RoundTrip(request)
}

func (t clientRoundTripper) CloseIdleConnections() {
	t.transport.CloseIdleConnections()
}

// checkedRoundTripper repeats the URL check after the wrapper, which may rewrite the request.
type checkedRoundTripper struct {
	transport *http.Transport
}

func (t checkedRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	if err := checkURL(request.URL); err != nil {
		return nil, err
	}
	return t.transport.RoundTrip(request)
}

func (p policy) dialPublic(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, fmt.Errorf("egress: invalid outbound address")
	}
	ips, err := p.resolve(ctx, host)
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
	var lastErr error
	for _, ip := range ips {
		connection, err := p.dial(ctx, network, net.JoinHostPort(ip.String(), port))
		if err == nil {
			return connection, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("egress: connect to public endpoint: %w", lastErr)
}

func (p policy) resolve(ctx context.Context, host string) ([]netip.Addr, error) {
	if ip, err := netip.ParseAddr(host); err == nil {
		return []netip.Addr{ip}, nil
	}
	return p.resolver.LookupNetIP(ctx, "ip", host)
}

func (p policy) publicAddresses(ctx context.Context, host string) error {
	if ip, err := netip.ParseAddr(host); err == nil {
		if !isPublic(ip) {
			return stack.Wrap(fmt.Errorf("egress: endpoint host is not publicly routable"))
		}
		return nil
	}
	if strings.Contains(host, "%") {
		return stack.Wrap(fmt.Errorf("egress: scoped IP addresses are not allowed"))
	}
	ips, err := p.resolver.LookupNetIP(ctx, "ip", host)
	if err != nil || len(ips) == 0 {
		return stack.Wrap(fmt.Errorf("egress: endpoint host could not be resolved"))
	}
	for _, ip := range ips {
		if !isPublic(ip) {
			return stack.Wrap(fmt.Errorf("egress: endpoint host resolved to a non-public address"))
		}
	}
	return nil
}

// IsPublic says whether traffic to a tenant-supplied host may go to ip. It is the check
// NewClient makes at dial, for callers that open their own connections.
func IsPublic(ip netip.Addr) bool { return isPublic(ip) }

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
