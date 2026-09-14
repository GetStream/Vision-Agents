package api

import (
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"strings"
)

// ForwardedHeader is where a proxy records who it forwarded a request for.
const ForwardedHeader = "X-Forwarded-For"

// TrustedProxies parses the CIDR ranges a deployment's own proxies sit in.
//
// It is a list of ranges rather than a count of hops because a count is silently wrong the
// day somebody puts another load balancer in front, and a range keeps being right.
func TrustedProxies(ranges []string) ([]netip.Prefix, error) {
	var trusted []netip.Prefix
	for _, entry := range ranges {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		// A bare address is the range holding only itself, so a deployment with one proxy
		// at a known address need not work out what to write after the slash.
		if !strings.Contains(entry, "/") {
			address, err := netip.ParseAddr(entry)
			if err != nil {
				return nil, fmt.Errorf("api: %q is not an address or a CIDR range: %w", entry, err)
			}
			trusted = append(trusted, netip.PrefixFrom(address, address.BitLen()))
			continue
		}
		prefix, err := netip.ParsePrefix(entry)
		if err != nil {
			return nil, fmt.Errorf("api: %q is not a CIDR range: %w", entry, err)
		}
		trusted = append(trusted, prefix.Masked())
	}
	return trusted, nil
}

// clientIP is where a request came from, as far as the deployment's own proxies can vouch
// for.
//
// X-Forwarded-For is a list each hop appends to, so the rightmost entry is the one written
// by the hop nearest here and the leftmost is whatever the original caller claimed. It is
// walked from the right, discarding entries written by proxies we trust, and the first
// address that is not one of ours is the client. Anything left of that was written by
// somebody we do not vouch for and is ignored.
//
// With no trusted ranges configured the header is not read at all. Trusting the leftmost
// entry because it is usually the client would make a per-address limit one header away
// from being no limit, since the caller writes that entry.
func clientIP(r *http.Request, trusted []netip.Prefix) string {
	direct := remoteAddr(r)
	if len(trusted) == 0 || !trustedAddr(direct, trusted) {
		return addressString(direct)
	}

	forwarded := r.Header.Values(ForwardedHeader)
	for hop := len(forwarded) - 1; hop >= 0; hop-- {
		entries := strings.Split(forwarded[hop], ",")
		for entry := len(entries) - 1; entry >= 0; entry-- {
			address, err := parseAddr(entries[entry])
			if err != nil {
				// A hop that wrote something unparseable is a hop that cannot be walked
				// past, since there is no telling whose entry the next one to the left is.
				return addressString(direct)
			}
			if !trustedAddr(address, trusted) {
				return addressString(address)
			}
		}
	}

	// Every entry was one of our own proxies, so there is nobody further out to name.
	return addressString(direct)
}

// remoteAddr is the address the connection itself came from, which is the proxy's when
// there is one in front.
func remoteAddr(r *http.Request) netip.Addr {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		// A RemoteAddr without a port is not something net/http produces, but a test
		// server and a unix socket both can.
		host = r.RemoteAddr
	}
	address, err := parseAddr(host)
	if err != nil {
		return netip.Addr{}
	}
	return address
}

// parseAddr reads one address, dropping the zone an IPv6 link-local address carries and the
// brackets a proxy may have written around it.
func parseAddr(value string) (netip.Addr, error) {
	value = strings.TrimSpace(value)
	value = strings.TrimPrefix(value, "[")
	value = strings.TrimSuffix(value, "]")
	address, err := netip.ParseAddr(value)
	if err != nil {
		return netip.Addr{}, err
	}
	return address.Unmap().WithZone(""), nil
}

func trustedAddr(address netip.Addr, trusted []netip.Prefix) bool {
	if !address.IsValid() {
		return false
	}
	for _, prefix := range trusted {
		if prefix.Contains(address) {
			return true
		}
	}
	return false
}

// addressString renders an address for use as part of a key, and is empty for one that
// could not be read, so an unreadable address counts against nobody rather than against
// everybody together.
func addressString(address netip.Addr) string {
	if !address.IsValid() {
		return ""
	}
	return address.String()
}
