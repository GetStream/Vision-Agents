package sipbridge

import (
	"context"
	"fmt"
	"net"
	"net/netip"
	"strconv"
	"strings"

	"github.com/emiago/sipgo/sip"

	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// resolver is the part of *net.Resolver the guard uses, so tests can answer offline.
type resolver interface {
	LookupNetIP(ctx context.Context, network, host string) ([]netip.Addr, error)
}

// guard keeps the customer leg off private networks, this host and the cloud metadata
// server. The customer picks the trunk host, and its answers pick the Contact and
// Record-Route that later requests in the dialog go to, so every request on that leg is
// checked where it is sent, not when the trunk is saved.
type guard struct {
	allow    func(netip.Addr) bool
	resolver resolver
}

// withDefaults fills what tests leave out: only public addresses, through the system resolver.
func (g guard) withDefaults() guard {
	if g.allow == nil {
		g.allow = egress.IsPublic
	}
	if g.resolver == nil {
		g.resolver = net.DefaultResolver
	}
	return g
}

// check refuses req unless every address its destination resolves to is allowed. Over UDP
// and TCP it then sends req to the address it checked. Over TLS it keeps the name, so the
// certificate can be checked against it.
func (g guard) check(ctx context.Context, req *sip.Request) error {
	dest := req.Destination()
	host, port, err := sip.ParseAddr(dest)
	if err != nil {
		return stack.Wrap(fmt.Errorf("sipbridge: refusing %s to %q: %w", req.Method, dest, err))
	}
	if port == 0 {
		port = sip.DefaultPort(req.Transport())
	}
	ips, err := g.addrs(ctx, host)
	if err != nil {
		return stack.Wrap(fmt.Errorf("sipbridge: refusing %s: %w", req.Method, err))
	}
	if !strings.EqualFold(req.Transport(), "tls") {
		req.SetDestination(net.JoinHostPort(preferIPv4(ips).Unmap().String(), strconv.Itoa(port)))
	}
	return nil
}

// addrs resolves host and refuses it unless every address is allowed.
func (g guard) addrs(ctx context.Context, host string) ([]netip.Addr, error) {
	host = strings.TrimSuffix(strings.TrimPrefix(host, "["), "]")
	if ip, err := netip.ParseAddr(host); err == nil {
		if !g.allow(ip) {
			return nil, stack.Wrap(fmt.Errorf("%s is not a public address", ip))
		}
		return []netip.Addr{ip}, nil
	}
	ips, err := g.resolver.LookupNetIP(ctx, "ip", host)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("resolve %q: %w", host, err))
	}
	if len(ips) == 0 {
		return nil, stack.Wrap(fmt.Errorf("%q has no address", host))
	}
	for _, ip := range ips {
		if !g.allow(ip) {
			return nil, stack.Wrap(fmt.Errorf("%q resolves to %s, which is not a public address", host, ip))
		}
	}
	return ips, nil
}

// preferIPv4 picks the address sipgo would have: it prefers IPv4 when it resolves.
func preferIPv4(ips []netip.Addr) netip.Addr {
	for _, ip := range ips {
		if ip.Unmap().Is4() {
			return ip
		}
	}
	return ips[0]
}

// guardedRequester sends a client's requests the way sipgo does without a requester, after
// the guard has passed them. It is sipgo's one hook that sees every request the client
// sends, in-dialog ones and ACKs included, after sipgo has added the route set.
type guardedRequester struct {
	guard guard
	tx    *sip.TransactionLayer
}

func (r guardedRequester) Request(ctx context.Context, req *sip.Request) (sip.ClientTransaction, error) {
	if err := r.guard.check(ctx, req); err != nil {
		return nil, err
	}
	// sipgo sends an ACK for a 2xx straight to the transport, outside any transaction, and
	// ignores the transaction returned for it.
	if req.IsAck() {
		return nil, stack.Wrap(r.tx.Transport().WriteMsg(req))
	}
	tx, err := r.tx.NewClientTransaction(ctx, req)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	fillContact(req)
	if err := tx.Init(); err != nil {
		tx.Terminate()
		return nil, stack.Wrap(err)
	}
	return tx, nil
}

// fillContact does what sipgo's DialogClientSession.Invite does once the connection is open,
// and skips when a requester is set: a Contact without a port gets the connection's address
// from Via.
func fillContact(req *sip.Request) {
	if !req.IsInvite() {
		return
	}
	contact, via := req.Contact(), req.Via()
	if contact == nil || via == nil || contact.Address.Port != 0 {
		return
	}
	if contact.Address.Host == "" {
		contact.Address.Host = via.Host
		contact.Address.Port = via.Port
		return
	}
	if via.Host == contact.Address.Host {
		contact.Address.Port = via.Port
	}
}
