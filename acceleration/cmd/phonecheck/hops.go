package main

import (
	"context"
	"log/slog"
	"net/netip"
	"strings"
)

// hop is one place the call passed through. How says how we know: "observed" (seen on the
// call: we resolved it ourselves, or the router's SIP bridge saw it in the SDP), "inferred"
// (resolved from a chosen place, not seen on the call) or "reported" (the router or Stream
// said so).
type hop struct {
	Name  string
	Host  string
	Addrs []netip.Addr
	How   string
}

type lookupFunc func(ctx context.Context, host string) ([]netip.Addr, error)

// callHosts are the hosts a call passed through. CustomerTrunk and StreamMedia are empty
// unless the call went out through a customer's own SIP trunk. StreamMedia is an address,
// not a host: where Stream sent its audio from, as the router's bridge saw it.
type callHosts struct {
	AgentSFU, CallerSFU, Router, StreamSIP, CustomerTrunk, StreamMedia string
}

// collectHops resolves each hop's host. resolveSIP looks the SIP host up from the place chosen
// with -sip-lookup-from. A host that does not resolve is logged and kept with no address.
func collectHops(ctx context.Context, logger *slog.Logger, lookup, resolveSIP lookupFunc, hosts callHosts) []hop {
	hops := []hop{
		resolvedSFU(ctx, logger, lookup, hop{Name: "agent_sfu", Host: hosts.AgentSFU, How: "reported"}),
		resolvedSFU(ctx, logger, lookup, hop{Name: "caller_sfu", Host: hosts.CallerSFU, How: "reported"}),
		resolved(ctx, logger, lookup, hop{Name: "router_address", Host: hosts.Router, How: "observed"}),
		resolved(ctx, logger, resolveSIP, hop{Name: "stream_sip", Host: hosts.StreamSIP, How: "inferred"}),
	}
	if hosts.CustomerTrunk != "" {
		hops = append(hops, resolved(ctx, logger, lookup, hop{Name: "customer_trunk", Host: hosts.CustomerTrunk, How: "reported"}))
	}
	if hosts.StreamMedia != "" {
		hops = append(hops, parsed(logger, hop{Name: "stream_media", Host: hosts.StreamMedia, How: "observed"}))
	}
	return hops
}

// parsed is a hop whose host is already an address.
func parsed(logger *slog.Logger, h hop) hop {
	addr, err := netip.ParseAddr(h.Host)
	if err != nil {
		logger.Warn("a hop's address is not an address", "hop", h.Name, "host", h.Host)
		return h
	}
	h.Addrs = []netip.Addr{addr}
	return h
}

func resolved(ctx context.Context, logger *slog.Logger, lookup lookupFunc, h hop) hop {
	if h.Host == "" {
		return h
	}
	addrs, err := lookup(ctx, h.Host)
	if err != nil {
		logger.Warn("could not resolve a hop", "hop", h.Name, "host", h.Host, "error", err)
	}
	h.Addrs = addrs
	return h
}

// resolvedSFU leaves an SFU name with no dot unresolved: it is Stream's id for the SFU, not a
// DNS name. Only SFUs: a router on "localhost" still resolves.
func resolvedSFU(ctx context.Context, logger *slog.Logger, lookup lookupFunc, h hop) hop {
	if !strings.Contains(h.Host, ".") {
		return h
	}
	return resolved(ctx, logger, lookup, h)
}

func firstAddress(hops []hop, name string) string {
	for _, h := range hops {
		if h.Name == name && len(h.Addrs) > 0 {
			return h.Addrs[0].String()
		}
	}
	return ""
}
