package main

import (
	"context"
	"log/slog"
	"net/netip"
)

// hop is one place the call passed through. How says how we know: "observed" (we resolved
// it ourselves), "inferred" (resolved from a chosen place, not seen on the call) or
// "reported" (the router or Stream said so).
type hop struct {
	Name  string
	Host  string
	Addrs []netip.Addr
	How   string
}

type lookupFunc func(ctx context.Context, host string) ([]netip.Addr, error)

// callHosts are the hosts a call passed through. CustomerTrunk is empty unless the call went
// out through a customer's own SIP trunk.
type callHosts struct {
	AgentSFU, Router, StreamSIP, CustomerTrunk string
}

// collectHops resolves each hop's host. resolveSIP looks the SIP host up from the place chosen
// with -sip-lookup-from. A host that does not resolve is logged and kept with no address.
func collectHops(ctx context.Context, logger *slog.Logger, lookup, resolveSIP lookupFunc, hosts callHosts) []hop {
	hops := []hop{
		resolved(ctx, logger, lookup, hop{Name: "agent_sfu", Host: hosts.AgentSFU, How: "reported"}),
		resolved(ctx, logger, lookup, hop{Name: "router_address", Host: hosts.Router, How: "observed"}),
		resolved(ctx, logger, resolveSIP, hop{Name: "stream_sip", Host: hosts.StreamSIP, How: "inferred"}),
	}
	if hosts.CustomerTrunk != "" {
		hops = append(hops, resolved(ctx, logger, lookup, hop{Name: "customer_trunk", Host: hosts.CustomerTrunk, How: "reported"}))
	}
	return hops
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

func firstAddress(hops []hop, name string) string {
	for _, h := range hops {
		if h.Name == name && len(h.Addrs) > 0 {
			return h.Addrs[0].String()
		}
	}
	return ""
}
