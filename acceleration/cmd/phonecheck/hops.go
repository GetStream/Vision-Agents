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

// collectHops resolves each hop's host. resolveSIP looks the SIP host up from the place chosen
// with -sip-lookup-from. A host that does not resolve is logged and kept with no address.
func collectHops(ctx context.Context, logger *slog.Logger, lookup, resolveSIP lookupFunc,
	agentSFUHost, routerHost, sipHost string) []hop {
	return []hop{
		resolved(ctx, logger, lookup, hop{Name: "agent_sfu", Host: agentSFUHost, How: "reported"}),
		resolved(ctx, logger, lookup, hop{Name: "router_address", Host: routerHost, How: "observed"}),
		resolved(ctx, logger, resolveSIP, hop{Name: "stream_sip", Host: sipHost, How: "inferred"}),
	}
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
