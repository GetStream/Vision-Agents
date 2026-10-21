package main

import (
	"context"
	"log/slog"
	"net/netip"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
	"github.com/GetStream/Vision-Agents/acceleration/internal/locate"
)

type locator interface {
	Locate(ctx context.Context, ip netip.Addr) (locate.Place, error)
}

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

// siteReads is what the router said about the call, and how to resolve hosts.
type siteReads struct {
	Direction  person.Direction
	Joined     joinedSites
	Placed     callPlacedRow
	Sites      callSitesRow
	RouterHost string
	Lookup     lookupFunc
	ResolveSIP lookupFunc // resolves from the place chosen with -sip-lookup-from
}

// collectHops resolves each hop's host. A host that does not resolve is logged and kept
// with no address.
func collectHops(ctx context.Context, logger *slog.Logger, in siteReads) []hop {
	sipHost := sipHostOf(in.Direction, in.Placed, in.Sites)
	return []hop{
		resolved(ctx, logger, in.Lookup, hop{Name: "agent_sfu", Host: in.Joined.SFUHost, How: "reported"}),
		resolved(ctx, logger, in.Lookup, hop{Name: "router_address", Host: in.RouterHost, How: "observed"}),
		resolved(ctx, logger, in.ResolveSIP, hop{Name: "stream_sip", Host: sipHost, How: "inferred"}),
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

// logHops writes one line per address. A place that cannot be found does not hide the address.
func logHops(ctx context.Context, logger *slog.Logger, loc locator, hops []hop) {
	for _, h := range hops {
		if len(h.Addrs) == 0 {
			logger.Info("hop", "hop", h.Name, "host", h.Host, "address", "", "place", "", "how", h.How)
			continue
		}
		for _, addr := range h.Addrs {
			place := ""
			if found, err := loc.Locate(ctx, addr); err == nil {
				place = found.String()
			}
			logger.Info("hop", "hop", h.Name, "host", h.Host, "address", addr.String(), "place", place, "how", h.How)
		}
	}
}

func logStreamSFUs(logger *slog.Logger, sites callSitesRow) {
	logger.Info("hop", "hop", "stream_sfus", "sfus_used", sites.SFUsUsed, "sfu_locations", sites.SFULocations, "how", "reported")
}
