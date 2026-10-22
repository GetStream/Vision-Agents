package main

import (
	"context"
	"log/slog"
	"net/netip"

	"github.com/GetStream/Vision-Agents/acceleration/internal/locate"
)

type locator interface {
	Locate(ctx context.Context, ip netip.Addr) (locate.Place, error)
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

// logStreamSFUs writes what Stream said about each side's SFU, which the hop lines cannot
// carry: where Stream placed each side, and how far that was from its SFU.
func logStreamSFUs(logger *slog.Logger, sites callSitesRow) {
	logger.Info("hop", "hop", "stream_sfus", "sfus_used", sites.SFUsUsed,
		"agent_sfu", sites.Agent.SFU, "agent_location", sites.Agent.Location, "agent_distance_to_sfu_km", sites.Agent.DistanceKm,
		"caller_sfu", sites.Caller.SFU, "caller_location", sites.Caller.Location, "caller_distance_to_sfu_km", sites.Caller.DistanceKm,
		"how", "reported")
}
