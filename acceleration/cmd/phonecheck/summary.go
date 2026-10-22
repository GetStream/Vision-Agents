package main

import (
	"log/slog"
	"time"
)

type summary struct {
	Direction, Provider, SessionID, Outcome string
	Connect, Talked, Total                  time.Duration
	AgentSFU, CallerSFU, RouterAddress      string
	SIPHost, SFUsUsed, RegionLabel          string
	VendorCallID                            string // empty for inbound
}

func (s summary) log(logger *slog.Logger) {
	args := []any{
		"direction", s.Direction, "provider", s.Provider, "session_id", s.SessionID, "outcome", s.Outcome,
		"connect_ms", s.Connect.Milliseconds(), "talked_ms", s.Talked.Milliseconds(), "total_ms", s.Total.Milliseconds(),
		"agent_sfu", s.AgentSFU, "caller_sfu", s.CallerSFU, "router_address", s.RouterAddress, "sip_host", s.SIPHost, "sfus_used", s.SFUsUsed,
		"region_label", s.RegionLabel,
	}
	if s.VendorCallID != "" {
		args = append(args, "vendor_call_id", s.VendorCallID)
	}
	logger.Info("call summary", args...)
}
