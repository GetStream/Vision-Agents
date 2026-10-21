package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

type joinedSites struct {
	SFUEdge, SFUHost string
	JoinedAt         time.Time
}

// callSitesRow is as the router stores it: strings. SIPHost is only there for an inbound call.
type callSitesRow struct {
	SFUsUsed, SFULocations, SIPHost, StreamSession string
}

type callPlacedRow struct {
	SIPHost, Vendor, VendorCallID string
}

// readJoined reads the first joined row; a rebuilt session has more than one.
func readJoined(ctx context.Context, r router, sessionID string) (joinedSites, error) {
	rows, err := r.LogRows(ctx, sessionID)
	if err != nil {
		return joinedSites{}, err
	}
	var first *acceleration.AgentLog
	for i, row := range rows {
		if row.EventType == "joined" && (first == nil || row.OccurredAt.Before(first.OccurredAt)) {
			first = &rows[i]
		}
	}
	if first == nil {
		return joinedSites{}, errors.New("the agent has no joined row in its log")
	}
	whole, err := r.LogRow(ctx, first.Id)
	if err != nil {
		return joinedSites{}, err
	}
	d := detailsOf(whole)
	return joinedSites{SFUEdge: d["sfu_edge"], SFUHost: d["sfu_host"], JoinedAt: whole.OccurredAt}, nil
}

// readCallPlaced is false when the router wrote no call_placed row, as for an inbound call.
func readCallPlaced(ctx context.Context, r router, sessionID string) (callPlacedRow, bool, error) {
	d, found, err := readEvent(ctx, r, sessionID, "call_placed")
	if err != nil || !found {
		return callPlacedRow{}, false, err
	}
	return callPlacedRow{SIPHost: d["stream_sip_host"], Vendor: d["vendor"], VendorCallID: d["vendor_call_id"]}, true, nil
}

// readCallSites polls because the router writes the row after the call ends. The last poll is
// at the deadline itself.
func readCallSites(ctx context.Context, r router, sessionID string, wait, every time.Duration) (callSitesRow, bool, error) {
	deadline := time.Now().Add(wait)
	for {
		d, found, err := readEvent(ctx, r, sessionID, "call_sites")
		if err != nil {
			return callSitesRow{}, false, err
		}
		if found {
			return callSitesRow{
				SFUsUsed: d["sfus_used"], SFULocations: d["sfu_locations"],
				SIPHost: d["stream_sip_host"], StreamSession: d["stream_session_id"],
			}, true, nil
		}
		left := time.Until(deadline)
		if left <= 0 {
			return callSitesRow{}, false, nil
		}
		select {
		case <-ctx.Done():
			return callSitesRow{}, false, ctx.Err()
		case <-time.After(min(every, left)):
		}
	}
}

// sipHostOf: the router logs the host on call_placed for an outbound call, on call_sites for an inbound one.
func sipHostOf(dir person.Direction, placed callPlacedRow, sites callSitesRow) string {
	if dir == person.Outbound {
		return placed.SIPHost
	}
	return sites.SIPHost
}

// readEvent finds the session's row of one event type. The list omits details, so the row is fetched whole.
func readEvent(ctx context.Context, r router, sessionID, eventType string) (map[string]string, bool, error) {
	rows, err := r.LogRows(ctx, sessionID)
	if err != nil {
		return nil, false, err
	}
	for _, row := range rows {
		if row.EventType != eventType {
			continue
		}
		whole, err := r.LogRow(ctx, row.Id)
		if err != nil {
			return nil, false, fmt.Errorf("read %s row: %w", eventType, err)
		}
		return detailsOf(whole), true, nil
	}
	return nil, false, nil
}

func detailsOf(row acceleration.AgentLog) map[string]string {
	out := map[string]string{}
	if row.Details == nil {
		return out
	}
	for key, value := range *row.Details {
		if text, ok := value.(string); ok {
			out[key] = text
		}
	}
	return out
}
