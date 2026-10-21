package api

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// callSites is how many SFUs Stream ran a call on, and where they were.
type callSites struct {
	SFUsUsed     int
	SFULocations []string
}

// sitesOf reads a call's SFUs out of its stats, one "<id> <datacenter> <country>-<subdivision>"
// per SFU.
func sitesOf(stats *getstream.QueryCallStatsMapResponse) callSites {
	sites := callSites{SFUsUsed: stats.Counts.SfusUsed}
	if stats.Sfus == nil {
		return sites
	}
	for _, sfu := range stats.Sfus.Locations {
		region := strings.Trim(sfu.Location.CountryIsoCode+"-"+sfu.Location.SubdivisionIsoCode, "-")
		sites.SFULocations = append(sites.SFULocations, joinPresent(sfu.ID, sfu.Datacenter, region))
	}
	return sites
}

func joinPresent(parts ...string) string {
	var present []string
	for _, part := range parts {
		if part != "" {
			present = append(present, part)
		}
	}
	return strings.Join(present, " ")
}

// readCallStats asks Stream for one session's call stats.
type readCallStats func(ctx context.Context, client *getstream.Stream, callType, callID, session string) (*getstream.QueryCallStatsMapResponse, error)

func streamCallStats(ctx context.Context, client *getstream.Stream, callType, callID, session string) (*getstream.QueryCallStatsMapResponse, error) {
	response, err := client.Video().GetCallStatsMap(ctx, callType, callID, session, &getstream.GetCallStatsMapRequest{})
	if err != nil {
		return nil, err
	}
	return &response.Data, nil
}

// defaultCallStatsWaits is how long to wait before each stats read. Stream fills a call's
// stats in some time after the call ends, so the first read waits and later ones try again.
var defaultCallStatsWaits = []time.Duration{10 * time.Second, 15 * time.Second, 15 * time.Second, 15 * time.Second}

// fetchCallSites reads a call's stats until they name where its SFUs were, waiting before
// each try. Stats that never fill give the last answer; no answer at all gives the last error.
func fetchCallSites(
	ctx context.Context,
	read func(context.Context) (*getstream.QueryCallStatsMapResponse, error),
	waits []time.Duration,
) (callSites, error) {
	var last *getstream.QueryCallStatsMapResponse
	var lastErr error
	for _, wait := range waits {
		select {
		case <-time.After(wait):
		case <-ctx.Done():
		}
		if err := ctx.Err(); err != nil {
			lastErr = err
			break
		}
		stats, err := read(ctx)
		if err != nil {
			lastErr = err
			continue
		}
		last = stats
		if stats.Sfus != nil && len(stats.Sfus.Locations) > 0 {
			break
		}
	}
	if last == nil {
		return callSites{}, lastErr
	}
	return sitesOf(last), nil
}

// endedCall is a phone call that has just ended, and whose it was.
type endedCall struct {
	customer      string
	app           int64
	callType      string
	callID        string
	streamSession string
	sipHost       string
}

// endedPhoneCall says whether an ended call was a phone call of this hook's app. It has to
// be asked before the call's resources are released, since those are what mark a placed call.
func (s *Server) endedPhoneCall(r *http.Request, origin hookOrigin, event callEvent) (endedCall, bool) {
	if s.store == nil {
		return endedCall{}, false
	}
	callType, callID, split := strings.Cut(event.CallCid, ":")
	if !split {
		return endedCall{}, false
	}
	call := endedCall{callType: callType, callID: callID, streamSession: event.SessionID}

	resources, err := s.store.CallResourcesInApp(r.Context(), origin.scope(), callType, callID)
	if err == nil && len(resources) > 0 {
		call.customer, call.app = resources[0].CustomerID, resources[0].StreamAppPK
	} else if rung, ok := customOf(event.Call.Custom)[phone.NumberKey]; ok {
		number, err := s.store.NumberRungInApp(r.Context(), origin.scope(), rung)
		if err != nil {
			return endedCall{}, false
		}
		call.customer, call.app = number.CustomerID, number.StreamAppPK
		call.sipHost = phone.BridgeHost(number.BridgeURI)
	} else {
		return endedCall{}, false
	}

	if !origin.deployment && call.customer != origin.customer {
		return endedCall{}, false
	}
	return call, true
}

// logCallSites reads where Stream ran an ended phone call and keeps it as a call_sites row.
// It runs after the webhook is answered: Stream's stats take a while to fill.
func (s *Server) logCallSites(call endedCall) {
	read, waits := s.readCallStats, s.callStatsWaits
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()

		sites, err := s.fetchSites(ctx, call, read, waits)
		details := map[string]any{
			"sfus_used":         strconv.Itoa(sites.SFUsUsed),
			"sfu_locations":     strings.Join(sites.SFULocations, ", "),
			"stream_session_id": call.streamSession,
		}
		logged := []any{"session_id", call.callID, "stream_session_id", call.streamSession,
			"sfus_used", sites.SFUsUsed, "sfu_locations", sites.SFULocations}
		if call.sipHost != "" {
			details["stream_sip_host"] = call.sipHost
			logged = append(logged, "stream_sip_host", call.sipHost)
		}
		if err != nil {
			details["error_message"] = err.Error()
			s.logger.Warn("could not read a phone call's sites", append(logged, "error", err)...)
		} else {
			s.logger.Info("phone call sites", logged...)
		}
		s.recordCallSites(ctx, call, details)
	}()
}

func (s *Server) fetchSites(ctx context.Context, call endedCall, read readCallStats, waits []time.Duration) (callSites, error) {
	bound, err := s.streamForApp(ctx, call.customer, call.app, true)
	if err != nil {
		return callSites{}, err
	}
	return fetchCallSites(ctx, func(ctx context.Context) (*getstream.QueryCallStatsMapResponse, error) {
		return read(ctx, bound.Client, call.callType, call.callID, call.streamSession)
	}, waits)
}

// recordCallSites writes the call_sites row, under the session's config and agent when the
// router ran one for the call.
func (s *Server) recordCallSites(ctx context.Context, call endedCall, details map[string]any) {
	entry := &store.AgentLog{
		CustomerID: call.customer,
		SessionID:  call.callID,
		Source:     "system",
		Severity:   "info",
		EventType:  "call_sites",
		Message:    "Call sites",
		Details:    details,
	}
	if session, err := s.store.StoredSession(ctx, call.customer, call.callID); err == nil {
		entry.ConfigID, entry.AgentID = session.ConfigID, session.AgentID
	}
	if err := s.store.RecordAgentLog(ctx, entry); err != nil {
		s.logger.Warn("could not record a phone call's sites", "session_id", call.callID, "error", err)
	}
}
