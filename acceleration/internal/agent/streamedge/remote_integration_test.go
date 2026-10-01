//go:build integration

package streamedge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

func (s *StreamEdgeIntegrationSuite) TestTheAgentsJoinAgainstARemoteDeployment() {
	// Against a real deployment (staging), pinned to STREAMEDGE_SFU_ID, over the network's
	// own round trips: nothing is injected. It reports rather than asserts a budget, since
	// the round trip is whatever the runner has. Each run prints one "agentjoin: {json}"
	// line on stdout.
	sfuID := os.Getenv("STREAMEDGE_SFU_ID")
	if sfuID == "" || os.Getenv("LOCAL_STACK") != "" {
		s.T().Skip("needs a remote deployment and STREAMEDGE_SFU_ID")
	}
	runs := joinRuns
	if value := os.Getenv("STREAMEDGE_JOIN_RUNS"); value != "" {
		parsed, err := strconv.Atoi(value)
		s.Require().NoError(err)
		runs = parsed
	}

	// The first join of a new user and the first call on an SFU pay for setup a running
	// agent has already done.
	s.reportRemoteJoin(s.measureRemoteJoin(-1))
	var failed []string
	for run := range runs {
		m := s.measureRemoteJoin(run)
		s.reportRemoteJoin(m)
		if m.Error != "" {
			failed = append(failed, fmt.Sprintf("run %d: %s", run, m.Error))
			continue
		}
		if !strings.HasPrefix(m.SFU, strings.TrimSuffix(sfuID, ".stream-io-video.com")) {
			failed = append(failed, fmt.Sprintf("run %d: joined %q, not the pinned %q", run, m.SFU, sfuID))
		}
	}
	s.Empty(failed)
}

// remoteJoin is one measured join of the agent against a remote deployment.
type remoteJoin struct {
	Run  int    `json:"run"`
	Flow string `json:"flow"`
	SFU  string `json:"sfu"`
	// JoinMs is how long Edge.Join took.
	JoinMs             float64            `json:"join_ms"`
	PublishToMediaMs   *float64           `json:"publish_to_media_ms"`
	SubscribeToMediaMs *float64           `json:"subscribe_to_media_ms"`
	RTTMs              map[string]float64 `json:"rtt_ms"`
	CriticalPath       []string           `json:"critical_path"`
	CriticalRTTs       float64            `json:"critical_rtts"`
	// AgentHeard and CallerHeard say whether each side heard the other's tone.
	AgentHeard  bool            `json:"agent_heard"`
	CallerHeard bool            `json:"caller_heard"`
	Error       string          `json:"error,omitempty"`
	Trace       json.RawMessage `json:"trace,omitempty"`
}

// measureRemoteJoin puts a caller who is already talking in a call of its own, joins the
// agent and waits for its join trace, which the SDK reports once media flows both ways. A
// failure is recorded in the result, so one bad run does not lose the others.
func (s *StreamEdgeIntegrationSuite) measureRemoteJoin(run int) remoteJoin {
	m := remoteJoin{Run: run}
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	s.callID = fmt.Sprintf("go-edge-%d", time.Now().UnixNano())

	caller, err := New(Options{CallID: s.callID, CallType: s.callType, User: User{ID: "go-edge-caller", Name: "go-edge-caller"},
		coordinatorOptions: s.pin})
	s.Require().NoError(err)
	if err := caller.Join(ctx); err != nil {
		m.Error = fmt.Sprintf("caller join: %v", err)
		return m
	}
	defer caller.Leave()
	s.speak(ctx, caller, 48_000)

	agentEdge, err := New(Options{CallID: s.callID, CallType: s.callType, User: User{ID: "go-edge-agent", Name: "go-edge-agent"},
		coordinatorOptions: s.pin})
	s.Require().NoError(err)
	started := time.Now()
	if err := agentEdge.Join(ctx); err != nil {
		m.Error = fmt.Sprintf("agent join: %v", err)
		return m
	}
	m.JoinMs = float64(time.Since(started).Microseconds()) / 1000
	defer agentEdge.Leave()
	if state := agentEdge.call.GetState(); state != nil {
		m.SFU = state.EdgeName
	}
	s.speak(ctx, agentEdge, 24_000)

	var joined agent.JoinTrace
	select {
	case joined = <-agentEdge.JoinTraces():
	case <-time.After(rtc.JoinTraceTimeout + 5*time.Second):
		m.Error = "the agent reported no join trace"
		return m
	}
	var report struct {
		PublishToMediaMs   *float64           `json:"publish_to_media_ms"`
		SubscribeToMediaMs *float64           `json:"subscribe_to_media_ms"`
		RTTMs              map[string]float64 `json:"rtt_ms"`
		CriticalPath       []string           `json:"critical_path"`
		CriticalRTTs       float64            `json:"critical_rtts"`
		Spans              []struct {
			Name string `json:"name"`
		} `json:"spans"`
	}
	s.Require().NoError(json.Unmarshal(joined.Trace, &report))
	m.Trace = joined.Trace
	m.PublishToMediaMs, m.SubscribeToMediaMs = report.PublishToMediaMs, report.SubscribeToMediaMs
	m.RTTMs, m.CriticalPath, m.CriticalRTTs = report.RTTMs, report.CriticalPath, report.CriticalRTTs
	// Read from the spans, so the same test names the flow on SDKs without Call.JoinFlow.
	m.Flow = "legacy"
	if slices.ContainsFunc(report.Spans, func(span struct {
		Name string `json:"name"`
	}) bool {
		return span.Name == "coord.fastjoin"
	}) {
		m.Flow = "fast"
	}

	callerHeard, callerErr := heardFrom(caller, "go-edge-agent")
	agentHeard, agentErr := heardFrom(agentEdge, "go-edge-caller")
	m.CallerHeard, m.AgentHeard = callerHeard, agentHeard
	switch {
	case report.PublishToMediaMs == nil:
		m.Error = "the agent's audio never reached the SFU"
	case report.SubscribeToMediaMs == nil:
		m.Error = "the caller's audio never reached the agent"
	case errors.Join(callerErr, agentErr) != nil:
		m.Error = errors.Join(callerErr, agentErr).Error()
	}
	return m
}

// heardFrom waits for a tone from the given user.
func heardFrom(edge *Edge, userID string) (bool, error) {
	deadline := time.After(audioArrivesWithin)
	for {
		select {
		case inbound, open := <-edge.Audio():
			if !open {
				return false, errors.New("the edge left before any audio arrived")
			}
			if loudest(inbound.Audio) > 500 {
				if inbound.Participant.UserID != userID {
					return false, fmt.Errorf("heard %q, want %q", inbound.Participant.UserID, userID)
				}
				return true, nil
			}
		case <-deadline:
			return false, fmt.Errorf("nothing from %q within %s", userID, audioArrivesWithin)
		}
	}
}

func (s *StreamEdgeIntegrationSuite) reportRemoteJoin(m remoteJoin) {
	line, err := json.Marshal(m)
	s.Require().NoError(err)
	fmt.Printf("agentjoin: %s\n", line)
	s.T().Logf("run %d: flow %s, sfu %s, Edge.Join %.0f ms, publish %s, subscribe %s, rtt %v %s",
		m.Run, m.Flow, m.SFU, m.JoinMs, msOrNone(m.PublishToMediaMs), msOrNone(m.SubscribeToMediaMs), m.RTTMs, m.Error)
}

func msOrNone(ms *float64) string {
	if ms == nil {
		return "none"
	}
	return fmt.Sprintf("%.0f ms", *ms)
}
