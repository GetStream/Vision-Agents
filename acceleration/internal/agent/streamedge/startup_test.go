package streamedge

import (
	"context"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
)

// mediaStartups is, from the registry /metrics serves, how many phone start-ups
// router_agent_startup_seconds timed in the media stage and how many
// router_agent_startup_failures_total counted there, by cause.
func mediaStartups() (timed uint64, failed map[string]float64, err error) {
	families, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		return 0, nil, err
	}
	failed = map[string]float64{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["stage"] != agent.StageMedia || labels["kind"] != agent.StartupPhone {
				continue
			}
			switch family.GetName() {
			case "router_agent_startup_seconds":
				timed = metric.GetHistogram().GetSampleCount()
			case "router_agent_startup_failures_total":
				failed[labels["cause"]] = metric.GetCounter().GetValue()
			}
		}
	}
	return timed, failed, nil
}

func (s *StreamEdgeSuite) startupSeries() (uint64, map[string]float64) {
	timed, failed, err := mediaStartups()
	s.Require().NoError(err)
	return timed, failed
}

// unconnected is an edge on a call it never joined, so its outgoing connection never comes
// up, and a start-up that has reached the media stage.
func (s *StreamEdgeSuite) unconnected() (*Edge, context.Context) {
	edge, err := New(Options{CallID: "call-1", User: User{ID: "agent"}, APIKey: "key", UserToken: "token"})
	s.Require().NoError(err)
	// Without the coordinator's websocket and the location lookup the client reaches nothing.
	client, err := rtc.NewClient("key", rtc.User{ID: "agent"}, rtc.StaticToken("token"),
		rtc.WithoutCoordinatorWS(), rtc.WithoutLocationDiscovery())
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = client.Close() })
	edge.call = client.Call(defaultCallType, "call-1")

	startup := agent.NewStartup(agent.StartupPhone, time.Now())
	startup.JoinStarted(time.Now())
	startup.JoinEnded(time.Now())
	return edge, agent.WithStartup(s.ctx, startup)
}

func (s *StreamEdgeSuite) TestAnOutgoingConnectionThatNeverComesUpIsAMediaTimeout() {
	// Catches the agent left unheard for the whole limit counted as no failure, or timed as a
	// media stage that ended.
	waited := iceTimeout
	iceTimeout = 50 * time.Millisecond
	s.T().Cleanup(func() { iceTimeout = waited })
	edge, ctx := s.unconnected()
	timed, failed := s.startupSeries()

	watched := make(chan struct{})
	go func() {
		edge.reportICE(ctx, time.Now())
		close(watched)
	}()
	select {
	case <-watched:
	case <-time.After(5 * time.Second):
		s.FailNow("the watch did not give up at iceTimeout")
	}

	timedAfter, failedAfter := s.startupSeries()
	s.Equal(timed, timedAfter, "a connection that never came up was timed")
	s.Equal(failed["timeout"]+1, failedAfter["timeout"])
	s.Equal(failed["error"], failedAfter["error"], "a timeout was counted as an error")
}

func (s *StreamEdgeSuite) TestAnEdgeLeftBeforeItsConnectionCameUpCountsNothing() {
	// Catches a session that ended for its own reason before the connection was up, such as a
	// caller hanging up at once, counted as a start-up that timed out.
	edge, ctx := s.unconnected()
	timed, failed := s.startupSeries()
	ctx, cancel := context.WithCancel(ctx)
	cancel()

	edge.reportICE(ctx, time.Now())

	timedAfter, failedAfter := s.startupSeries()
	s.Equal(timed, timedAfter)
	s.Equal(failed, failedAfter)
}
