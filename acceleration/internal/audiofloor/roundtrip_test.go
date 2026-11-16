package audiofloor

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// trips is how many requests of kind have come to outcome.
func trips(t *testing.T, kind, outcome string) uint64 {
	t.Helper()
	var metric dto.Metric
	require.NoError(t, roundTrips.WithLabelValues(kind, outcome).(prometheus.Metric).Write(&metric))
	return metric.GetHistogram().GetSampleCount()
}

// missed is how many requests of kind have missed their deadline.
func missed(kind string) float64 { return testutil.ToFloat64(deadlineMisses.WithLabelValues(kind)) }

// overAgent is a turn put while the caller talks over the agent.
var overAgent = harness.FlowTurn{Speaking: true, Unfinished: true}

// A ruling while the caller talks over the agent is waited for as long as the settings say,
// and by default for the flow controller's own share of its deadline over the agent, 250 ms.
// A pause's checks keep the flow controller's share whatever they say, with the agent talking
// or not, and so does the flow controller's Hearer.
func TestTheSettingsSayHowLongARulingOverTheAgentIsWaitedFor(t *testing.T) {
	const server = "http://floor:8080"
	require.Equal(t, 250*time.Millisecond, harness.HearDeadline(overAgent))
	defaults := config.Defaults().Agent
	defaults.AudioFloorURL = server
	for name, client := range map[string]*Client{"new": New(server, Opus, nil), "unset": Open(config.Agent{AudioFloorURL: server}), "defaults": Open(defaults)} {
		require.Equal(t, harness.HearDeadline(overAgent), client.HearDeadline(overAgent), name)
	}

	client := Open(config.Agent{AudioFloorURL: server, AudioFloorOverlapTimeout: 400 * time.Millisecond})
	require.Equal(t, 400*time.Millisecond, client.HearDeadline(overAgent))
	require.Equal(t, 400*time.Millisecond, NewHearer(client, EndOfTurn{}, slog.New(slog.DiscardHandler)).HearDeadline(overAgent))
	for _, pause := range []harness.FlowTurn{{}, {Speaking: true}, {Unfinished: true}} {
		require.Equal(t, time.Second, client.HearDeadline(pause), "%+v", pause)
		require.Equal(t, harness.HearDeadline(pause), client.HearDeadline(pause), "%+v", pause)
	}
}

// Every request for a decision or for words is timed, by what it was for and what became of
// it. A request the client never sent is not.
func (s *ClientSuite) TestEveryRequestIsTimedByWhatItWasForAndWhatBecameOfIt() {
	ctx := context.Background()
	pause, over, words := trips(s.T(), kindPause, "ok"), trips(s.T(), kindOverAgent, "ok"), trips(s.T(), kindTranscribe, "ok")
	_, err := s.client.Decide(ctx, booking, Turn{}, some)
	s.Require().NoError(err)
	_, err = s.client.Decide(ctx, booking, Turn{Speaking: true}, some)
	s.Require().NoError(err)
	_, err = s.client.Decide(ctx, booking, turnOf(overAgent), some)
	s.Require().NoError(err)
	_, err = s.client.Transcribe(ctx, some, nil)
	s.Require().NoError(err)
	s.Equal(pause+2, trips(s.T(), kindPause, "ok"), "a pause, with the agent talking or not")
	s.Equal(over+1, trips(s.T(), kindOverAgent, "ok"))
	s.Equal(words+1, trips(s.T(), kindTranscribe, "ok"))

	failed := trips(s.T(), kindPause, "error")
	s.server.mu.Lock()
	s.server.fail = true
	s.server.mu.Unlock()
	_, err = s.client.Decide(ctx, booking, Turn{}, some)
	s.Require().Error(err)
	s.Equal(failed+1, trips(s.T(), kindPause, "error"))

	canceled := trips(s.T(), kindPause, "canceled")
	gone, cancel := context.WithCancel(ctx)
	cancel()
	_, err = s.client.Decide(gone, booking, Turn{}, some)
	s.Require().Error(err)
	s.Equal(canceled+1, trips(s.T(), kindPause, "canceled"), "newer words, not a miss")

	_, err = s.client.Decide(ctx, booking, Turn{}, harness.Audio{})
	s.Require().Error(err)
	s.Equal(failed+1, trips(s.T(), kindPause, "error"), "nothing to send")
}

// A ruling over the agent the server is slower to give than the wait is a deadline missed,
// counted and timed as one, and acted on not at all, as before: the agent talks on, and the
// utterance ends at its pause as ever.
func (s *PausesSuite) TestACheckOverTheAgentPastTheWaitIsADeadlineMissed() {
	misses, timedOut := missed(kindOverAgent), trips(s.T(), kindOverAgent, "deadline")
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.decision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	s.server.words = "No, stop."
	s.server.unfinishedAfter = 400 * time.Millisecond
	s.server.mu.Unlock()
	s.say(400, true)
	s.Eventually(func() bool { return missed(kindOverAgent) == misses+1 }, 5*time.Second, time.Millisecond,
		"given up at the default 250 ms, before the server answered at 400 ms")
	s.Equal(timedOut+1, trips(s.T(), kindOverAgent, "deadline"))
	s.Empty(s.agent.overheardSoFar(), "nothing to act on")

	s.agent.speaks(false)
	s.ending(0.999)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Equal("No, stop.", end.Text)
	s.Len(s.agent.rulings(), 1, "ended by its pause's check")
}

// A deployment whose server needs longer over the agent has the router wait that long: the
// same ruling, given in 300 ms, stops the agent.
func (s *PausesSuite) TestACheckOverTheAgentHasTheDeploymentsWait() {
	s.client.overlapTimeout = 450 * time.Millisecond
	misses, answered := missed(kindOverAgent), trips(s.T(), kindOverAgent, "ok")
	s.agent.speaks(true)
	s.server.mu.Lock()
	s.server.decision = `{"answer":"stop","answer_probability":0.9,"options":[],"decide_ms":1,"ruling":{"disposition":"wait","floor":"stop"}}`
	s.server.words = "No, stop."
	s.server.unfinishedAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.say(400, true)
	s.Eventually(func() bool { return len(s.agent.overheardSoFar()) == 1 }, 5*time.Second, time.Millisecond)
	s.Equal(harness.Stop, s.agent.overheardSoFar()[0].Ruling.Floor)
	s.Equal(answered+1, trips(s.T(), kindOverAgent, "ok"))
	s.Equal(misses, missed(kindOverAgent))
}

// The wait over the agent is no pause's: a pause's check has the flow controller's own share
// of its deadline, however short the deployment's wait over the agent.
func (s *PausesSuite) TestAPausesCheckKeepsItsOwnDeadline() {
	s.client.overlapTimeout = 100 * time.Millisecond
	misses, answered := missed(kindPause), trips(s.T(), kindPause, "ok")
	s.server.mu.Lock()
	s.server.decideAfter = 300 * time.Millisecond
	s.server.mu.Unlock()
	s.say(1000, true)
	s.say(220, false)
	end := next[stt.Transcript](&s.transcribing)
	s.Equal(stt.ModeFinal, end.Mode)
	s.Len(s.agent.rulings(), 1, "ruled in 300 ms, past the wait over the agent")
	s.Equal(answered+1, trips(s.T(), kindPause, "ok"))
	s.Equal(misses, missed(kindPause))
}
