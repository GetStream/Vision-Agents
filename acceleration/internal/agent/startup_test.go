package agent

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

// failedStartups is how many start-ups router_agent_startup_failures_total counted in stage
// for kind and cause.
func failedStartups(stage, kind, cause string) float64 {
	return testutil.ToFloat64(startupFailures.WithLabelValues(stage, kind, cause))
}

// timedStartups is how many start-ups router_agent_startup_seconds timed in stage for kind.
func (s *AgentSuite) timedStartups(stage, kind string) uint64 {
	written := &dto.Metric{}
	s.Require().NoError(startupSeconds.WithLabelValues(stage, kind).(prometheus.Metric).Write(written))
	return written.GetHistogram().GetSampleCount()
}

func (s *AgentSuite) TestAJoinEndingAsTheConnectionComesUpTimesEachStageOnce() {
	// Catches the race between the agent seeing its join return and Stream's edge seeing its
	// connection up on another goroutine: a stage lost, timed twice, or a data race (-race).
	const startups = 50
	join, media := s.timedStartups(StageJoin, StartupPhone), s.timedStartups(StageMedia, StartupPhone)

	for range startups {
		startup := NewStartup(StartupPhone, time.Now())
		startup.JoinStarted(time.Now())
		var both sync.WaitGroup
		both.Go(func() { startup.JoinEnded(time.Now()) })
		both.Go(func() { startup.Connected(time.Now()) })
		both.Wait()
	}

	s.Equal(join+startups, s.timedStartups(StageJoin, StartupPhone))
	s.Equal(media+startups, s.timedStartups(StageMedia, StartupPhone))
}

func (s *AgentSuite) TestAStageThatFailedIsNotTimedWhenItsEndComesAfter() {
	// Catches a session Create gave up on while its connection was coming up counted both as
	// a failure and as a media stage that ended.
	media, failed := s.timedStartups(StageMedia, StartupPhone), failedStartups(StageMedia, StartupPhone, causeError)
	startup := NewStartup(StartupPhone, time.Now())
	startup.JoinStarted(time.Now())
	startup.JoinEnded(time.Now())

	startup.Failed()
	startup.Connected(time.Now())

	s.Equal(media, s.timedStartups(StageMedia, StartupPhone), "the failed stage was timed")
	s.Equal(failed+1, failedStartups(StageMedia, StartupPhone, causeError))
}

func (s *AgentSuite) TestAStartUpStopsOnceAndNotAfterItEnded() {
	// Catches a start-up counted as failed twice, by its edge's timeout and then by Create,
	// and one that ended with the connection up counted as failed after.
	failures, timeouts := failedStartups(StageMedia, StartupPhone, causeError), failedStartups(StageMedia, StartupPhone, causeTimeout)
	stopped := NewStartup(StartupPhone, time.Now())
	stopped.JoinStarted(time.Now())
	stopped.JoinEnded(time.Now())
	stopped.TimedOut()
	stopped.Failed()

	ended := NewStartup(StartupPhone, time.Now())
	ended.JoinStarted(time.Now())
	ended.JoinEnded(time.Now())
	ended.Connected(time.Now())
	ended.Failed()
	ended.TimedOut()

	s.Equal(timeouts+1, failedStartups(StageMedia, StartupPhone, causeTimeout))
	s.Equal(failures, failedStartups(StageMedia, StartupPhone, causeError))
}
