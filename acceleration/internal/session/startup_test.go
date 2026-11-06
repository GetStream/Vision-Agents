package session

import (
	"context"
	"errors"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
)

// The stages, kinds and causes the start-up metrics document. Written out rather than read
// from package agent, so a value added, dropped or renamed there fails here.
var (
	documentedStages = []string{"create", "join", "media"}
	documentedKinds  = []string{"web", "phone"}
	// documentedFailures is each stage's causes.
	documentedFailures = map[string][]string{
		"create": {"error"},
		"join":   {"error"},
		"media":  {"error", "timeout"},
	}
)

// startups is, for each stage and kind, how many start-ups router_agent_startup_seconds
// timed and how many router_agent_startup_failures_total counted, keyed stage/kind and
// stage/kind/cause.
type startups struct {
	timed  map[string]uint64
	sums   map[string]float64
	failed map[string]float64
}

// startups reads the start-up metrics from the registry /metrics serves.
func (s *SessionSuite) startups() startups {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	read := startups{timed: map[string]uint64{}, sums: map[string]float64{}, failed: map[string]float64{}}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			switch family.GetName() {
			case "router_agent_startup_seconds":
				read.timed[labels["stage"]+"/"+labels["kind"]] = metric.GetHistogram().GetSampleCount()
				read.sums[labels["stage"]+"/"+labels["kind"]] = metric.GetHistogram().GetSampleSum()
			case "router_agent_startup_failures_total":
				read.failed[labels["stage"]+"/"+labels["kind"]+"/"+labels["cause"]] = metric.GetCounter().GetValue()
			}
		}
	}
	return read
}

// startedUp checks that, since before, the start-ups timed grew by one for each of stages of
// kind, and the failures by one for failed (stage/kind/cause, or empty for none), and that
// nothing else moved.
func (s *SessionSuite) startedUp(before startups, kind string, stages []string, failed string) {
	after := s.startups()
	want := startups{timed: map[string]uint64{}, failed: map[string]float64{}}
	for key, count := range before.timed {
		want.timed[key] = count
	}
	for key, count := range before.failed {
		want.failed[key] = count
	}
	for _, stage := range stages {
		want.timed[stage+"/"+kind]++
	}
	if failed != "" {
		want.failed[failed]++
	}
	s.Equal(want.timed, after.timed, "the stages timed")
	s.Equal(want.failed, after.failed, "the failures counted")
}

func (s *SessionSuite) TestAVoiceSessionsStartUpIsTimedOnceAtEachStage() {
	// Catches a stage left untimed, timed twice by an edge that reports its connection again,
	// or timed again when the session's agent is built anew for another call, and a session
	// counted under another kind.
	edge := &connectingEdge{attendedEdge: attendedEdge{newQuietEdge()}, connect: make(chan struct{})}
	s.opens = edge
	s.manages()
	before := s.startups()

	created := s.joins(Spec{})
	s.startedUp(before, "web", []string{"create", "join"}, "")

	close(edge.connect)
	s.eventually(func() bool { return s.startups().timed["media/web"] == before.timed["media/web"]+1 },
		"the connection coming up after the join was not timed")
	s.Require().NoError(created.StopVoice(s.ctx))
	s.Require().NoError(created.StartVoice(s.ctx, Call{}))
	s.Require().NoError(created.Close())
	s.startedUp(before, "web", documentedStages, "")
}

func (s *SessionSuite) TestAConnectionUpBeforeTheJoinReturnsIsTimedAtNoWait() {
	// Catches the media stage lost when Stream's edge sees its connection up before the agent
	// sees the join return, or timed from the wrong end as a negative wait.
	s.opens = &connectingEdge{attendedEdge: attendedEdge{newQuietEdge()}, early: true}
	s.manages()
	before := s.startups()

	s.joins(Spec{})

	s.startedUp(before, "web", documentedStages, "")
	s.Equal(before.sums["media/web"], s.startups().sums["media/web"], "the media stage waited")
}

func (s *SessionSuite) TestAFailedJoinIsAFailureOfTheJoinAndNoLatency() {
	// Catches a join that failed timed as though it ended, counted as a failure of the create,
	// or counted as a start-up failure without the session's create_failed end beside it.
	s.opens = refusedEdge{newQuietEdge()}
	s.manages()
	before, ends := s.startups(), s.allEnds()

	_, err := s.manager.Create(s.ctx, Spec{
		CallID: "call-1", CustomerID: "acme",
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
	})
	s.Require().ErrorContains(err, "the call refused the agent")

	s.startedUp(before, "web", []string{"create"}, "join/web/error")
	s.endedOnce(ends, "create_failed")
}

func (s *SessionSuite) TestASessionWhoseModelWouldNotOpenFailsInTheCreate() {
	// Catches a session given up on before it joined counted as a failed join, or not at all.
	s.speech = func() (sts.STS, error) { return nil, errors.New("the provider is down") }
	s.manages()
	before, ends := s.startups(), s.allEnds()

	_, err := s.manager.Create(s.ctx, Spec{CallID: "call-1", CustomerID: "acme", STSTarget: "en-low-latency"})
	s.Require().Error(err)

	s.startedUp(before, "web", nil, "create/web/error")
	s.endedOnce(ends, EndCreateVendorFailed)
}

func (s *SessionSuite) TestAPhoneSessionsStartUpIsCountedAsPhone() {
	// Catches a session on a number counted as a web one.
	s.manages()
	before := s.startups()

	s.rings(outbound())

	s.startedUp(before, "phone", []string{"create", "join"}, "")
}

func (s *SessionSuite) TestATextSessionHasNoStartUp() {
	// Catches a session that joins no call timed or failed as a voice start-up.
	s.manages()
	before := s.startups()

	s.Require().NoError(s.writes(Spec{}).Close())
	_, ends := s.startups(), s.allEnds()
	_, err := s.manager.Create(s.ctx, Spec{
		CustomerID: "acme", Text: true, LLMTarget: "en-low-latency", Guardrail: "{not a policy",
	})
	s.Require().Error(err)

	s.startedUp(before, "web", nil, "")
	s.Equal(ends["create_failed"]+1, s.ends("create_failed"), "the text session did not fail in its create")
}

func (s *SessionSuite) TestACreateTurnedAwayIsNoStartUpFailure() {
	// Catches a create a router shutting down turned away counted as a start-up that failed,
	// which router_session_ends_total does not count either.
	late := newJoiningEdge()
	s.opens = late
	s.manages()
	lost := make(chan error, 1)
	before := s.startups()
	go func() {
		_, err := s.manager.Create(s.ctx, Spec{
			CallID: "call-3", CustomerID: "acme",
			LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
		})
		lost <- err
	}()
	<-late.entered
	s.Require().NoError(s.manager.Shutdown())
	close(late.allowed)
	s.Require().ErrorContains(<-lost, "the manager is shut down")

	s.startedUp(before, "web", []string{"create", "join"}, "")
}

func (s *SessionSuite) TestASessionOnAnEdgeItsCallerHandsInHasNoStartUp() {
	// Catches an audio socket's or a simulation's session counted as a web start-up: a create
	// and join of no call, a media stage that never comes, and a failure after the join as a
	// failed media stage.
	s.manages()
	before := s.startups()

	s.Require().NoError(s.joins(Spec{Edge: attendedEdge{newQuietEdge()}}).Close())
	ends := s.allEnds()
	_, err := s.manager.Create(s.ctx, Spec{
		CallID: "call-2", CustomerID: "acme", Edge: refusedEdge{newQuietEdge()},
		LLMTarget: "en-low-latency", STTTarget: "en-low-latency", TTSTarget: "en-low-latency",
	})
	s.Require().ErrorContains(err, "the call refused the agent")

	s.startedUp(before, "web", nil, "")
	s.endedOnce(ends, "create_failed")
}

func (s *SessionSuite) TestTheCreateStageRunsFromCreateBeingCalled() {
	// Catches the create stage started late in Create, missing the reads before it such as
	// which Stream app the session acts in.
	const resolving = 50 * time.Millisecond
	s.apps = streamapp.NewClients(slowSource{resolving}, streamapp.ClientsOptions{})
	s.manages()
	before := s.startups()

	s.joins(Spec{})

	s.startedUp(before, "web", []string{"create", "join"}, "")
	s.GreaterOrEqual(s.startups().sums["create/web"]-before.sums["create/web"], resolving.Seconds(),
		"the create stage left out the wait for the Stream app")
}

// slowSource takes wait to say a customer has no Stream app.
type slowSource struct{ wait time.Duration }

func (s slowSource) For(context.Context, string) (streamapp.Identity, error) {
	time.Sleep(s.wait)
	return streamapp.Identity{}, streamapp.ErrNoIdentity
}

func (s slowSource) ForApp(context.Context, string, int64) (streamapp.Identity, error) {
	time.Sleep(s.wait)
	return streamapp.Identity{}, streamapp.ErrNoIdentity
}

func (s *SessionSuite) TestEveryStartUpSeriesIsADocumentedStageKindAndCause() {
	// Catches a label value outside the documented sets, such as an id, and a series missing
	// before its first count, which a rate over the first failure would not read.
	s.manages()
	s.joins(Spec{})

	read := s.startups()
	timed, failed := map[string]bool{}, map[string]bool{}
	for key := range read.timed {
		timed[key] = true
	}
	for key := range read.failed {
		failed[key] = true
	}
	wantTimed, wantFailed := map[string]bool{}, map[string]bool{}
	for _, kind := range documentedKinds {
		for _, stage := range documentedStages {
			wantTimed[stage+"/"+kind] = true
			for _, cause := range documentedFailures[stage] {
				wantFailed[stage+"/"+kind+"/"+cause] = true
			}
		}
	}
	s.Equal(wantTimed, timed)
	s.Equal(wantFailed, failed)
}

// connectingEdge is a call whose outgoing connection comes up as Stream's edge reports it: to
// the start-up its Join was given, which is the session's on its first join and none after.
// early has it come up before Join returns, as it can while the edge's ICE watch runs beside
// the end of the join; otherwise once connect is closed, reported twice.
type connectingEdge struct {
	attendedEdge
	early   bool
	connect chan struct{}
}

func (e *connectingEdge) Join(ctx context.Context) error {
	if err := e.attendedEdge.Join(ctx); err != nil {
		return err
	}
	startup := agent.StartupOf(ctx)
	if e.early {
		startup.Connected(time.Now())
		return nil
	}
	go func() {
		<-e.connect
		startup.Connected(time.Now())
		startup.Connected(time.Now())
	}()
	return nil
}
