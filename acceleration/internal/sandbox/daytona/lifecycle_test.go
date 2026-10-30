package daytona

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox/daytona/daytonatest"
)

// told is what a sandbox told its meter, in order.
type told struct {
	mu   sync.Mutex
	said []string
}

func (t *told) add(said string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.said = append(t.said, said)
}

func (t *told) Created(r Reserved) {
	t.add(fmt.Sprintf("created %s %g vCPU %g GiB %g GiB %s", r.ID, r.CPU, r.MemoryGB, r.DiskGB, r.Class))
}
func (t *told) Stopped()  { t.add("stopped") }
func (t *told) Started()  { t.add("started") }
func (t *told) Orphaned() { t.add("orphaned") }
func (t *told) Gone()     { t.add("gone") }

func (t *told) all() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return slices.Clone(t.said)
}

// lockedLog is a log a sandbox's timers write while the test reads it.
type lockedLog struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (l *lockedLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.Write(p)
}

func (l *lockedLog) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.buf.String()
}

// LifecycleSuite runs sandboxes through the states Daytona keeps them in, against a Daytona
// in process.
type LifecycleSuite struct {
	suite.Suite
	ctx     context.Context
	daytona *daytonatest.Server
	meter   *told
	logs    *lockedLog
}

func TestLifecycleSuite(t *testing.T) {
	suite.Run(t, new(LifecycleSuite))
}

func (s *LifecycleSuite) SetupTest() {
	s.ctx = context.Background()
	s.daytona = daytonatest.Start(s.T())
	s.meter = &told{}
	s.logs = &lockedLog{}
}

// box is a sandbox on the test's Daytona, metered by the test, on lifecycle.
func (s *LifecycleSuite) box(lifecycle sandbox.Lifecycle, labels map[string]string) *Sandbox {
	box, err := New(Options{APIKey: "key-1", APIURL: s.daytona.APIURL(), Lifecycle: lifecycle, Labels: labels, Meter: s.meter,
		Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

// stalled is a broker whose BeforeRun waits until let is closed, so a run holds its sandbox
// for as long as a test needs.
type stalled struct {
	entered chan struct{}
	let     chan struct{}
	once    sync.Once
}

// release lets the run go on.
func (b *stalled) release() { b.once.Do(func() { close(b.let) }) }

func (b *stalled) Prepare(context.Context, []sandbox.Grant) (sandbox.Mount, error) {
	return sandbox.Mount{}, nil
}

func (b *stalled) BeforeRun(ctx context.Context) error {
	b.entered <- struct{}{}
	select {
	case <-b.let:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *stalled) Close(context.Context) error { return nil }

// busy is a sandbox on lifecycle with a run inside it until the broker's let is closed. Close
// gives up waiting for that run after half a second, which a create on the test's Daytona takes
// far less than.
func (s *LifecycleSuite) busy(lifecycle sandbox.Lifecycle) (*Sandbox, *stalled) {
	broker := &stalled{entered: make(chan struct{}, 1), let: make(chan struct{})}
	box, err := New(Options{APIKey: "key-1", APIURL: s.daytona.APIURL(), Lifecycle: lifecycle, Meter: s.meter,
		Broker: broker, Timeout: 500 * time.Millisecond, Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(err)
	ran := make(chan error, 1)
	go func() {
		_, err := box.Run(s.ctx, "print(1)", nil)
		ran <- err
	}()
	s.T().Cleanup(func() {
		broker.release()
		s.NoError(<-ran, "the run itself")
	})
	select {
	case <-broker.entered:
	case err := <-ran:
		s.Require().NoError(err, "the run ended before it held the sandbox")
	}
	return box, broker
}

func (s *LifecycleSuite) run(box *Sandbox) {
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
}

// told waits for the meter to have been told want.
func (s *LifecycleSuite) told(want ...string) {
	s.Eventually(func() bool { return slices.Equal(s.meter.all(), want) }, 5*time.Second, 10*time.Millisecond,
		"the meter was told %v", s.meter.all())
}

func (s *LifecycleSuite) TestASandboxIsCreatedWithLabelsAndALifecycle() {
	labels := map[string]string{"stream.env": "test", "stream.router": "shard-a", "stream.organization": "org-1",
		"stream.app": "app-1", "stream.session": "session-1"}
	box := s.box(sandbox.Lifecycle{IdleStop: 15 * time.Minute, StoppedDelete: time.Hour}, labels)

	s.run(box)

	s.Require().Len(s.daytona.Creations(), 1)
	created := s.daytona.Creations()[0]
	s.EqualValues(15, created["autoStopInterval"], "Daytona stops it when the meter does")
	s.EqualValues(60, created["autoDeleteInterval"], "and deletes it an hour after")
	s.NotContains(created, "autoArchiveInterval", "it is deleted long before Daytona would archive it")
	held, ok := s.daytona.Sandbox("sandbox-1")
	s.Require().True(ok)
	for name, value := range labels {
		s.Equal(value, held.Labels[name], name)
	}
	s.Equal("python", held.Labels["code-toolbox-language"], "Daytona's own label is kept")
	s.Equal([]string{"created sandbox-1 1 vCPU 1 GiB 3 GiB container"}, s.meter.all(),
		"billed from the create, at the size Daytona says it made")
}

func (s *LifecycleSuite) TestALifecycleIsInWholeMinutes() {
	box := s.box(sandbox.Lifecycle{IdleStop: 90 * time.Second, StoppedDelete: time.Second}, nil)

	s.run(box)

	created := s.daytona.Creations()[0]
	s.EqualValues(2, created["autoStopInterval"], "rounded up, so Daytona never stops it before the meter does")
	s.EqualValues(1, created["autoDeleteInterval"])
}

func (s *LifecycleSuite) TestWithoutALifecycleDaytonasIdleStopIsKept() {
	box := s.box(sandbox.Lifecycle{}, nil)

	s.run(box)

	created := s.daytona.Creations()[0]
	s.EqualValues(15, created["autoStopInterval"])
	s.NotContains(created, "autoDeleteInterval", "Daytona never deletes a stopped sandbox on its own")
}

func (s *LifecycleSuite) TestABusySandboxIsNotReadBeforeEachRun() {
	box := s.box(sandbox.Lifecycle{IdleStop: time.Minute}, nil)

	for range 3 {
		s.run(box)
	}

	s.Equal([]string{"POST /api/sandbox"}, s.daytona.Calls(), "a sandbox in use is as it was left")
}

func (s *LifecycleSuite) TestAStoppedSandboxIsStartedAgain() {
	box := s.box(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, nil)
	s.run(box)
	s.daytona.Stop("sandbox-1")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")

	_, err := box.Run(s.ctx, "print(2)", nil)

	s.Require().NoError(err, "the next run after Daytona stopped it")
	s.Equal([]string{"sandbox-1: print(1)", "sandbox-1: print(2)"}, s.daytona.Ran(), "in the same sandbox, files and all")
	s.Contains(s.daytona.Calls(), "POST /api/sandbox/sandbox-1/start")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "started")
	generation, err := box.Ready(s.ctx)
	s.Require().NoError(err)
	s.Equal(1, generation, "the same sandbox")
}

func (s *LifecycleSuite) TestADeletedSandboxIsMadeAgain() {
	box := s.box(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, nil)
	s.run(box)
	s.daytona.Gone("sandbox-1")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")

	generation, err := box.Ready(s.ctx)

	s.Require().NoError(err)
	s.Equal(2, generation, "a new sandbox, holding nothing written into the last")
	s.run(box)
	s.Equal([]string{"sandbox-1: print(1)", "sandbox-2: print(1)"}, s.daytona.Ran())
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "gone", "created sandbox-2 1 vCPU 1 GiB 3 GiB container")
}

func (s *LifecycleSuite) TestASandboxDeletedWhileBusyIsMadeAgainAndTheRunRetried() {
	box := s.box(sandbox.Lifecycle{IdleStop: time.Minute}, nil)
	s.run(box)
	s.daytona.Gone("sandbox-1")

	ran, err := box.Run(s.ctx, "print(2)", nil)

	s.Require().NoError(err, "a sandbox someone else deleted while in use is made again")
	s.Equal("ran\n", ran.Output)
	s.Equal([]string{"sandbox-1: print(1)", "sandbox-2: print(2)"}, s.daytona.Ran(), "the run tried once more, in the new one")
	s.Equal([]string{"created sandbox-1 1 vCPU 1 GiB 3 GiB container", "gone", "created sandbox-2 1 vCPU 1 GiB 3 GiB container"},
		s.meter.all(), "the old sandbox's meter stopped")
	s.Contains(s.logs.String(), `msg="a sandbox in use was deleted by someone else; making another" sandbox=sandbox-1`)

	s.Require().NoError(box.Write(s.ctx, "data/in.csv", []byte("a,b")))
	s.daytona.Gone("sandbox-2")
	s.Require().NoError(box.Write(s.ctx, "data/in.csv", []byte("c,d")))
	s.Equal(map[string][]byte{"data/in.csv": []byte("c,d")}, s.daytona.Files("sandbox-3"), "a write is retried the same way")
	generation, err := box.Ready(s.ctx)
	s.Require().NoError(err)
	s.Equal(3, generation)
}

func (s *LifecycleSuite) TestASandboxInAFailedStateIsDeletedAndMadeAgain() {
	for _, state := range []string{"error", "build_failed", "unknown"} {
		s.SetupTest()
		box := s.box(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, nil)
		s.run(box)
		s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")
		s.daytona.SetState("sandbox-1", state)

		_, err := box.Run(s.ctx, "print(2)", nil)

		s.Require().NoError(err, state)
		s.Equal([]string{"sandbox-1"}, s.daytona.Deleted(), "%s: deleted, not started", state)
		s.NotContains(s.daytona.Calls(), "POST /api/sandbox/sandbox-1/start", state)
		s.Equal([]string{"sandbox-1: print(1)", "sandbox-2: print(2)"}, s.daytona.Ran(), state)
		s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "gone", "created sandbox-2 1 vCPU 1 GiB 3 GiB container")
		s.Contains(s.logs.String(), "deleted a sandbox in a state it cannot run code in; making another\" sandbox=sandbox-1 state="+state)
	}
}

func (s *LifecycleSuite) TestASandboxIsDeletedWhenTheRunItsCloseGaveUpOnEnds() {
	box, run := s.busy(sandbox.Lifecycle{IdleStop: time.Hour})

	s.ErrorIs(box.Close(), context.DeadlineExceeded, "the run still holds the sandbox")
	s.Empty(s.daytona.Deleted(), "nothing is deleted under a run")
	run.release()

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "gone")
	s.Equal([]string{"sandbox-1"}, s.daytona.Deleted(), "deleted as the run ended, long before it idled")
}

func (s *LifecycleSuite) TestASandboxNotDeletedWhenTheRunItsCloseGaveUpOnEndsIsDeletedOnceIdle() {
	box, run := s.busy(sandbox.Lifecycle{IdleStop: 300 * time.Millisecond, StoppedDelete: time.Hour})
	s.Error(box.Close())
	s.daytona.RefuseDeletes(true)
	run.release()

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "orphaned")
	s.daytona.RefuseDeletes(false)

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "orphaned", "gone")
	s.Equal([]string{"sandbox-1"}, s.daytona.Deleted(), "deleted once it idled, by its own timer")
	// The warning is written before the gate is let go, so by the timer's delete it is there.
	s.Contains(s.logs.String(), `msg="could not close a sandbox once the run its Close gave up on ended"`)
}

func (s *LifecycleSuite) TestAnOrphanIsDeletedOnceIdleWhenTheDeleteAfterItsRunOutlastedTheIdleStop() {
	deletes := &stallingDeletes{}
	broker := &stalled{entered: make(chan struct{}, 1), let: make(chan struct{})}
	// The delete after the run stalls for the whole 500 ms Timeout, past the 300 ms IdleStop, so
	// the timer the run armed fires while that delete holds the sandbox.
	box, err := New(Options{APIKey: "key-1", APIURL: s.daytona.APIURL(),
		Lifecycle: sandbox.Lifecycle{IdleStop: 300 * time.Millisecond, StoppedDelete: time.Hour}, Meter: s.meter, Broker: broker,
		Timeout: 500 * time.Millisecond, HTTPClient: &http.Client{Transport: deletes}, Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(err)
	ran := make(chan error, 1)
	go func() {
		_, err := box.Run(s.ctx, "print(1)", nil)
		ran <- err
	}()
	<-broker.entered

	s.Error(box.Close())
	deletes.stall.Store(true)
	broker.release()
	s.NoError(<-ran)

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "orphaned", "gone")
	s.Equal([]string{"sandbox-1"}, s.daytona.Deleted(), "deleted by its timer, armed again after the stalled delete")
}

// stallingDeletes passes requests to the test's Daytona, but holds the first delete after stall
// is set until its request gives up.
type stallingDeletes struct{ stall atomic.Bool }

func (t *stallingDeletes) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Method == http.MethodDelete && t.stall.CompareAndSwap(true, false) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	}
	return http.DefaultTransport.RoundTrip(r)
}

func (s *LifecycleSuite) TestASandboxIsDeletedWhenItsRunEndsAsCloseGivesUp() {
	box, run := s.busy(sandbox.Lifecycle{IdleStop: time.Hour})
	ending := time.AfterFunc(box.timeout, run.release)
	defer ending.Stop()

	_ = box.Close()

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "gone")
	s.Equal([]string{"sandbox-1"}, s.daytona.Deleted(), "deleted whether Close or the run's end got there first")
}

func (s *LifecycleSuite) TestAnIdleSandboxIsBilledAsStoppedThenGone() {
	box := s.box(sandbox.Lifecycle{IdleStop: 100 * time.Millisecond, StoppedDelete: 200 * time.Millisecond}, nil)

	s.run(box)

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "gone")
	held, _ := s.daytona.Sandbox("sandbox-1")
	s.Equal(daytonatest.Started, held.State, "the meter keeps Daytona's clock; it deletes nothing itself")
}

func (s *LifecycleSuite) TestASandboxDaytonaDidNotStopIsReported() {
	box := s.box(sandbox.Lifecycle{IdleStop: 100 * time.Millisecond}, nil)
	box.grace = 100 * time.Millisecond

	s.run(box)

	s.Eventually(func() bool { return strings.Contains(s.logs.String(), "Daytona has not stopped a sandbox") },
		5*time.Second, 10*time.Millisecond)
	s.Contains(s.logs.String(), `level=WARN msg="Daytona has not stopped a sandbox that is billed as stopped" sandbox=sandbox-1 billed_as=stopped state=started`)
}

func (s *LifecycleSuite) TestClosingDeletesTheSandboxAndEndsItsBill() {
	box := s.box(sandbox.Lifecycle{IdleStop: time.Minute}, nil)
	s.run(box)

	s.Require().NoError(box.Close())

	s.Equal([]string{"sandbox-1"}, s.daytona.Deleted())
	s.Equal([]string{"created sandbox-1 1 vCPU 1 GiB 3 GiB container", "gone"}, s.meter.all())
}

func (s *LifecycleSuite) TestASandboxDaytonaDeletedIsClosedWithoutComplaint() {
	box := s.box(sandbox.Lifecycle{IdleStop: 100 * time.Millisecond, StoppedDelete: 100 * time.Millisecond}, nil)
	s.run(box)
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "gone")
	s.daytona.Gone("sandbox-1")

	s.NoError(box.Close())
	s.Equal([]string{"created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "gone"}, s.meter.all(), "gone once")
}

func (s *LifecycleSuite) TestAnOrphanIsDeletedOnceIdle() {
	box := s.box(sandbox.Lifecycle{IdleStop: 300 * time.Millisecond, StoppedDelete: time.Hour}, nil)
	s.run(box)
	s.daytona.RefuseDeletes(true)

	s.Error(box.Close(), "Daytona would not delete it")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "orphaned")
	s.daytona.RefuseDeletes(false)

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "orphaned", "gone")
	s.Equal([]string{"sandbox-1"}, s.daytona.Deleted(), "deleted once it idled, by its own timer")
}

func (s *LifecycleSuite) TestAnOrphanDaytonaWillNotDeleteIsBilledUntilDaytonaDeletesIt() {
	box := s.box(sandbox.Lifecycle{IdleStop: 100 * time.Millisecond, StoppedDelete: 200 * time.Millisecond}, nil)
	s.run(box)
	s.daytona.RefuseDeletes(true)

	s.Error(box.Close())

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "orphaned", "stopped", "gone")
	s.Contains(s.logs.String(), "could not delete a sandbox whose session closed")
}
