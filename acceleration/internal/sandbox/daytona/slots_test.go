package daytona

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// errNoSlot is a slot refused because most sandboxes are running already.
var errNoSlot = errors.New("most sandboxes are running already")

// counted is slots for at most most sandboxes, counted in process.
type counted struct {
	mu      sync.Mutex
	most    int
	running int
}

func (c *counted) Take(context.Context) (func(), error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.running >= c.most {
		return nil, errNoSlot
	}
	c.running++
	var once sync.Once
	return func() {
		once.Do(func() {
			c.mu.Lock()
			defer c.mu.Unlock()
			c.running--
		})
	}, nil
}

func (c *counted) held() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.running
}

// slotted is a sandbox on the test's Daytona, metered by the test, holding slots.
func (s *LifecycleSuite) slotted(lifecycle sandbox.Lifecycle, slots sandbox.Slots) *Sandbox {
	box, err := New(Options{APIKey: "key-1", APIURL: s.daytona.APIURL(), Lifecycle: lifecycle, Meter: s.meter, Slots: slots,
		Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

func (s *LifecycleSuite) TestASandboxHoldsASlotOnlyWhileItRuns() {
	slots := &counted{most: 1}
	box := s.slotted(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, slots)
	s.Zero(slots.held(), "nothing is held before code runs")

	s.run(box)
	s.Equal(1, slots.held())
	s.daytona.Stop("sandbox-1")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")
	s.Zero(slots.held(), "given back once it stopped for idling")

	s.run(box)
	s.Equal(1, slots.held(), "taken again to start again")
	s.Require().NoError(box.Close())
	s.Zero(slots.held(), "given back once deleted")
}

func (s *LifecycleSuite) TestASandboxRefusedASlotIsNotMade() {
	box := s.slotted(sandbox.Lifecycle{}, &counted{most: 0})

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().ErrorIs(err, errNoSlot, "the refusal is what the model is told")
	s.Empty(s.daytona.Creations(), "nothing is made, so nothing is billed")
	s.Empty(s.meter.all())
}

func (s *LifecycleSuite) TestAStoppedSandboxRefusedASlotStaysStopped() {
	slots := &counted{most: 1}
	box := s.slotted(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, slots)
	s.run(box)
	s.daytona.Stop("sandbox-1")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")
	elsewhere, err := slots.Take(s.ctx)
	s.Require().NoError(err, "another sandbox of the organization's starts")

	_, err = box.Run(s.ctx, "print(2)", nil)

	s.Require().ErrorIs(err, errNoSlot)
	s.NotContains(s.daytona.Calls(), "POST /api/sandbox/sandbox-1/start")
	s.Equal([]string{"created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped"}, s.meter.all(), "billed as stopped still")
	elsewhere()

	s.run(box)
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "started")
}

func (s *LifecycleSuite) TestADeletedSandboxGivesItsSlotToTheOneMadeAgain() {
	slots := &counted{most: 1}
	box := s.slotted(sandbox.Lifecycle{IdleStop: time.Minute}, slots)
	s.run(box)
	s.daytona.Gone("sandbox-1")

	s.run(box)

	s.Equal([]string{"sandbox-1: print(1)", "sandbox-2: print(1)"}, s.daytona.Ran(), "made again in the one slot")
	s.Equal(1, slots.held())
}
