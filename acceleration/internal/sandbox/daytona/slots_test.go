//go:build integration

package daytona

import (
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// slotsFor are an organization's slots on a real enforcer, counted in the suites' Redis, for at
// most most of its sandboxes at once, as its own sandbox limits say.
func (s *LifecycleSuite) slotsFor(most int) sandbox.Slots {
	_, sharing := policytest.SharedHolds(s.T())
	money := policytest.Start(s.T(), "daytona_slots", sharing)
	owner := money.Owner(s.T(), 1_000_000)
	money.Usage.SetLimits(s.T(), owner.Attribution.OrganizationID, usage.Limits{Sandbox: &usage.SandboxLimits{MaxConcurrent: &most}})
	var slots sandbox.Slots
	s.Require().Eventually(func() bool {
		var err error
		slots, err = money.Enforcer.Sandbox(s.ctx, owner, policy.SandboxAsk{})
		if err != nil || slots == nil {
			return false
		}
		var releases []func()
		defer func() {
			for _, release := range releases {
				release()
			}
		}()
		for range most {
			release, err := slots.Take(s.ctx, false)
			if err != nil {
				return false
			}
			releases = append(releases, release)
		}
		_, err = slots.Take(s.ctx, false)
		return err != nil
	}, 5*time.Second, 20*time.Millisecond, "the organization's sandbox limits are read")
	return slots
}

// held is how many of at most one slot are held: one when another is refused.
func (s *LifecycleSuite) held(slots sandbox.Slots) int {
	release, err := slots.Take(s.ctx, false)
	if err != nil {
		s.Require().ErrorIs(err, policy.ErrSandboxLimit)
		return 1
	}
	release()
	return 0
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
	slots := s.slotsFor(1)
	box := s.slotted(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, slots)
	s.Zero(s.held(slots), "nothing is held before code runs")

	s.run(box)
	s.Equal(1, s.held(slots))
	s.daytona.Stop("sandbox-1")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")
	s.Zero(s.held(slots), "given back once it stopped for idling")

	s.run(box)
	s.Equal(1, s.held(slots), "taken again to start again")
	s.Require().NoError(box.Close())
	s.Zero(s.held(slots), "given back once deleted")
}

func (s *LifecycleSuite) TestASandboxRefusedASlotIsNotMade() {
	box := s.slotted(sandbox.Lifecycle{}, s.slotsFor(0))

	_, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().ErrorIs(err, policy.ErrSandboxLimit, "the refusal is what the model is told")
	s.Empty(s.daytona.Creations(), "nothing is made, so nothing is billed")
	s.Empty(s.meter.all())
}

func (s *LifecycleSuite) TestAStoppedSandboxRefusedASlotStaysStopped() {
	slots := s.slotsFor(1)
	box := s.slotted(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, slots)
	s.run(box)
	s.daytona.Stop("sandbox-1")
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")
	elsewhere, err := slots.Take(s.ctx, false)
	s.Require().NoError(err, "another sandbox of the organization's starts")

	_, err = box.Run(s.ctx, "print(2)", nil)

	s.Require().ErrorIs(err, policy.ErrSandboxLimit)
	s.NotContains(s.daytona.Calls(), "POST /api/sandbox/sandbox-1/start")
	s.Equal([]string{"created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped"}, s.meter.all(), "billed as stopped still")
	elsewhere()

	s.run(box)
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "started")
}

func (s *LifecycleSuite) TestASandboxDaytonaHadNotStoppedRunsWhateverTheCount() {
	slots := s.slotsFor(1)
	box := s.slotted(sandbox.Lifecycle{IdleStop: 200 * time.Millisecond}, slots)
	s.run(box)
	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped")
	elsewhere, err := slots.Take(s.ctx, false)
	s.Require().NoError(err, "its slot was given back at the idle stop, and another sandbox took it")

	s.run(box)

	s.told("created sandbox-1 1 vCPU 1 GiB 3 GiB container", "stopped", "started")
	elsewhere()
	s.Equal(1, s.held(slots), "the one Daytona ran all along is counted")
}

func (s *LifecycleSuite) TestADeletedSandboxGivesItsSlotToTheOneMadeAgain() {
	slots := s.slotsFor(1)
	box := s.slotted(sandbox.Lifecycle{IdleStop: time.Minute}, slots)
	s.run(box)
	s.daytona.Gone("sandbox-1")

	s.run(box)

	s.Equal([]string{"sandbox-1: print(1)", "sandbox-2: print(1)"}, s.daytona.Ran(), "made again in the one slot")
	s.Equal(1, s.held(slots))
}
