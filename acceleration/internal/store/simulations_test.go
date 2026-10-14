//go:build integration

package store

import (
	"time"
)

func (s *StoreSuite) TestARunItsRouterStillRenewsIsNotWrittenOff() {
	// However long ago it started: a second router booting must not abandon what the first
	// is still having.
	run := s.startedRun("app", time.Now().Add(time.Minute))

	s.Require().NoError(s.store.AbandonSimulationRuns(s.ctx, time.Now()))

	found, err := s.store.SimulationRun(s.ctx, "app", run.ID)
	s.Require().NoError(err)
	s.Equal(SimulationRunning, found.State)
}

func (s *StoreSuite) TestARunWhoseRouterStoppedRenewingIsWrittenOff() {
	run := s.startedRun("app", time.Now().Add(-time.Second))

	s.Require().NoError(s.store.AbandonSimulationRuns(s.ctx, time.Now()))

	found, err := s.store.SimulationRun(s.ctx, "app", run.ID)
	s.Require().NoError(err)
	s.Equal(SimulationErrored, found.State)
	cases, err := s.store.SimulationCases(s.ctx, run.ID)
	s.Require().NoError(err)
	s.Equal(SimulationCancelled, cases[0].State, "nobody is going to have the conversation")
}

func (s *StoreSuite) TestRenewingARunKeepsItFromBeingWrittenOff() {
	run := s.startedRun("app", time.Now().Add(time.Second))

	stop, err := s.store.RenewSimulationRuns(s.ctx, []string{run.ID}, time.Now().Add(time.Hour))
	s.Require().NoError(err)
	s.Empty(stop)
	s.Require().NoError(s.store.AbandonSimulationRuns(s.ctx, time.Now().Add(time.Minute)))

	found, err := s.store.SimulationRun(s.ctx, "app", run.ID)
	s.Require().NoError(err)
	s.Equal(SimulationRunning, found.State)
}

func (s *StoreSuite) TestARunAskedToStopIsNotRenewedAndIsToldToStop() {
	run := s.startedRun("app", time.Now().Add(time.Minute))
	s.Require().NoError(s.store.StopSimulationRun(s.ctx, "app", run.ID))

	stop, err := s.store.RenewSimulationRuns(s.ctx, []string{run.ID}, time.Now().Add(time.Hour))
	s.Require().NoError(err)

	s.Equal([]string{run.ID}, stop)
}

func (s *StoreSuite) TestAFinishedRunIsToldToStop() {
	run := s.startedRun("app", time.Now().Add(time.Minute))
	run.State = SimulationPassed
	s.Require().NoError(s.store.FinishSimulationRun(s.ctx, run))

	stop, err := s.store.RenewSimulationRuns(s.ctx, []string{run.ID}, time.Now().Add(time.Hour))
	s.Require().NoError(err)

	s.Equal([]string{run.ID}, stop)
}

func (s *StoreSuite) TestAnotherCustomerCannotStopARun() {
	run := s.startedRun("app", time.Now().Add(time.Minute))
	s.Require().NoError(s.store.StopSimulationRun(s.ctx, "somebody-else", run.ID))

	stop, err := s.store.RenewSimulationRuns(s.ctx, []string{run.ID}, time.Now().Add(time.Hour))
	s.Require().NoError(err)

	s.Empty(stop)
}

// startedRun starts a run of one conversation for a customer, held until a given time.
func (s *StoreSuite) startedRun(customerID string, heldUntil time.Time) SimulationRun {
	simulation := Simulation{
		CustomerID: customerID, Name: "order status", Mode: SimulationText,
		ConfigID: "config", Scenario: "ask where the order is", Assertion: "it says where",
	}
	s.Require().NoError(s.store.CreateSimulation(s.ctx, &simulation))
	run := SimulationRun{CustomerID: customerID, SimulationID: simulation.ID, Mode: SimulationText}
	s.Require().NoError(s.store.StartSimulationRun(s.ctx, &run,
		[]SimulationCase{{Scenario: simulation.Scenario}}, heldUntil))
	return run
}
