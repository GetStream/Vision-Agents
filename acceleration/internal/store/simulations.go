package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

// defaultMaxTurns is how long a conversation a simulation that did not say runs for. A
// scenario with three things to ask needs more turns than three, because the agent asks
// questions back.
const defaultMaxTurns = 12

// maxTurns and maxVariations bound what a customer may ask for, since every turn is a
// model call and every variation is a whole conversation.
const (
	maxTurns      = 30
	maxVariations = 10
)

// defaultRunLimit is how many runs are listed when nobody said.
const defaultRunLimit = 50

// CreateSimulation stores a simulation and fills in its id and timestamps.
func (s *Store) CreateSimulation(ctx context.Context, simulation *Simulation) error {
	if err := checkSimulation(simulation); err != nil {
		return err
	}

	simulation.ID = newID()
	now := time.Now().UTC()
	simulation.CreatedAt = now
	simulation.UpdatedAt = now
	simulation.DeletedAt = nil
	normalizeSimulation(simulation)

	if _, err := s.db.NewInsert().Model(simulation).Exec(ctx); err != nil {
		return fmt.Errorf("store: create simulation: %w", err)
	}
	return nil
}

// UpdateSimulation replaces what a simulation asks. The runs it already has are left
// alone: they carry their own copy of what they tested.
func (s *Store) UpdateSimulation(ctx context.Context, simulation *Simulation) error {
	if simulation.ID == "" {
		return errors.New("store: a simulation id is required")
	}
	if err := checkSimulation(simulation); err != nil {
		return err
	}

	simulation.UpdatedAt = time.Now().UTC()
	normalizeSimulation(simulation)

	result, err := s.db.NewUpdate().Model(simulation).
		Column("name", "mode", "config_id", "scenario", "assertion", "variations",
			"judge_target", "caller_target", "caller_tts", "caller_stt", "caller_voice",
			"max_turns", "tags", "updated_at").
		Where("id = ?", simulation.ID).
		Where("customer_id = ?", simulation.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: update simulation: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: update simulation: %w", err)
	}
	if affected == 0 {
		return unknownSimulation(simulation.ID)
	}
	return nil
}

// DeleteSimulation marks a simulation as gone. The row stays, because the runs that named
// it are still worth reading.
func (s *Store) DeleteSimulation(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a simulation id are required")
	}

	result, err := s.db.NewUpdate().Model((*Simulation)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete simulation: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete simulation: %w", err)
	}
	if affected == 0 {
		return unknownSimulation(id)
	}
	return nil
}

// Simulation returns one simulation a customer holds.
func (s *Store) Simulation(ctx context.Context, customerID, id string) (Simulation, error) {
	if customerID == "" || id == "" {
		return Simulation{}, errors.New("store: a customer and a simulation id are required")
	}

	var simulation Simulation
	err := s.db.NewSelect().Model(&simulation).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Simulation{}, unknownSimulation(id)
	}
	if err != nil {
		return Simulation{}, fmt.Errorf("store: simulation: %w", err)
	}
	return simulation, nil
}

// CustomerSimulations returns a customer's simulations, newest first.
func (s *Store) CustomerSimulations(ctx context.Context, customerID string) ([]Simulation, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var simulations []Simulation
	err := s.db.NewSelect().Model(&simulations).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: customer simulations: %w", err)
	}
	return simulations, nil
}

// StartSimulationRun writes a run and all of its conversations at once, so a run that has
// only just begun already shows how many ways it is going to ask.
func (s *Store) StartSimulationRun(ctx context.Context, run *SimulationRun, cases []SimulationCase) error {
	if run.CustomerID == "" || run.SimulationID == "" {
		return errors.New("store: a customer and a simulation id are required")
	}
	if len(cases) == 0 {
		return errors.New("store: a run needs at least one conversation to have")
	}

	run.ID = newID()
	run.State = SimulationRunning
	run.Cases = len(cases)
	run.Passed = 0
	run.Failed = 0
	run.StartedAt = time.Now().UTC()
	run.FinishedAt = nil

	for i := range cases {
		cases[i].ID = newID()
		cases[i].RunID = run.ID
		cases[i].State = Pending
		cases[i].StartedAt = run.StartedAt
		if cases[i].Transcript == nil {
			cases[i].Transcript = []SimulationLine{}
		}
	}

	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.NewInsert().Model(run).Exec(ctx); err != nil {
			return err
		}
		_, err := tx.NewInsert().Model(&cases).Exec(ctx)
		return err
	})
	if err != nil {
		return fmt.Errorf("store: start simulation run: %w", err)
	}
	return nil
}

// StartSimulationCase records which session is holding a conversation, which is what makes
// a run in progress watchable.
func (s *Store) StartSimulationCase(ctx context.Context, id, callID string) error {
	if id == "" {
		return errors.New("store: a case id is required")
	}

	_, err := s.db.NewUpdate().Model((*SimulationCase)(nil)).
		Set("state = ?", SimulationRunning).
		Set("call_id = ?", callID).
		Where("id = ?", id).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: start simulation case: %w", err)
	}
	return nil
}

// FinishSimulationCase records how one conversation went and what the judge made of it.
func (s *Store) FinishSimulationCase(ctx context.Context, kase SimulationCase) error {
	if kase.ID == "" {
		return errors.New("store: a case id is required")
	}
	if kase.Transcript == nil {
		kase.Transcript = []SimulationLine{}
	}

	now := time.Now().UTC()
	kase.FinishedAt = &now

	_, err := s.db.NewUpdate().Model(&kase).
		Column("state", "call_id", "transcript", "turns", "passed", "verdict", "score",
			"ended", "error", "finished_at").
		Where("id = ?", kase.ID).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish simulation case: %w", err)
	}
	return nil
}

// FinishSimulationRun records how a run ended and what its conversations came to.
func (s *Store) FinishSimulationRun(ctx context.Context, run SimulationRun) error {
	if run.ID == "" {
		return errors.New("store: a run id is required")
	}

	now := time.Now().UTC()
	run.FinishedAt = &now

	_, err := s.db.NewUpdate().Model(&run).
		Column("state", "passed", "failed", "error", "finished_at").
		Where("id = ?", run.ID).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: finish simulation run: %w", err)
	}
	return nil
}

// SimulationRun returns one run a customer holds.
func (s *Store) SimulationRun(ctx context.Context, customerID, id string) (SimulationRun, error) {
	if customerID == "" || id == "" {
		return SimulationRun{}, errors.New("store: a customer and a run id are required")
	}

	var run SimulationRun
	err := s.db.NewSelect().Model(&run).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return SimulationRun{}, unknownSimulationRun(id)
	}
	if err != nil {
		return SimulationRun{}, fmt.Errorf("store: simulation run: %w", err)
	}
	return run, nil
}

// SimulationRuns lists runs, newest first. It answers both what one simulation has come to
// and what every simulation has come to lately, which is the same question asked twice.
func (s *Store) SimulationRuns(ctx context.Context, filter SimulationRunFilter) ([]SimulationRun, error) {
	if filter.CustomerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	limit := filter.Limit
	if limit <= 0 {
		limit = defaultRunLimit
	}

	var runs []SimulationRun
	query := s.db.NewSelect().Model(&runs).
		Where("customer_id = ?", filter.CustomerID)
	if filter.SimulationID != "" {
		query = query.Where("simulation_id = ?", filter.SimulationID)
	}
	if filter.State != "" {
		query = query.Where("state = ?", filter.State)
	}

	if err := query.Order("started_at DESC").Limit(limit).Scan(ctx); err != nil {
		return nil, fmt.Errorf("store: simulation runs: %w", err)
	}
	return runs, nil
}

// SimulationCases returns a run's conversations, in the order they were asked.
func (s *Store) SimulationCases(ctx context.Context, runID string) ([]SimulationCase, error) {
	if runID == "" {
		return nil, errors.New("store: a run id is required")
	}

	var cases []SimulationCase
	err := s.db.NewSelect().Model(&cases).
		Where("run_id = ?", runID).
		Order("variation ASC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: simulation cases: %w", err)
	}
	return cases, nil
}

// AbandonSimulationRuns marks as errored the runs an older process left saying they were
// running. Nothing else will finish them: the conversations were held in that process, and
// it is gone.
func (s *Store) AbandonSimulationRuns(ctx context.Context, before time.Time) error {
	now := time.Now().UTC()

	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		_, err := tx.NewUpdate().Model((*SimulationCase)(nil)).
			Set("state = ?", SimulationCancelled).
			Set("finished_at = ?", now).
			Where("state IN (?)", bun.In([]string{Pending, SimulationRunning})).
			Where("run_id IN (SELECT id FROM simulation_runs WHERE state = ? AND started_at < ?)",
				SimulationRunning, before).
			Exec(ctx)
		if err != nil {
			return err
		}

		_, err = tx.NewUpdate().Model((*SimulationRun)(nil)).
			Set("state = ?", SimulationErrored).
			Set("error = ?", "the router restarted while this run was still going").
			Set("finished_at = ?", now).
			Where("state = ?", SimulationRunning).
			Where("started_at < ?", before).
			Exec(ctx)
		return err
	})
	if err != nil {
		return fmt.Errorf("store: abandon simulation runs: %w", err)
	}
	return nil
}

// checkSimulation refuses a simulation that could not be run, so the complaint arrives when
// it is written rather than once per conversation at whatever hour it was started.
func checkSimulation(simulation *Simulation) error {
	switch {
	case simulation.CustomerID == "":
		return errors.New("store: customer id is required")
	case simulation.Name == "":
		return errors.New("store: a simulation needs a name")
	case simulation.ConfigID == "":
		return errors.New("store: a simulation needs an agent to run against")
	case simulation.Scenario == "":
		return errors.New("store: a simulation needs something to ask")
	case simulation.Assertion == "":
		return errors.New("store: a simulation needs something to check")
	case simulation.Mode != SimulationText && simulation.Mode != SimulationAudio:
		return fmt.Errorf("store: a simulation is %s or %s", SimulationText, SimulationAudio)
	case simulation.Variations > maxVariations:
		return fmt.Errorf("store: a simulation may try at most %d ways of asking", maxVariations)
	case simulation.MaxTurns > maxTurns:
		return fmt.Errorf("store: a conversation may run at most %d turns", maxTurns)
	}
	return nil
}

// normalizeSimulation fills in what was left out, so nothing downstream has to decide what
// an empty column meant.
func normalizeSimulation(simulation *Simulation) {
	if simulation.Variations <= 0 {
		simulation.Variations = 1
	}
	if simulation.MaxTurns <= 0 {
		simulation.MaxTurns = defaultMaxTurns
	}
	if simulation.Tags == nil {
		simulation.Tags = map[string]string{}
	}
}

func unknownSimulation(id string) error {
	return fmt.Errorf("store: there is no simulation %s", id)
}

func unknownSimulationRun(id string) error {
	return fmt.Errorf("store: there is no simulation run %s", id)
}
