package store

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestASimulationIsBoundedInVariationsAndTurns(t *testing.T) {
	simulation := func(variations, turns int) *Simulation {
		return &Simulation{
			CustomerID: "acme",
			Name:       "refund",
			ConfigID:   "support",
			Scenario:   "ask for a refund",
			Assertion:  "the agent offers one",
			Mode:       SimulationText,
			Variations: variations,
			MaxTurns:   turns,
		}
	}

	require.NoError(t, checkSimulation(simulation(10, 200)))
	require.Error(t, checkSimulation(simulation(11, 12)))
	require.Error(t, checkSimulation(simulation(1, 201)))
}
