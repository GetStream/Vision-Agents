package activity

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Defect: a recorder with nowhere to write, which would hold every request until it panics on
// its first write.
func TestARecorderNeedsAStore(t *testing.T) {
	_, err := New(Options{})

	require.Error(t, err)
}

// Defect: a changed default, which every test overriding the interval and capacity would let
// pass unseen.
func TestAnUnconfiguredRecorderWritesEveryMinuteAndHoldsTenThousandRows(t *testing.T) {
	recorder, err := New(Options{Store: &store.Store{}})
	require.NoError(t, err)

	require.Equal(t, DefaultInterval, recorder.interval)
	require.Equal(t, DefaultCapacity, recorder.capacity)
	require.Equal(t, 60_000_000_000, int(DefaultInterval))
	require.Equal(t, 10_000, DefaultCapacity)
}
