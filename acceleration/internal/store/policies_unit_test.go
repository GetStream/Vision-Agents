package store

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestABudgetWindowStartsOnAUTCBoundary(t *testing.T) {
	// A Wednesday afternoon.
	now := time.Date(2026, 9, 23, 15, 42, 7, 0, time.UTC)

	cases := map[BudgetInterval][2]time.Time{
		BudgetHourly:  {time.Date(2026, 9, 23, 15, 0, 0, 0, time.UTC), time.Date(2026, 9, 23, 16, 0, 0, 0, time.UTC)},
		BudgetDaily:   {time.Date(2026, 9, 23, 0, 0, 0, 0, time.UTC), time.Date(2026, 9, 24, 0, 0, 0, 0, time.UTC)},
		BudgetWeekly:  {time.Date(2026, 9, 21, 0, 0, 0, 0, time.UTC), time.Date(2026, 9, 28, 0, 0, 0, 0, time.UTC)},
		BudgetMonthly: {time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC), time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)},
	}
	for interval, want := range cases {
		start, end := interval.Window(now)
		require.Equal(t, want[0], start, interval)
		require.Equal(t, want[1], end, interval)
	}
}

func TestAWeeklyBudgetOnASundayBelongsToTheWeekThatStartedMonday(t *testing.T) {
	start, _ := BudgetWeekly.Window(time.Date(2026, 9, 27, 23, 0, 0, 0, time.UTC))

	require.Equal(t, time.Date(2026, 9, 21, 0, 0, 0, 0, time.UTC), start)
}
