package imagejobs

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// A picture already drawn waits out an outage of Stream Chat of a couple of hours rather than
// being given up after minutes: drawing it again would bill it twice.
func TestAPostIsTriedAgainOverHoursBeforeItIsGivenUp(t *testing.T) {
	var total time.Duration
	for _, wait := range defaultPostBackoff {
		total += wait
	}

	require.GreaterOrEqual(t, total, 2*time.Hour)
}
