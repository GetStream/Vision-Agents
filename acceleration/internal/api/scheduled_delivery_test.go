package api

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestAPeerIsGivenLongEnoughToRingAndHoldATasksCall(t *testing.T) {
	// The ring, the task sent while a reply is waited out, and the call held
	// to its limit: 90 s + 10 min + 10 min.
	require.Equal(t, 21*time.Minute+30*time.Second, scheduledPeerWindow())
	require.Greater(t, scheduledPeerWindow(), scheduledPhoneWindow+scheduledCallLimit+time.Minute,
		"a call answered at the last ring and held to its limit is not cut by the asker")
}
