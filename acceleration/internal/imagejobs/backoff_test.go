package imagejobs

import (
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
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

// A write Stream Chat may take on a later try is tried again: a 408 is its RequestTimeoutError,
// and a message refused as already written, after reading it back failed, is the post that got
// through. Given up, a drawn picture would be thrown away or the person told nothing.
func TestOnlyAStreamChatRefusalALaterTryGetsTooIsPermanent(t *testing.T) {
	refusal := func(status, code int, message string) error {
		return fmt.Errorf("conversation: post: %w", &getstream.StreamError{StatusCode: status, Code: code, Message: message})
	}

	require.False(t, permanent(refusal(http.StatusRequestTimeout, 23, "request timed out")))
	require.False(t, permanent(refusal(http.StatusTooManyRequests, 9, "too many requests")))
	require.False(t, permanent(refusal(http.StatusBadRequest, 4, "SendMessage failed with error: \"a message with ID job-1 already exists\"")))
	require.False(t, permanent(refusal(http.StatusBadGateway, 0, "bad gateway")))
	require.False(t, permanent(errors.New("dial tcp: connection refused")))
	require.True(t, permanent(refusal(http.StatusForbidden, 17, "the channel is frozen")))
	require.True(t, permanent(refusal(http.StatusBadRequest, 4, "the file is too large")))
}
