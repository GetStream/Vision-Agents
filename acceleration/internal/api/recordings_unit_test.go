package api

import (
	"crypto/rand"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// An inline job is answered in the response and never stored, and its error is a provider's
// text, which can echo the key it was sent: the caller is told it without the key. One with
// no key is told as it was.
func TestInlineResultTellsAFailureWithoutTheKeyItEchoed(t *testing.T) {
	key := "sk-test-" + rand.Text()
	for failure, told := range map[string]string{
		"deepgram: 401 Unauthorized\nAuthorization: Bearer " + key: "deepgram: 401 Unauthorized\n[redacted]",
		"deepgram: 503 Service Unavailable":                        "deepgram: 503 Service Unavailable",
		strings.Repeat("x", 9000):                                  strings.Repeat("x", 2048) + "… [truncated]",
	} {
		job := inlineResult(store.Recording{}, "deepgram", "nova", nil, errors.New(failure))
		require.Equal(t, "failed", job.Status)
		require.Equal(t, told, job.Error)
	}
}
