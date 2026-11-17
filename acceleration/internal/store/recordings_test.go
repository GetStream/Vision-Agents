//go:build integration

package store

import (
	"crypto/rand"
	"errors"
	"strings"
)

// A failed job's error is a provider's text, which can echo the key it was sent, and the
// caller reads it back: it is kept without the key, and cut as the requests table cuts an
// error. One with no key is kept as it was.
func (s *StoreSuite) TestAFailedRecordingsErrorIsKeptWithoutTheKeyItEchoed() {
	key := "sk-test-" + rand.Text()
	for failure, kept := range map[string]string{
		"deepgram: 401 Unauthorized\nAuthorization: Bearer " + key: "deepgram: 401 Unauthorized\n[redacted]",
		"deepgram: 503 Service Unavailable":                        "deepgram: 503 Service Unavailable",
		strings.Repeat("x", 9000):                                  strings.Repeat("x", 2048) + "… [truncated]",
	} {
		job := Recording{CustomerID: "acme", Modality: "stt"}
		s.Require().NoError(s.store.CreateRecording(s.ctx, &job))

		s.Require().NoError(s.store.FinishRecording(s.ctx, job.ID, "deepgram", "nova", nil, errors.New(failure)))

		found, err := s.store.Recording(s.ctx, "acme", job.ID)
		s.Require().NoError(err)
		s.Equal(RecordingFailed, found.Status)
		s.Equal(kept, found.Error)
	}
}
