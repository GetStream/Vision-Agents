package main

import (
	"bytes"
	"crypto/rand"
	"errors"
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/require"
)

// What main prints for a failure has a key taken out of it: a command that names one is
// unknown, and the refusal is printed without it.
func TestTheFailureIsPrintedWithoutTheKeyInIt(t *testing.T) {
	if command := os.Getenv("PHONE_MAIN_TEST_COMMAND"); command != "" {
		os.Args = []string{"phone", command}
		main()
		return
	}
	key := rand.Text()
	command := exec.Command(os.Args[0], "-test.run=^TestTheFailureIsPrintedWithoutTheKeyInIt$")
	command.Env = append(os.Environ(), "PHONE_MAIN_TEST_COMMAND=Bearer sk-test-"+key)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	require.Error(t, command.Run(), "phone ran a command it does not know")

	require.NotContains(t, stderr.String(), key)
	require.Contains(t, stderr.String(), `error: unknown command "[redacted]"`)
}

// The command's logger keeps a vendor's reason and not the key it echoed, and logs a warning.
func TestTheLoggerKeepsAVendorsReasonAndNotTheKeyItEchoed(t *testing.T) {
	key := rand.Text()
	var out bytes.Buffer
	logger := newLogger(&out)
	logger.Info("below the level")
	logger.Warn("vendor search failed", "error", errors.New("502 Bad Gateway: request headers: Authorization: Bearer sk-test-"+key))

	require.NotContains(t, out.String(), key)
	require.NotContains(t, out.String(), "below the level")
	require.Contains(t, out.String(), `msg="vendor search failed" error="502 Bad Gateway: request headers: [redacted]"`)
}
