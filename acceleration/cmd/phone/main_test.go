package main

import (
	"bytes"
	"crypto/rand"
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
