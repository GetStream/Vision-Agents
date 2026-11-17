package main

import (
	"bytes"
	"crypto/rand"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// The logger main builds takes a key out of a failure: a ROUTER_CONFIG path that names one is
// not there, and the failure is logged without it.
func TestTheFailureIsLoggedWithoutTheKeyInIt(t *testing.T) {
	if os.Getenv("CHAT_MAIN_TEST") == "1" {
		os.Args = []string{"chat", "-text", "hello"}
		main()
		return
	}
	key := rand.Text()
	dir := t.TempDir()
	command := exec.Command(os.Args[0], "-test.run=^TestTheFailureIsLoggedWithoutTheKeyInIt$")
	command.Env = append(os.Environ(), "CHAT_MAIN_TEST=1", "ROUTER_CONFIG="+filepath.Join(dir, "Bearer sk-test-"+key))
	var stderr bytes.Buffer
	command.Stderr = &stderr
	require.Error(t, command.Run(), "chat read a config that does not exist")

	require.NotContains(t, stderr.String(), key)
	require.Contains(t, stderr.String(), `msg="chat stopped" error="routing: read config `+dir+`/[redacted] open `+dir+`/[redacted] no such file or directory"`)
}
