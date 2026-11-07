package main

import (
	"bytes"
	"crypto/rand"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/getsentry/sentry-go"
	"github.com/stretchr/testify/require"
)

// The service's Sentry events carry a failure's text: a key in it is taken out of the
// exception, and the reason is kept.
func TestTheServicesSentryEventsKeepTheReasonAndNotTheKey(t *testing.T) {
	key := rand.Text()
	options := sentryOptions()
	require.NotNil(t, options.BeforeSend, "the service's events are sent as they are")
	var sent []*sentry.Event
	redact := options.BeforeSend
	options.BeforeSend = func(event *sentry.Event, hint *sentry.EventHint) *sentry.Event {
		sent = append(sent, redact(event, hint))
		return nil
	}
	client, err := sentry.NewClient(options)
	require.NoError(t, err)
	sentry.NewHub(client, sentry.NewScope()).CaptureException(errors.New("ledger: refused: Authorization: Bearer sk-test-" + key))

	require.Len(t, sent, 1)
	require.Equal(t, "ledger: refused: [redacted]", sent[0].Exception[len(sent[0].Exception)-1].Value)
}

// The logger main builds takes a key out of what it logs: openapi writing to a path that names
// one fails, and the failure is logged without it.
func TestTheServicesLogTakesTheKeyOutOfAFailure(t *testing.T) {
	if path := os.Getenv("USAGE_MAIN_TEST_PATH"); path != "" {
		os.Args = []string{"usage", "openapi", path}
		main()
		return
	}
	key := rand.Text()
	path := filepath.Join(t.TempDir(), "missing", "Bearer sk-test-"+key)
	command := exec.Command(os.Args[0], "-test.run=^TestTheServicesLogTakesTheKeyOutOfAFailure$")
	command.Env = append(os.Environ(), "USAGE_MAIN_TEST_PATH="+path)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	require.Error(t, command.Run(), "openapi wrote to a directory that does not exist")

	require.NotContains(t, stderr.String(), key)
	require.Contains(t, stderr.String(), `msg="usage stopped" error="open `+strings.TrimSuffix(filepath.Dir(path), "/")+`/[redacted] no such file or directory"`)
}
