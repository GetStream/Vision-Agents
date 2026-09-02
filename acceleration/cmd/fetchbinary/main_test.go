package main

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// sha is the checksum a sidecar would carry for this content.
func sha(content string) string {
	sum := sha256.Sum256([]byte(content))
	return hex.EncodeToString(sum[:])
}

func TestParseChecksum(t *testing.T) {
	want := sha("router")

	t.Run("reads a sidecar written by sha256sum", func(t *testing.T) {
		got, err := parseChecksum([]byte(want + "  router-linux-amd64\n"))
		require.NoError(t, err)
		require.Equal(t, want, got)
	})

	t.Run("reads a sidecar that is only the hash", func(t *testing.T) {
		got, err := parseChecksum([]byte(want + "\n"))
		require.NoError(t, err)
		require.Equal(t, want, got)
	})

	t.Run("refuses anything that is not a sha256", func(t *testing.T) {
		for _, body := range []string{"", "not-a-hash", want[:16], want + "ff", strings.ToUpper("zz") + want[2:]} {
			_, err := parseChecksum([]byte(body))
			require.Error(t, err, body)
		}
	})
}

func TestInstall(t *testing.T) {
	t.Run("writes the binary when the hash matches", func(t *testing.T) {
		dest := filepath.Join(t.TempDir(), "bin", "router")

		require.NoError(t, install(strings.NewReader("router"), dest, sha("router"), 0o755))

		content, err := os.ReadFile(dest)
		require.NoError(t, err)
		require.Equal(t, "router", string(content))

		info, err := os.Stat(dest)
		require.NoError(t, err)
		require.Equal(t, os.FileMode(0o755), info.Mode().Perm())
	})

	t.Run("leaves nothing behind when the hash does not match", func(t *testing.T) {
		// A corrupt download that landed at the destination anyway would be a crashloop
		// with no explanation, so the rename has to be the last thing that happens.
		dir := t.TempDir()
		dest := filepath.Join(dir, "router")

		err := install(strings.NewReader("tampered"), dest, sha("router"), 0o755)
		require.Error(t, err)

		_, err = os.Stat(dest)
		require.ErrorIs(t, err, os.ErrNotExist)

		left, err := os.ReadDir(dir)
		require.NoError(t, err)
		require.Empty(t, left, "the temporary file should have been cleaned up")
	})

	t.Run("does not disturb what is already installed when the hash does not match", func(t *testing.T) {
		dest := filepath.Join(t.TempDir(), "router")
		require.NoError(t, os.WriteFile(dest, []byte("the previous release"), 0o755))

		require.Error(t, install(strings.NewReader("tampered"), dest, sha("router"), 0o755))

		content, err := os.ReadFile(dest)
		require.NoError(t, err)
		require.Equal(t, "the previous release", string(content))
	})

	t.Run("replaces an older binary at the same path", func(t *testing.T) {
		dest := filepath.Join(t.TempDir(), "router")
		require.NoError(t, os.WriteFile(dest, []byte("the previous release"), 0o755))

		require.NoError(t, install(strings.NewReader("router"), dest, sha("router"), 0o755))

		content, err := os.ReadFile(dest)
		require.NoError(t, err)
		require.Equal(t, "router", string(content))
	})
}

func TestHashOf(t *testing.T) {
	t.Run("is the hash a matching sidecar would carry", func(t *testing.T) {
		dest := filepath.Join(t.TempDir(), "router")
		require.NoError(t, os.WriteFile(dest, []byte("router"), 0o755))

		got, err := hashOf(dest)
		require.NoError(t, err)
		require.Equal(t, sha("router"), got)
	})

	t.Run("says so when there is nothing installed", func(t *testing.T) {
		_, err := hashOf(filepath.Join(t.TempDir(), "absent"))
		require.Error(t, err)
	})
}

func TestRunNeedsEnoughToActOn(t *testing.T) {
	// These are checked before any S3 client is built, so a misconfigured initContainer
	// fails immediately rather than after a timeout against a bucket it cannot name.
	for name, opts := range map[string]options{
		"no binary":  {Dest: "/opt/stream/bin/router", Version: "v0.1.0"},
		"no dest":    {Binary: "router", Version: "v0.1.0"},
		"no version": {Binary: "router", Dest: "/opt/stream/bin/router"},
	} {
		err := run(t.Context(), slogDiscard(), opts)
		require.Error(t, err, name)
	}
}

// slogDiscard is a logger that writes nowhere, so a failing case does not print.
func slogDiscard() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
