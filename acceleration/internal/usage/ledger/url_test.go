package ledger_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
)

func TestAMalformedRedisURLIsRefusedWithoutItsPassword(t *testing.T) {
	for _, url := range []string{
		"redis://usage:s3cret@valkey:port",
		"redis://usage:12/s3cret@valkey:6379",
		"redis://usage:s3cret@valkey:6379/notadb",
	} {
		_, err := ledger.NewMirror(ledger.MirrorOptions{URL: url})

		require.Error(t, err, url)
		require.NotContains(t, err.Error(), "s3cret", url)
		require.Contains(t, err.Error(), "the Redis URL does not parse", url)
	}
}
