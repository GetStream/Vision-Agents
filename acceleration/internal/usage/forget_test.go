package usage

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestANewReplacementIsRandomAndRecognised(t *testing.T) {
	first, second := NewReplacement(), NewReplacement()

	require.NotEqual(t, first, second)
	require.Len(t, first, len(ForgottenPrefix)+32)
	require.True(t, IsReplacement(first))
	for _, id := range []string{"", "ada@example.com", ForgottenPrefix, ForgottenPrefix + "ada", first + "0", first[:len(first)-1] + "g",
		ForgottenPrefix + "0123456789ABCDEF0123456789ABCDEF"} {
		require.False(t, IsReplacement(id), id)
	}
}
