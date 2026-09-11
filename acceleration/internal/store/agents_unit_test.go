package store

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNormalizeConfigVideoFrames(t *testing.T) {
	for _, requested := range []int{0, 1, 8} {
		config := AgentConfig{VideoMaxFrames: requested}
		normalizeConfig(&config)
		expected := requested
		if expected == 0 {
			expected = 1
		}
		require.Equal(t, expected, config.VideoMaxFrames)
	}
}
