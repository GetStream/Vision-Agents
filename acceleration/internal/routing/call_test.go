package routing

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// asked is work that used one request.
type asked struct{}

func (asked) Used() pricing.Quantities { return pricing.Quantities{pricing.Requests: 1} }

func TestACallWithNoRecorderIsStillAnswered(t *testing.T) {
	result, err := Call(context.Background(), Meter{}, func(context.Context) (asked, error) { return asked{}, nil })

	require.NoError(t, err)
	require.Equal(t, asked{}, result)
}

func TestAStatTakesWhatTheWorkUsedIntoItsColumnsAndMeters(t *testing.T) {
	var stat Stat
	stat.Use(pricing.Quantities{
		pricing.InputTokens:           3,
		pricing.CachedInputTokens:     11,
		pricing.CacheWriteInputTokens: 2_386,
		pricing.OutputTokens:          22,
		pricing.Requests:              1,
		pricing.OutputAudioTokens:     40,
		pricing.Seconds:               0,
	})

	require.Equal(t, Usage{InputTokens: 2_400, CachedInputTokens: 11, OutputTokens: 22, Requests: 1}, stat.Usage)
	require.Equal(t, pricing.Quantities{pricing.CacheWriteInputTokens: 2_386, pricing.OutputAudioTokens: 40}, stat.Meters)
}
