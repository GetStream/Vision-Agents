package store

import (
	"reflect"
	"slices"
	"strings"
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

// TestUpdatingAConfigWritesEveryFieldOfIt holds the update's column list against the
// model, since a field the two disagree about is a setting that saves once and then
// silently stops saving - which is worse than one that never worked.
func TestUpdatingAConfigWritesEveryFieldOfIt(t *testing.T) {
	// What an update cannot change: who owns the row, which row it is, when it appeared,
	// and whether it is gone. Deleting is its own statement.
	immutable := []string{"id", "customer_id", "created_at", "deleted_at"}

	model := reflect.TypeFor[AgentConfig]()
	for index := range model.NumField() {
		tag := model.Field(index).Tag.Get("bun")
		column, _, _ := strings.Cut(tag, ",")
		if column == "" || strings.HasPrefix(tag, "table:") || slices.Contains(immutable, column) {
			continue
		}
		require.Containsf(t, configColumns, column,
			"AgentConfig.%s is stored on create and dropped on every update", model.Field(index).Name)
	}

	for _, column := range configColumns {
		require.NotContainsf(t, immutable, column, "%s is not an update's to change", column)
	}
}
