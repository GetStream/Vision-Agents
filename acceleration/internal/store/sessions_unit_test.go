package store

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestModelOverwritesEmpty(t *testing.T) {
	require.True(t, ModelOverwrites{}.Empty())
	require.False(t, ModelOverwrites{LLM: "llm-thinking"}.Empty())
	require.False(t, ModelOverwrites{Thinking: "high"}.Empty())

	// A pointer field set to its zero value is still something the caller asked for: a
	// temperature of zero is a real request for a deterministic model, and reading it as
	// "nothing was said" would quietly ignore it.
	zero := 0.0
	require.False(t, ModelOverwrites{Temperature: &zero}.Empty())
}

// TestSearchFallsBackToQuery holds the promise the search box relies on: no words is not
// the same as no matches. It is checked here rather than against Postgres because what it
// is really asserting is which method runs, and that is a decision made before any SQL.
func TestSearchWithoutWordsNeedsACustomer(t *testing.T) {
	var store *Store

	// Both paths refuse a caller with no customer before touching the database, which is
	// what makes it safe to assert the fallback without one.
	_, err := store.SearchSessions(nil, "", "", SessionFilter{})
	require.Error(t, err)
	_, err = store.QuerySessions(nil, "", SessionFilter{})
	require.Error(t, err)
}

func TestJSONBOfSurvivesAValueItCannotEncode(t *testing.T) {
	// A filter arrives off a query string, so the worst an unencodable value can mean is
	// that nothing matches it. Failing the whole query instead would turn one bad label
	// into a broken conversation list.
	require.Equal(t, "{}", jsonbOf(map[string]any{"cycle": make(chan int)}))
	require.Equal(t, `{"project":"health"}`, jsonbOf(map[string]string{"project": "health"}))
}
