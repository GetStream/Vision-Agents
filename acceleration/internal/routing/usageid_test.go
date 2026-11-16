package routing

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestUsageIDIsStable(t *testing.T) {
	// Written down once: a new value here means every deterministic id already stored would
	// be billed again under another one.
	// Python's uuid.uuid5 gives the same for the namespace and "8:call_leg5:leg-1".
	require.Equal(t, "83b90697-5cfd-5843-b5b7-62a8d9f8a7e9", UsageID("call_leg", "leg-1").String())
}

func TestUsageIDTellsKeysApart(t *testing.T) {
	require.NotEqual(t, UsageID("number", "n1", "2026-10"), UsageID("number", "n1", "2026-11"))
	require.NotEqual(t, UsageID("number", "n1"), UsageID("call_leg", "n1"))
}

func TestUsageIDTellsKeysApartHoweverTheirPartsAreSplit(t *testing.T) {
	require.NotEqual(t, UsageID("leg", "a:b", "c"), UsageID("leg", "a", "b:c"))
	require.NotEqual(t, UsageID("leg", "ab"), UsageID("leg", "a", "b"))
	require.NotEqual(t, UsageID("leg:a"), UsageID("leg", "a"))
	require.NotEqual(t, UsageID("leg", ""), UsageID("leg"))
}
