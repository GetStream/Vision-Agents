package main

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

// TELNYX_CONNECTION_ID is a TeXML application or the bench's FQDN connection, never a Call
// Control Application, so placing calls on it would fail at Telnyx instead of here.
func TestTelnyxCallerDoesNotPlaceCallsOnTheConnection(t *testing.T) {
	t.Setenv("TELNYX_API_KEY", "key")
	t.Setenv("TELNYX_CONNECTION_ID", "1234567890")
	t.Setenv("TELNYX_CALL_CONTROL_ID", "")

	_, err := NewTelnyxCaller(http.DefaultClient)

	require.ErrorContains(t, err, "TELNYX_CALL_CONTROL_ID")
}
