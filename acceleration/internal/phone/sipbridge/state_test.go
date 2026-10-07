package sipbridge

import (
	"testing"

	"github.com/emiago/sipgo/sip"
	"github.com/stretchr/testify/require"
)

func TestLegStateHoldsWhatContinuingTheDialogNeeds(t *testing.T) {
	var target sip.Uri
	require.NoError(t, sip.ParseUri("sip:+15550000001@trunk.example.com", &target))
	invite := sip.NewRequest(sip.INVITE, target)
	invite.AppendHeader(sip.NewHeader("Call-ID", "call-1"))
	invite.AppendHeader(sip.NewHeader("From", "<sip:+15550000002@bridge.example.com>;tag=local-tag"))
	invite.AppendHeader(sip.NewHeader("To", "<sip:+15550000001@trunk.example.com>"))
	invite.AppendHeader(sip.NewHeader("CSeq", "1 INVITE"))

	answer := sip.NewResponseFromRequest(invite, 200, "OK", nil)
	// NewResponseFromRequest copies a parsed To; ReplaceHeader would leave that cached copy,
	// so the tag goes onto it directly.
	answer.To().Params.Add("tag", "remote-tag")
	answer.AppendHeader(sip.NewHeader("Contact", "<sip:peer@192.0.2.10:5060;transport=tcp>"))
	answer.AppendHeader(sip.NewHeader("Record-Route", "<sip:proxy-b.example.com;lr>"))
	answer.AppendHeader(sip.NewHeader("Record-Route", "<sip:proxy-a.example.com;lr>"))

	got := legState(invite, answer, 7, []byte("v=0\r\n"))

	require.Equal(t, LegState{
		CallID:       "call-1",
		LocalTag:     "local-tag",
		RemoteTag:    "remote-tag",
		LocalCSeq:    7,
		RemoteTarget: "sip:peer@192.0.2.10:5060;transport=tcp",
		// A UAC's route set is the 2xx's Record-Route in reverse order (RFC 3261 12.1.2).
		RouteSet: []string{"<sip:proxy-a.example.com;lr>", "<sip:proxy-b.example.com;lr>"},
		LastSDP:  []byte("v=0\r\n"),
	}, got)
}
