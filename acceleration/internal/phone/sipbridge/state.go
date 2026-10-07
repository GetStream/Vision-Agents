package sipbridge

import (
	"slices"

	"github.com/emiago/sipgo/sip"
)

// LegState is what another process needs to continue one leg's dialog: send it a
// re-INVITE with a new Contact, then BYE or OPTIONS as before. Nothing reads it yet; it is
// kept in one place so that moving calls between nodes does not have to dig it out of sipgo.
type LegState struct {
	CallID    string
	LocalTag  string
	RemoteTag string
	// LocalCSeq is the last CSeq this side sent in the dialog.
	LocalCSeq uint32
	// RemoteTarget is the Contact of the peer's 2xx, the Request-URI of in-dialog requests.
	RemoteTarget string
	// RouteSet is the Route headers in-dialog requests carry, in sending order.
	RouteSet []string
	// LastSDP is the last SDP this side sent, which a re-INVITE repeats unchanged.
	LastSDP []byte
}

func legState(invite *sip.Request, answer *sip.Response, cseq uint32, sdp []byte) LegState {
	state := LegState{
		CallID:    invite.CallID().Value(),
		LocalTag:  invite.From().Params.GetOr("tag", ""),
		RemoteTag: answer.To().Params.GetOr("tag", ""),
		LocalCSeq: cseq,
		LastSDP:   slices.Clone(sdp),
	}
	if contact := answer.Contact(); contact != nil {
		state.RemoteTarget = contact.Address.String()
	}
	for _, h := range answer.GetHeaders("Record-Route") {
		state.RouteSet = append(state.RouteSet, h.Value())
	}
	slices.Reverse(state.RouteSet)
	return state
}
