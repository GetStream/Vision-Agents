package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
)

// unansweredPeerPath is where a router asks the router holding a placed call's session to end
// it, nobody having answered. It is served by hand and named in no spec, as scheduledPeerPath:
// only a peer calls it, and only through the peer forwarder.
const unansweredPeerPath = "/internal/phone/end-unanswered"

// unansweredPeerRequest names the session a peer is asked to end.
type unansweredPeerRequest struct {
	CustomerID string `json:"customer_id"`
	SessionID  string `json:"session_id"`
}

// endUnanswered ends the session of a placed call nobody will answer, here or on the node
// holding it, off the vendor's request: a vendor waits a few seconds for its answer, less than
// a forward to a peer may take, and a vendor that hangs up must not cancel the ending.
// A session not open yet is refused when it opens (session.CallUnansweredError).
func (s *Server) endUnanswered(ctx context.Context, customerID, sessionID string) {
	if s.sessions == nil {
		return
	}
	ctx = context.WithoutCancel(ctx)
	go func() {
		ctx, cancel := context.WithTimeout(ctx, forwardTimeout)
		defer cancel()
		s.endUnansweredNow(ctx, customerID, sessionID)
	}()
}

// endUnansweredNow is endUnanswered on the caller's goroutine.
func (s *Server) endUnansweredNow(ctx context.Context, customerID, sessionID string) {
	if s.sessions.EndUnanswered(customerID, sessionID) || s.directory == nil || s.forwarder == nil {
		return
	}
	address, err := s.directory.Node(ctx, sessionID)
	if err != nil {
		s.logger.Warn("could not find the node holding an unanswered call's session",
			"session", sessionID, "error", err)
		return
	}
	if address == "" || address == s.directory.Address() {
		return
	}
	answer, err := s.askToEndUnanswered(ctx, address, unansweredPeerRequest{CustomerID: customerID, SessionID: sessionID})
	if err != nil {
		s.logger.Warn("could not ask the node holding an unanswered call's session to end it",
			"session", sessionID, "node", address, "error", err)
		return
	}
	if answer.GetStatus() != http.StatusNoContent {
		s.logger.Warn("the node holding an unanswered call's session did not end it",
			"session", sessionID, "node", address, "status", answer.GetStatus())
	}
}

// askToEndUnanswered posts asked to unansweredPeerPath on the node at address.
func (s *Server) askToEndUnanswered(ctx context.Context, address string, asked unansweredPeerRequest) (*node.ForwardResponse, error) {
	body, err := json.Marshal(asked)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, unansweredPeerPath, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	return s.forwarder.Ask(ctx, address, request, body)
}

// endUnansweredForPeer is endUnanswered's other end. Anything not handed over by a peer is
// answered as no such route: the request names a customer outright, which is only a peer's to
// say. Even a peer's word is checked against the call as stored: one that is not kept, or that
// somebody answered or is still ringing, is answered as no such call and nothing is ended.
func (s *Server) endUnansweredForPeer(w http.ResponseWriter, r *http.Request) {
	if !node.Forwarded(r.Context()) {
		writeError(w, notFound("no such route"))
		return
	}
	var asked unansweredPeerRequest
	if err := json.NewDecoder(r.Body).Decode(&asked); err != nil {
		writeError(w, invalidRequest("the request could not be read"))
		return
	}
	if s.sessions == nil {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	// A node without a store keeps no calls to check an ask against.
	if s.phone == nil || s.phone.Outcomes() == nil {
		writeError(w, errUnknownCall)
		return
	}
	// The asker sends this only once the call's row says nobody answered, so a valid ask finds
	// that row.
	call, err := s.phone.Outcomes().Read(r.Context(), asked.CustomerID, asked.SessionID)
	switch {
	case errors.Is(err, outcome.ErrUnknownCall), err == nil && !call.NobodyAnswered():
		s.logger.Warn("refused a peer's ask to end a session its stored call does not say went unanswered",
			"session", asked.SessionID)
		writeError(w, errUnknownCall)
		return
	case err != nil:
		s.logger.Error("could not read a peer's unanswered call", "session", asked.SessionID, "error", err)
		writeError(w, internalError())
		return
	}
	s.sessions.EndUnanswered(call.CustomerID, call.CallID)
	w.WriteHeader(http.StatusNoContent)
}
