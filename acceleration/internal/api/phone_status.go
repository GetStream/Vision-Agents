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
// say.
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
	if s.sessions != nil {
		s.sessions.EndUnanswered(asked.CustomerID, asked.SessionID)
	}
	w.WriteHeader(http.StatusNoContent)
}

// refuseUnanswered is callUnanswered for the session sessionID when the call placed for it
// ended with nobody answering, or nil. Such a session is not carried on in writing: nobody
// was ever on its conversation, and an SDK attaching to it would hold it open for nothing.
func (s *Server) refuseUnanswered(ctx context.Context, customerID, sessionID string) error {
	if s.phone == nil {
		return nil
	}
	call, err := s.phone.Outcomes().Read(ctx, customerID, sessionID)
	if errors.Is(err, outcome.ErrUnknownCall) {
		return nil
	}
	if err != nil {
		s.logger.Warn("could not read whether a session's placed call went unanswered",
			"session", sessionID, "error", err)
		return nil
	}
	if call.NobodyAnswered() {
		return callUnanswered(call.Outcome)
	}
	return nil
}
