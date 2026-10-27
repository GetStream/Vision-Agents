package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
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
// holding it. A session not open yet is left to its start timeout.
func (s *Server) endUnanswered(ctx context.Context, customerID, sessionID string) {
	if s.sessions == nil || s.sessions.EndUnanswered(customerID, sessionID) {
		return
	}
	if s.directory == nil || s.forwarder == nil {
		return
	}
	address, err := s.directory.Node(ctx, sessionID)
	if err != nil || address == "" || address == s.directory.Address() {
		return
	}
	ask, cancel := context.WithTimeout(ctx, forwardTimeout)
	defer cancel()
	answer, err := s.askToEndUnanswered(ask, address, unansweredPeerRequest{CustomerID: customerID, SessionID: sessionID})
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
