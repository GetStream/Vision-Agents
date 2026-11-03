package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// machinePeerPath is where a router tells the router holding a placed call's session what
// picked up the call. Served by hand and named in no spec, as unansweredPeerPath.
const machinePeerPath = "/internal/phone/machine"

// machinePeerRequest names the session a peer is told about, and what picked up its call.
type machinePeerRequest struct {
	CustomerID string        `json:"customer_id"`
	SessionID  string        `json:"session_id"`
	Machine    phone.Machine `json:"machine"`
}

// tellMachine hands what picked up a placed call to its session, here or on the node holding
// it, off the vendor's request, as endUnanswered does.
func (s *Server) tellMachine(ctx context.Context, customerID, sessionID string, machine phone.Machine) {
	if s.sessions == nil {
		return
	}
	ctx = context.WithoutCancel(ctx)
	go func() {
		ctx, cancel := context.WithTimeout(ctx, forwardTimeout)
		defer cancel()
		s.tellMachineNow(ctx, machinePeerRequest{CustomerID: customerID, SessionID: sessionID, Machine: machine})
	}()
}

// tellMachineNow is tellMachine on the caller's goroutine.
func (s *Server) tellMachineNow(ctx context.Context, told machinePeerRequest) {
	if s.sessions.Machine(told.CustomerID, told.SessionID, told.Machine) || s.directory == nil || s.forwarder == nil {
		return
	}
	address, err := s.directory.Node(ctx, told.SessionID)
	if err != nil {
		s.logger.Warn("could not find the node holding the session of a call a machine report is for",
			"session", told.SessionID, "error", err)
		return
	}
	if address == "" || address == s.directory.Address() {
		return
	}
	answer, err := s.askMachine(ctx, address, told)
	if err != nil {
		s.logger.Warn("could not tell the node holding a session what picked up its call",
			"session", told.SessionID, "node", address, "error", err)
		return
	}
	if answer.GetStatus() != http.StatusNoContent {
		s.logger.Warn("the node holding a session did not take what picked up its call",
			"session", told.SessionID, "node", address, "status", answer.GetStatus())
	}
}

// askMachine posts told to machinePeerPath on the node at address.
func (s *Server) askMachine(ctx context.Context, address string, told machinePeerRequest) (*node.ForwardResponse, error) {
	body, err := json.Marshal(told)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, machinePeerPath, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	return s.forwarder.Ask(ctx, address, request, body)
}

// tellMachineForPeer is tellMachine's other end. Anything not handed over by a peer is
// answered as no such route: the request names a customer outright, which is only a peer's to
// say. The peer read the report off a call token, so it is taken as told.
func (s *Server) tellMachineForPeer(w http.ResponseWriter, r *http.Request) {
	if !node.Forwarded(r.Context()) {
		writeError(w, notFound("no such route"))
		return
	}
	var told machinePeerRequest
	if err := json.NewDecoder(r.Body).Decode(&told); err != nil {
		writeError(w, invalidRequest("the request could not be read"))
		return
	}
	if s.sessions != nil {
		s.sessions.Machine(told.CustomerID, told.SessionID, told.Machine)
	}
	w.WriteHeader(http.StatusNoContent)
}
