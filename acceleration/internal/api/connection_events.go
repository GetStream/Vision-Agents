package api

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
)

// receiveConnectionEvent is the unauthenticated callback a connection's MCP server delivers
// the events of a binding's subscription to (internal/mcpevents). The token in the path names
// the subscription, and the signature is checked against its own secret, so it is the server
// that subscription was made with or nobody. It is apart from receiveProviderAppEvent, which
// verifies with a provider app's secret.
func (s *Server) receiveConnectionEvent(w http.ResponseWriter, r *http.Request) {
	if s.mcpEvents == nil {
		writeError(w, gone("this deployment subscribes to no MCP events"))
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, mcpevents.MaxEventBytes+1))
	if err != nil {
		writeError(w, invalidRequest(err.Error()))
		return
	}
	if len(body) > mcpevents.MaxEventBytes {
		writeError(w, payloadTooLarge("a delivery is at most 256 KiB"))
		return
	}
	reply := s.mcpEvents.Receive(r.Context(), r.PathValue("token"), r.Header, body)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(reply.Status)
	_ = json.NewEncoder(w).Encode(reply.Body)
}
