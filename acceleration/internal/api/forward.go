package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/node"
)

// forwardTimeout bounds one request carried to the node that can answer it. Generous,
// because what is carried is an ordinary API call and the node answering it applies
// whatever timeout that call already had; short enough that a node which has stopped
// answering does not hold a caller indefinitely.
const forwardTimeout = 30 * time.Second

// unreachableNode is what a caller is told when the session exists but the node running
// it does not answer, which is a different thing from a session that is not there.
const unreachableNode = "the node running this session could not be reached"

// onOwningNode carries a request about a session to the node running it.
//
// A session lives in one process's memory and a load balancer has no reason to prefer
// that process, so without this a request lands on a node that cannot answer it and is
// told the session does not exist.
//
// Outermost of this server's own middlewares, so a request that belongs elsewhere is not
// authenticated, logged or counted against a quota twice: the node that answers does all
// three, and this one only hands it over.
func (s *Server) onOwningNode(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		address, elsewhere := s.owningNode(r)
		if !elsewhere {
			next.ServeHTTP(w, r)
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), forwardTimeout)
		defer cancel()
		if err := s.forwarder.Forward(ctx, address, w, r); err != nil {
			s.logger.Error("could not reach the node running a session",
				"node", address, "path", r.URL.Path, "error", err)
			writeError(w, unavailable(unreachableNode))
		}
	})
}

// owningNode is where to send a request this node cannot answer, and false when this node
// is the one to answer it.
func (s *Server) owningNode(r *http.Request) (string, bool) {
	if s.directory == nil || s.forwarder == nil {
		return "", false
	}
	// A request a peer handed over is answered here. Two nodes each believing the other
	// holds the session would otherwise pass it back and forth.
	if r.Header.Get(node.ForwardedHeader) != "" {
		return "", false
	}
	// A socket cannot be carried over a call that answers once, and does not need to be:
	// the relay reaches a session's events wherever the socket lands.
	if r.Header.Get("Upgrade") != "" {
		return "", false
	}
	id, ok := sessionInPath(r.URL.Path)
	if !ok {
		return "", false
	}
	if s.sessions != nil && s.sessions.Running(id) {
		return "", false
	}

	address, err := s.directory.Node(r.Context(), id)
	if err != nil {
		s.logger.Warn("could not find out which node is running a session",
			"session", id, "error", err)
		return "", false
	}
	// Nothing says it is running: the session ended, or never existed, and what reads the
	// row is as able to answer here as anywhere.
	if address == "" || address == s.directory.Address() {
		return "", false
	}
	return address, true
}

// sessionInPath returns the session a path is about, if it is about one.
//
// The collection paths -- query, search -- come back as ids nothing is running, which
// costs them a lookup that finds nobody and is answered here as before.
func sessionInPath(path string) (string, bool) {
	rest, ok := strings.CutPrefix(path, "/v1/agents/sessions/")
	if !ok {
		return "", false
	}
	id, _, _ := strings.Cut(rest, "/")

	return id, id != ""
}

// forwardedMessage hands an arriving message to the node running the conversation on its
// channel, reporting whether this node is not the one to answer it.
//
// Without it an agent answering on another node is invisible from here and the message
// goes to a worker instead, which is a second agent writing into a conversation the first
// is already answering in. So the directory decides, not the handover: a node that holds
// the conversation and cannot be reached is still the only one that may answer.
func (s *Server) forwardedMessage(r *http.Request, body []byte, event messageEvent) bool {
	if s.directory == nil || s.forwarder == nil || r.Header.Get(node.ForwardedHeader) != "" {
		return false
	}

	address, err := s.directory.NodeByAgent(r.Context(), event.ChannelID)
	if err != nil {
		s.logger.Warn("could not find out which node is answering in a channel",
			"channel", event.ChannelID, "error", err)
		return false
	}
	if address == "" || address == s.directory.Address() {
		return false
	}

	ctx, cancel := context.WithTimeout(r.Context(), forwardTimeout)
	defer cancel()
	// Handed over whole, signature and all, because the node answering it verifies the
	// delivery for itself rather than taking this node's word that it came from Stream.
	answer, err := s.forwarder.Ask(ctx, address, r, body)
	switch {
	case err != nil:
		s.logger.Error("could not reach the node answering in a channel",
			"node", address, "channel", event.ChannelID, "error", err)
	case answer.GetStatus() != http.StatusOK:
		s.logger.Warn("the node answering in a channel would not take a message",
			"node", address, "channel", event.ChannelID, "status", answer.GetStatus())
	}

	return true
}
