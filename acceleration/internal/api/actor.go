package api

import (
	"context"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The headers a client says who is at the keyboard with. They are what the audit records
// as the actor, and they are read from a server-side caller only: a client header is
// unsigned, so what it buys is a name beside a change somebody already had the credential
// to make, never permission to make it.
//
// X-Stream-Client is the client itself (dashboard, cli, sdk), which Stream's own clients
// already tag themselves with. The other two are the person behind it, which only a client
// that signs its users in can know: the dashboard has the operator who clicked save, the
// CLI has whoever ran it, and an SDK syncing on startup has nobody and says nothing.
const (
	clientHeader    = "X-Stream-Client"
	actorIDHeader   = "X-Stream-Actor-Id"
	actorNameHeader = "X-Stream-Actor-Name"
)

// actorNameLimit is how much of a name is kept. A name is written into every row a client
// makes, and nothing is served by storing a kilobyte of it.
const actorNameLimit = 200

// actorContextKey holds who made the request, as their client named them.
type actorContextKey struct{}

// Actor is who a change is recorded against: the client it was made from, and the person
// that client says was at the keyboard. Every field may be empty, because a caller holding
// a server-side credential is free to say nothing about itself.
type Actor struct {
	// Source is one of store's audit sources, never empty: a caller that named no client
	// is the API itself.
	Source string
	ID     string
	Name   string
}

// actorOf reads the actor a request names. A caller that is not server-side gets none:
// every audited operation is server-side only, so a client header on anything else is
// somebody asking to be written into a log they cannot write to anyway.
func actorOf(r *http.Request, serverSide bool) Actor {
	actor := Actor{Source: store.AuditSourceAPI}
	if !serverSide {
		return actor
	}
	actor.Source = store.AuditSource(strings.ToLower(strings.TrimSpace(r.Header.Get(clientHeader))))
	actor.ID = trimmedTo(r.Header.Get(actorIDHeader), actorNameLimit)
	actor.Name = trimmedTo(r.Header.Get(actorNameHeader), actorNameLimit)
	return actor
}

// ActorFrom returns who the request says it is from. A request nobody named is the API
// with no person behind it, which is what an unauthenticated one reads as too.
func ActorFrom(ctx context.Context) Actor {
	actor, ok := ctx.Value(actorContextKey{}).(Actor)
	if !ok || actor.Source == "" {
		return Actor{Source: store.AuditSourceAPI}
	}
	return actor
}

func trimmedTo(value string, most int) string {
	trimmed := strings.TrimSpace(value)
	if len(trimmed) > most {
		return trimmed[:most]
	}
	return trimmed
}
