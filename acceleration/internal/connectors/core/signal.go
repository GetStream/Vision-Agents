package core

import "net/http"

// Verifier checks one inbound provider event and names the grants it is about. An event
// that fails verification changes nothing, so the inbound endpoint needs no API auth.
type Verifier interface {
	Name() string
	Verify(r *http.Request, body []byte, m ResolvedManifest) ([]Signal, error)
}

// SignalKind is what an inbound event says happened to a grant.
type SignalKind string

const (
	SignalRevoked     SignalKind = "revoked"
	SignalUninstalled SignalKind = "uninstalled"
	SignalRotated     SignalKind = "rotated"
)

// Signal is a verified event about the grants of one account. It names the account, not a
// connection, because the provider knows only its own ids.
type Signal struct {
	ConnectorID string
	AccountID   string
	Kind        SignalKind
}
