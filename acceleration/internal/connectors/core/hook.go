package core

import (
	"context"
	"encoding/json"
	"net/url"
)

// Hook is the escape hatch for what a manifest cannot say. It is registered by name,
// referenced from a manifest, and called at exactly three points, so the number of places
// provider code can run stays fixed and every hook can be counted.
type Hook func(ctx context.Context, hc *HookContext) error

// The three points a hook can run at, as named in a manifest's hooks.
const (
	HookBeforeAuthorize = "before_authorize"
	HookBeforeComplete  = "before_complete"
	HookAfterToken      = "after_token"
)

// HookContext is what a hook sees at its point. Only the field for that point is set.
type HookContext struct {
	Point    string
	Manifest ResolvedManifest
	// AuthorizeURL is, before authorize, the URL about to be sent; a hook may add to it.
	AuthorizeURL *url.URL
	// Query is, before complete, the full callback query, such as for a signed callback.
	Query url.Values
	// TokenResponse is, after token, the raw token response body.
	TokenResponse json.RawMessage
}
