package core

import (
	"context"
	"errors"
	"net/http"
	"time"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// ToolSource discovers and runs tools of one kind: mcp, http, openapi, caller.
type ToolSource interface {
	Kind() string
	// Discover lists what the connection offers, each tool with a schema digest, so a
	// ToolGrant can pin the exact schema it was reviewed against.
	Discover(ctx context.Context, b ResolvedBinding) ([]ToolSpec, error)
	// Open returns a Toolset that exposes only the granted tools whose digest still
	// matches.
	Open(ctx context.Context, b ResolvedBinding, grants []ToolGrant) (Toolset, error)
}

// EventSource is a ToolSource whose server can also deliver events to a webhook the router
// serves: MCP Events (experimental-ext-triggers-events «Webhook-Based Delivery», at 6682596d).
// It reaches the server through ResolvedBinding.HTTP, as Discover and Open do.
type EventSource interface {
	// Subscribe creates or refreshes the subscription: the server keys it on the
	// connection's principal, the URL, the event and its arguments, so asking again with
	// the same four refreshes it. ErrNoEvents is a server that offers no events.
	Subscribe(ctx context.Context, b ResolvedBinding, sub EventSubscription) (EventGrant, error)
	// Unsubscribe stops it, named as it was made; Secret is not sent.
	Unsubscribe(ctx context.Context, b ResolvedBinding, sub EventSubscription) error
}

// EventSubscription is one event, with its filters, delivered to one URL signed with one
// secret.
type EventSubscription struct {
	Name      string
	Arguments map[string]any
	URL       string
	// Secret is the subscription's own Standard Webhooks secret, whsec_ and base64, which the
	// client supplies and the server signs every delivery with.
	Secret string
}

// EventGrant is what the server granted.
type EventGrant struct {
	// ID is the server's id for the subscription, for routing only.
	ID string
	// RefreshBefore is when the server stops delivering unless subscribed to again. Nil is a
	// grant that does not expire.
	RefreshBefore *time.Time
}

// ErrNoEvents is a connection's server that offers no events.
var ErrNoEvents = errors.New("core: the server offers no events")

// Toolset is the tools of one opened ToolSource, for one session.
type Toolset interface {
	// Tools is in llm's shape rather than harness's, so core never pulls in harness, and
	// through it llmrouter and store, and the store can still validate against core.
	Tools() []llm.Tool
	Call(ctx context.Context, call llm.ToolCall) (Result, error)
	Close()
}

// Result is what a tool call gives back to the model.
type Result struct {
	Parts []llm.ContentPart
}

// MaxResultBytes is the most of a tool's result a Toolset hands back to the model: a longer
// one is cut to fit and ends with TruncatedMarker, within the same bound. 32 KiB is the
// architecture doc's SourceContract («a result over 32 KiB is cut with the marker»,
// architecture.md on connectors/planning) and the prototype's maxMCPToolResultBytes
// (internal/mcp/mcp.go:26 on codex/connector-support at cf62af0d). Unverified, not measured:
// a choice that keeps one result from taking over the model's context.
const MaxResultBytes = 32 << 10

// TruncatedMarker ends a result that was cut at MaxResultBytes, so the model knows it read
// part of it. The prototype's truncatedToolResultNotice (internal/mcp/mcp.go:27 at cf62af0d).
const TruncatedMarker = "\n[connector result truncated]"

// CutResult is text when it fits MaxResultBytes, and otherwise as much of it as fits with
// TruncatedMarker after it, never splitting a UTF-8 sequence. A Toolset cuts its results with
// it, and the session's dispatcher cuts again whatever a Toolset hands back, so the cap holds
// for every source.
func CutResult(text string) string {
	if len(text) <= MaxResultBytes {
		return text
	}
	kept := text[:MaxResultBytes-len(TruncatedMarker)]
	for !utf8.ValidString(kept) {
		kept = kept[:len(kept)-1]
	}
	return kept + TruncatedMarker
}

// ToolError is a tool that ran and reported its own failure, such as an MCP result with
// isError. Its Message is the tool's, for the model to read and act on; any other error from
// Toolset.Call is the call failing to happen or to come back.
type ToolError struct {
	Message string
}

func (e *ToolError) Error() string {
	return e.Message
}

// ResolvedBinding is one binding resolved against one connection.
type ResolvedBinding struct {
	Binding    Binding
	Connection Connection
	Manifest   ResolvedManifest
	// HTTP is the connection's outbound client, from Transports.Client. It already carries
	// the resolver, the scheme and the egress policy, in that order, so a ToolSource cannot
	// get the wrapping order wrong or skip the egress check. It is a client, not a
	// RoundTripper, because the redirect policy that keeps a credential in its origin is the
	// client's (egress.NewClient). Send with it as it is; never replace its Transport.
	HTTP *http.Client
}

// Binding attaches a connector's tools to an agent config under an alias.
type Binding struct {
	// Name is the alias the tools are exposed under.
	Name        string
	ConnectorID string
	// Selection is fixed, an app-owned ConnectionID chosen in the config, or session,
	// the verified caller's own connection chosen when the session starts.
	Selection    string
	ConnectionID string
	Tools        []ToolGrant
	Required     bool
	Timeout      time.Duration
}

// ToolGrant is one exact tool, pinned to the schema it was granted against. There is no
// wildcard, because a wildcard offered once cannot be taken back.
type ToolGrant struct {
	Name         string
	SchemaDigest string
}

// ToolSpec is one discovered tool, in the source's own naming.
type ToolSpec struct {
	Name         string
	Description  string
	InputSchema  map[string]any
	SchemaDigest string
	// NeedsScopes are the scopes a call of the tool needs, from the manifest's ToolRule, so a
	// grant that lacks one is found when the connection is validated, not when a call fails.
	// Empty when nothing says. They are not part of SchemaDigest, which is what the model sees.
	NeedsScopes []string
}

// Connection is one account at one connector, owned by the app or by one user.
type Connection struct {
	ID          string
	ConnectorID string
	// DefinitionRevision is the manifest revision the connection was created from.
	DefinitionRevision int
	// OwnerType is app or user. An org-wide installation is an account, not a third
	// owner.
	OwnerType string
	OwnerID   string
	AccountID string
	Inputs    map[string]string
	Metadata  map[string]string
	Status    string
}
