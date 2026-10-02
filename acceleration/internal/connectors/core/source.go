package core

import (
	"context"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// Source discovers and runs tools of one kind: mcp, http, openapi, caller.
type Source interface {
	Kind() string
	// Discover lists what the connection offers, each tool with a schema digest, so a
	// grant can pin the exact schema it was reviewed against.
	Discover(ctx context.Context, b Bound) ([]ToolSpec, error)
	// Open returns a runtime that exposes only the granted tools whose digest still
	// matches.
	Open(ctx context.Context, b Bound, grants []ToolGrant) (Runtime, error)
}

// Runtime is one opened source for one session.
type Runtime interface {
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

// Bound is one binding resolved against one connection.
type Bound struct {
	Binding    Binding
	Connection Connection
	Profile    Profile
	// Transport already carries the resolver and the egress policy, so a Source cannot
	// get the wrapping order wrong or skip the egress check.
	Transport func(context.Context) (http.RoundTripper, error)
}

// Binding attaches a connector's tools to an agent config under an alias.
type Binding struct {
	// Name is the alias the tools are exposed under.
	Name        string
	ConnectorID string
	// Selection is fixed, an app-owned ConnectionID chosen in the config, or session, the
	// verified caller's own connection chosen when the session starts.
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
