package sandbox

import (
	"context"
	"errors"
	"net/url"
	"strconv"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// Grant is a connection whose credential code in the sandbox may send, and only to Hosts.
// The sandbox holds a placeholder in Env; the broker's egress puts the credential in its
// place.
type Grant struct {
	Connection core.ConnectionRef
	// Binding is the agent config's binding the grant comes from.
	Binding string
	// Scheme is the connection's auth scheme, which reads its token from its credential.
	Scheme string
	// Env is the env var code in the sandbox reads the placeholder from, such as GITHUB_TOKEN.
	Env string
	// Hosts are the only hosts a request may carry the credential to: exact hosts, or *. and
	// a domain.
	Hosts []string
	// Required refuses the sandbox when the credential cannot be had, as a required binding
	// refuses the session.
	Required bool
}

// Token is a grant's credential as code sends it in `Authorization: Bearer`, and the revision
// of the stored credentials it came from. JSON leaves Value out, and String and GoString
// redact it, so no log line prints it.
type Token struct {
	Value    string `json:"-"`
	Revision int
}

func (t Token) String() string {
	return "Token{Value:redacted Revision:" + strconv.Itoa(t.Revision) + "}"
}

func (t Token) GoString() string { return t.String() }

// Tokens is a grant's credential as it is now. An error wrapping ErrNoCredential means the
// connection cannot be used until someone reconnects it; any other error may pass, such as a
// provider that could not renew the credential just now.
type Tokens func(ctx context.Context, grant Grant) (Token, error)

// ErrNoCredential is a connection that gives no credential until it is reconnected: it is
// disconnected, waiting for a reconnect, deleted, or its scheme sends no bearer token.
var ErrNoCredential = errors.New("sandbox: the connection gives no credential until it is reconnected")

// Mount is what a sandbox is created with so that its code reaches its grants.
type Mount struct {
	// Secrets maps an env var to the name the sandbox provider keeps the credential under,
	// as Daytona's SandboxBaseParams.Secrets takes it.
	Secrets map[string]string
	// Env are env vars the sandbox is created with as they are, such as a placeholder the
	// broker's egress knows.
	Env map[string]string
	// RunEnv are env vars every command and run of code in the sandbox gets, such as its proxy
	// and the CA files it trusts. Set per run, they win over what the provider sets in the
	// sandbox, which Env does not: Daytona's netleash replaces its proxy and CA env vars (live
	// probe, 2026-10-21).
	RunEnv map[string]string
	// Proxy, when set, is the only address the sandbox may reach: an http or https proxy at an
	// IPv4 address, with the sandbox's credential in the userinfo. RunEnv sends the sandbox's
	// requests there, or to a tunnel in the sandbox to it.
	Proxy *url.URL
	// Files are written into the sandbox, by absolute path, before code runs in it.
	Files map[string][]byte
	// Setup are shell commands run in the sandbox, in order, once Files are written and before
	// any code. One that fails stops the run, and runs again, with those after it, before the
	// next.
	Setup []string
	// EachRun are shell commands run in the sandbox before each run of code, once Setup is
	// done. One that fails fails the run, and no code runs.
	EachRun []string
}

// CredentialBroker lets code in a sandbox call a connector's API with a connection's
// credential it never holds. The session knows only this interface, so the broker is a
// router setting rather than a rewrite.
type CredentialBroker interface {
	// Prepare puts each grant's credential where the sandbox's egress reads it and returns
	// what the sandbox is created with. A grant whose credential cannot be had is left out
	// with a warning, unless it is required, which fails Prepare. A broker whose egress asks
	// for the credential on every request leaves out only a grant whose connection gives
	// ErrNoCredential.
	Prepare(ctx context.Context, grants []Grant) (Mount, error)
	// BeforeRun runs before each piece of code: it hands the egress a credential whose
	// revision moved since it last did, and withdraws one whose connection gives
	// ErrNoCredential, so code never runs with a credential the router would no longer send
	// itself. A credential that cannot be had just now is kept, and asked for again next run.
	BeforeRun(ctx context.Context) error
	// Close removes everything Prepare left with the provider. Safe to call twice, and
	// Prepare may run again after it.
	Close(ctx context.Context) error
}
