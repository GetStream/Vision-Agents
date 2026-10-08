// Package apikey is the api_key scheme: a static key the developer supplies, sent in a
// header the developer names, on every request.
//
// Begin is Done, since nobody consents. Complete checks the header and the key and seals
// both. Retrieve hands the key out as it is, with no expiry, since nothing renews it. Wrap
// sets the header. Classify reads a provider's answer as oauth2_code does, plus one rule: a
// 401 is InvalidGrant, since only a new key helps. Revoke says the key cannot be revoked
// from here.
package apikey

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"golang.org/x/net/http/httpguts"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// Name is the registry name manifests list under schemes. It is the prototype's AuthAPIKey
// (internal/connectors/runtime.go:17 on codex/connector-support at cf62af0d), and the name
// core's fixtures already use (core/testdata/manifests/custom_crm.yaml, shopify.yaml).
const Name = "api_key"

// The keys of core.CompleteInput.Supplied this scheme reads. Key is the prototype's
// credentials field name, api_key (internal/api/connectors.go:534-535 at cf62af0d); header is
// the field its definition kept as api_key_header (connectors.go:112).
const (
	SuppliedKey    = "api_key"
	SuppliedHeader = "header"
)

// payloadVersion is the shape of payload. A new shape is a new version, so a sealed blob is
// always read as what it was written as.
const payloadVersion = 1

// ErrNotRevocable is Revoke: a static key has no revocation endpoint to call (there is no
// RFC 7009 for an API key), so nothing was sent and the key still works at the provider
// until someone revokes it there, in the provider's own console.
var ErrNotRevocable = errors.New("apikey: a static key cannot be revoked from the router; revoke it at the provider")

// errNoKey is what a request carried by Wrap fails with when the credential is not one this
// scheme issued, rather than leave without one.
var errNoKey = errors.New("apikey: the credential is not an api_key credential")

// Scheme is the api_key scheme. It holds no state and is safe for concurrent use.
type Scheme struct{}

var (
	_ core.Scheme   = (*Scheme)(nil)
	_ core.Exporter = (*Scheme)(nil)
)

// New returns the scheme. It takes no configuration: the header is the connection's, and
// supplied with the key.
func New() *Scheme {
	return &Scheme{}
}

// Name is Name.
func (*Scheme) Name() string {
	return Name
}

// payload is the sealed payload of the StoredCredentials, and also the AccessCredential's
// secret: the key is the whole credential, so the two are one shape.
type payload struct {
	Header string `json:"header"`
	Key    string `json:"key"`
}

// Begin is Done: the key is supplied, nobody consents.
func (*Scheme) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{Done: true}, nil
}

// Complete checks the supplied header and key and seals them. Its errors name what is
// wrong, never the value.
func (*Scheme) Complete(_ context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	for name := range in.Supplied {
		if name != SuppliedKey && name != SuppliedHeader {
			return core.StoredCredentials{}, core.AccountInfo{}, fmt.Errorf("apikey: %q is not a value api_key takes; it takes %s and %s", name, SuppliedKey, SuppliedHeader)
		}
	}
	p := payload{Header: http.CanonicalHeaderKey(in.Supplied[SuppliedHeader]), Key: in.Supplied[SuppliedKey]}
	if err := p.check(); err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	raw, err := json.Marshal(p)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: raw}, core.AccountInfo{}, nil
}

// Retrieve hands the key out with no expiry, and stored as it is, so the resolver has
// nothing to persist.
func (*Scheme) Retrieve(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest, _ core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	if _, err := open(stored); err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	return core.NewAccessCredential(Name, time.Time{}, stored.Payload), stored, nil
}

// Wrap sets the header on a clone of every request. A credential this scheme did not issue
// fails each request instead.
func (*Scheme) Wrap(base http.RoundTripper, c core.AccessCredential) http.RoundTripper {
	var p payload
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &p) != nil || p.check() != nil {
		return refuse{}
	}
	return header{base: base, name: p.Header, value: p.Key}
}

// Export is the key in its header, as Wrap sends it. No OAuth client issued it: the customer
// supplied it. A credential this scheme did not issue is refused.
func (*Scheme) Export(c core.AccessCredential) (core.ExportedCredential, error) {
	var p payload
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &p) != nil || p.check() != nil {
		return core.ExportedCredential{}, errNoKey
	}
	return core.ExportedCredential{Header: p.Header, Value: p.Key, ExpiresAt: c.ExpiresAt}, nil
}

// Classify is oauth2code.ClassifyStatic: oauth2code's reading of a provider's answer, and a
// 401 is InvalidGrant, since a static key is never renewed and only a new key helps.
func (*Scheme) Classify(resp *http.Response, body []byte, err error) core.Outcome {
	return oauth2code.ClassifyStatic(resp, body, err)
}

// Revoke sends nothing and returns ErrNotRevocable: the key lives on at the provider.
func (*Scheme) Revoke(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest) error {
	if _, err := open(stored); err != nil {
		return err
	}
	return ErrNotRevocable
}

// open reads the payload this scheme sealed. Its errors never quote the payload.
func open(stored core.StoredCredentials) (payload, error) {
	if stored.Scheme != Name || stored.Version != payloadVersion {
		return payload{}, fmt.Errorf("apikey: stored credentials are %q version %d, not %q version %d", stored.Scheme, stored.Version, Name, payloadVersion)
	}
	var p payload
	if json.Unmarshal(stored.Payload, &p) != nil {
		return payload{}, errors.New("apikey: stored credentials payload is unreadable")
	}
	if err := p.check(); err != nil {
		return payload{}, err
	}
	return p, nil
}

// check refuses a header or a key that cannot go on the wire as it is. Its errors name what
// is wrong, never the value.
func (p payload) check() error {
	switch {
	case p.Header == "":
		return errors.New("apikey: the header is required")
	// RFC 9110 section 5.1: field-name = token.
	case !httpguts.ValidHeaderFieldName(p.Header):
		return errors.New("apikey: the header is not a valid HTTP field name")
	case forbidden(p.Header):
		return errors.New("apikey: the header is one the router sets itself or must not send to a provider")
	case p.Key == "":
		return errors.New("apikey: the key is required")
	// RFC 9110 section 5.5: CR, LF and NUL in a field value «are invalid and dangerous», and
	// «a field value does not include leading or trailing whitespace», so a key with either
	// would not arrive as it was supplied.
	case !httpguts.ValidHeaderFieldValue(p.Key) || strings.TrimSpace(p.Key) != p.Key:
		return errors.New("apikey: the key is not a valid HTTP field value: it has a control character or leading or trailing whitespace")
	}
	return nil
}

// forbidden is the prototype's list of headers an api_key may not be sent in
// (forbiddenAPIKeyHeader, internal/api/connectors.go:1050-1057 on codex/connector-support at
// cf62af0d), for a name already canonical:
//
//   - Authorization belongs to the schemes that set it (bearer, oauth2_code), so a key here
//     would be a bearer token under another scheme's name.
//   - Cookie would mix a key into the provider's session state.
//   - Host (RFC 9110 section 7.2) picks the virtual host the request reaches, and egress
//     checks the URL, not this header.
//   - Content-Length and Transfer-Encoding (RFC 9112 section 6) frame the body, and Connection
//     (RFC 9110 section 7.6.1) names hop-by-hop fields; setting them would let a key change
//     how the request is read.
//   - Proxy-Authorization, Proxy-Authenticate and every other Proxy- field (RFC 9110 sections
//     11.7.1, 11.7.2) are for a proxy, not the provider.
//
// and, past the prototype's list, the fields that control the connection or the framing,
// which net/http (go1.27) drops, refuses, or fails the request on with the key in the error:
//
//   - Keep-Alive, TE and Upgrade are connection-specific (RFC 9110 section 7.6.1), and an
//     HTTP/2 message «MUST NOT» carry them (RFC 9113 section 8.2.2; TE only as "trailers").
//     Egress negotiates HTTP/2 (internal/egress/public.go:124 clones DefaultTransport,
//     ForceAttemptHTTP2), where Upgrade fails the request with the key in the error, TE
//     gets a 400 and Keep-Alive is dropped.
//   - Trailer lists the trailer fields to come (RFC 9110 section 6.6.2); net/http drops it
//     from the header section on HTTP/1.1 and HTTP/2.
//   - Expect asks the server for a behaviour (RFC 9110 section 10.1.1); any value but
//     100-continue may be answered 417, as it is on HTTP/1.1.
func forbidden(name string) bool {
	switch name {
	case "Authorization", "Cookie", "Host", "Content-Length", "Connection", "Proxy-Authorization", "Proxy-Authenticate", "Transfer-Encoding",
		"Keep-Alive", "Te", "Trailer", "Upgrade", "Expect":
		return true
	}
	return strings.HasPrefix(name, "Proxy-")
}

// header is the RoundTripper Wrap returns.
type header struct {
	base  http.RoundTripper
	name  string
	value string
}

// RoundTrip sets the header on a clone: net/http's RoundTripper contract is that
// «RoundTrip should not modify the request».
func (h header) RoundTrip(r *http.Request) (*http.Response, error) {
	out := r.Clone(r.Context())
	out.Header.Set(h.name, h.value)
	return h.base.RoundTrip(out)
}

type refuse struct{}

func (refuse) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		// net/http's RoundTripper contract: it «must always close the body, including on
		// errors».
		_ = r.Body.Close()
	}
	return nil, errNoKey
}
