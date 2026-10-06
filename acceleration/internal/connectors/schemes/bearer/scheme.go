// Package bearer is the bearer scheme: a static token the developer supplies, such as a
// bot token or a personal access token, sent as an RFC 6750 bearer token on every request.
//
// Begin is Done, since nobody consents. Complete checks the token and seals it. Retrieve
// hands it out as it is, with no expiry, since nothing renews it. Wrap sets
// Authorization: Bearer. Classify reads a provider's answer as oauth2_code does, plus one
// rule: a 401 is InvalidGrant, since only a new token helps. Revoke says the token cannot be
// revoked from here.
package bearer

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// Name is the registry name manifests list under schemes. It is the prototype's AuthBearer
// (internal/connectors/runtime.go:16 on codex/connector-support at cf62af0d).
const Name = "bearer"

// SuppliedToken is the key of core.CompleteInput.Supplied this scheme reads.
const SuppliedToken = "token"

// payloadVersion is the shape of payload. A new shape is a new version, so a sealed blob is
// always read as what it was written as.
const payloadVersion = 1

// prefix is RFC 6750 section 2.1: credentials = "Bearer" 1*SP b64token. One space, as the
// prototype sent it (internal/connectors/runtime.go:194 at cf62af0d).
const prefix = "Bearer "

// b64token is RFC 6750 section 2.1's: 1*( ALPHA / DIGIT / "-" / "." / "_" / "~" / "+" / "/" )
// *"=". A token outside it is not a bearer credential on the wire.
var b64token = regexp.MustCompile(`^[A-Za-z0-9._~+/-]+=*$`)

// ErrNotRevocable is Revoke: a static token has no revocation endpoint this scheme knows of
// (RFC 7009 revokes tokens an OAuth server issued to this client, and nothing here says one
// did), so nothing was sent and the token still works at the provider until someone revokes
// it there.
var ErrNotRevocable = errors.New("bearer: a static token cannot be revoked from the router; revoke it at the provider")

// errNoToken is what a request carried by Wrap fails with when the credential is not one
// this scheme issued, rather than leave without one.
var errNoToken = errors.New("bearer: the credential is not a bearer credential")

// Scheme is the bearer scheme. It holds no state and is safe for concurrent use.
type Scheme struct {
	answers *oauth2code.Scheme
}

var _ core.Scheme = (*Scheme)(nil)

// New returns the scheme. It takes no configuration: the header and its prefix are RFC
// 6750's.
func New() *Scheme {
	// Classify is oauth2code's, which sends nothing. oauth2code.New requires a client all
	// the same, so it gets one that refuses every request; with HTTP set it has nothing
	// else to refuse, so an error here is a change in oauth2code, not in what it was given.
	answers, err := oauth2code.New(oauth2code.Config{HTTP: &http.Client{Transport: refuse{}}})
	if err != nil {
		panic(err)
	}
	return &Scheme{answers: answers}
}

// Name is Name.
func (*Scheme) Name() string {
	return Name
}

// payload is the sealed payload of the StoredCredentials, and also the AccessCredential's
// secret: the token is the whole credential, so the two are one shape.
type payload struct {
	Token string `json:"token"`
}

// Begin is Done: the token is supplied, nobody consents.
func (*Scheme) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{Done: true}, nil
}

// Complete checks the supplied token and seals it. Its errors name what is wrong, never the
// value.
func (*Scheme) Complete(_ context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	for name := range in.Supplied {
		if name != SuppliedToken {
			return core.StoredCredentials{}, core.AccountInfo{}, fmt.Errorf("bearer: %q is not a value bearer takes; it takes %s", name, SuppliedToken)
		}
	}
	p := payload{Token: in.Supplied[SuppliedToken]}
	if err := p.check(); err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	raw, err := json.Marshal(p)
	if err != nil {
		return core.StoredCredentials{}, core.AccountInfo{}, err
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: raw}, core.AccountInfo{}, nil
}

// Retrieve hands the token out with no expiry, and stored as it is, so the resolver has
// nothing to persist.
func (*Scheme) Retrieve(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest) (core.AccessCredential, core.StoredCredentials, error) {
	if _, err := open(stored); err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	return core.NewAccessCredential(Name, time.Time{}, stored.Payload), stored, nil
}

// Wrap sets Authorization: Bearer on a clone of every request (RFC 6750 section 2.1). A
// credential this scheme did not issue fails each request instead.
func (*Scheme) Wrap(base http.RoundTripper, c core.AccessCredential) http.RoundTripper {
	var p payload
	if c.Scheme != Name || json.Unmarshal(c.Secret(), &p) != nil || p.check() != nil {
		return refuse{}
	}
	return authorization{base: base, value: prefix + p.Token}
}

// Classify is oauth2code's reading of a provider's answer (RFC 6750 section 3 challenges,
// RFC 9110 statuses, 429), with one more rule: a 401 is InvalidGrant. RFC 9110 section
// 15.5.2 says the request «lacks valid authentication credentials», and a static token is
// never renewed, so only a new token helps.
func (s *Scheme) Classify(resp *http.Response, body []byte, err error) core.Outcome {
	outcome := s.answers.Classify(resp, body, err)
	if outcome.Kind == core.OutcomeOK && resp != nil && resp.StatusCode == http.StatusUnauthorized {
		return core.Outcome{Kind: core.OutcomeInvalidGrant}
	}
	return outcome
}

// Revoke sends nothing and returns ErrNotRevocable: the token lives on at the provider.
func (*Scheme) Revoke(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest) error {
	if _, err := open(stored); err != nil {
		return err
	}
	return ErrNotRevocable
}

// open reads the payload this scheme sealed. Its errors never quote the payload.
func open(stored core.StoredCredentials) (payload, error) {
	if stored.Scheme != Name || stored.Version != payloadVersion {
		return payload{}, fmt.Errorf("bearer: stored credentials are %q version %d, not %q version %d", stored.Scheme, stored.Version, Name, payloadVersion)
	}
	var p payload
	if json.Unmarshal(stored.Payload, &p) != nil {
		return payload{}, errors.New("bearer: stored credentials payload is unreadable")
	}
	if err := p.check(); err != nil {
		return payload{}, err
	}
	return p, nil
}

// check refuses a token that is not an RFC 6750 b64token. Its errors never quote the token.
func (p payload) check() error {
	switch {
	case p.Token == "":
		return errors.New("bearer: the token is required")
	case !b64token.MatchString(p.Token):
		return errors.New("bearer: the token is not an RFC 6750 b64token: letters, digits and -._~+/ then any = padding")
	}
	return nil
}

// authorization is the RoundTripper Wrap returns.
type authorization struct {
	base  http.RoundTripper
	value string
}

// RoundTrip sets the header on a clone: net/http's RoundTripper contract is that
// «RoundTrip should not modify the request».
func (a authorization) RoundTrip(r *http.Request) (*http.Response, error) {
	out := r.Clone(r.Context())
	out.Header.Set("Authorization", a.value)
	return a.base.RoundTrip(out)
}

type refuse struct{}

func (refuse) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		// net/http's RoundTripper contract: it «must always close the body, including on
		// errors».
		_ = r.Body.Close()
	}
	return nil, errNoToken
}
