// Package none is the none scheme: a connector that needs no credential, such as a public
// MCP server. A connection still exists, so it can be bound, listed and deleted like any
// other; its stored credentials hold nothing.
//
// Begin is Done. Complete takes no supplied value and seals an empty payload. Retrieve hands
// out a credential with no secret and no expiry. Wrap leaves every request as it is.
// Classify reads a provider's answer as oauth2_code does, plus one rule: a 401 is
// InvalidGrant, since the provider wants a credential this connection does not have. Revoke
// has nothing to revoke.
package none

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// Name is the registry name manifests list under schemes. It is the prototype's AuthNone
// (internal/connectors/runtime.go:15 on codex/connector-support at cf62af0d).
const Name = "none"

// payloadVersion is the shape of the empty payload. A new shape is a new version, so a sealed
// blob is always read as what it was written as.
const payloadVersion = 1

// empty is the payload: a JSON object with nothing in it, so the sealed blob is still JSON.
var empty = []byte(`{}`)

// errNothingTaken is Complete given a supplied value.
var errNothingTaken = errors.New("none: the none scheme takes no supplied values")

// Scheme is the none scheme. It holds no state and is safe for concurrent use.
type Scheme struct {
	answers *oauth2code.Scheme
}

var _ core.Scheme = (*Scheme)(nil)

// New returns the scheme. It takes no configuration.
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

// Begin is Done: there is nothing to consent to.
func (*Scheme) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{Done: true}, nil
}

// Complete seals an empty payload. A supplied value is refused rather than dropped: whoever
// sent one expected it to be used.
func (*Scheme) Complete(_ context.Context, in core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	if len(in.Supplied) > 0 {
		return core.StoredCredentials{}, core.AccountInfo{}, errNothingTaken
	}
	return core.StoredCredentials{Scheme: Name, Version: payloadVersion, Payload: append([]byte(nil), empty...)}, core.AccountInfo{}, nil
}

// Retrieve hands out a credential with no secret and no expiry, and stored as it is, so the
// resolver has nothing to persist.
func (*Scheme) Retrieve(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest) (core.AccessCredential, core.StoredCredentials, error) {
	if err := open(stored); err != nil {
		return core.AccessCredential{}, core.StoredCredentials{}, err
	}
	return core.NewAccessCredential(Name, time.Time{}, nil), stored, nil
}

// Wrap is base: there is nothing to apply, whatever credential it is handed.
func (*Scheme) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return base
}

// Classify is oauth2code's reading of a provider's answer (RFC 6750 section 3 challenges,
// RFC 9110 statuses, 429), with one more rule: a 401 is InvalidGrant. RFC 9110 section
// 15.5.2 says the request «lacks valid authentication credentials», which no retry without a
// credential changes: only a connection with another scheme helps.
func (s *Scheme) Classify(resp *http.Response, body []byte, err error) core.Outcome {
	outcome := s.answers.Classify(resp, body, err)
	if outcome.Kind == core.OutcomeOK && resp != nil && resp.StatusCode == http.StatusUnauthorized {
		return core.Outcome{Kind: core.OutcomeInvalidGrant}
	}
	return outcome
}

// Revoke sends nothing and returns nil: the provider holds nothing for this connection, so
// once it is deleted nothing of it is left anywhere.
func (*Scheme) Revoke(_ context.Context, stored core.StoredCredentials, _ core.ResolvedManifest) error {
	return open(stored)
}

// open checks the stored credentials are this scheme's empty payload.
func open(stored core.StoredCredentials) error {
	if stored.Scheme != Name || stored.Version != payloadVersion {
		return fmt.Errorf("none: stored credentials are %q version %d, not %q version %d", stored.Scheme, stored.Version, Name, payloadVersion)
	}
	if string(stored.Payload) != string(empty) {
		return errors.New("none: stored credentials payload is not empty")
	}
	return nil
}

type refuse struct{}

func (refuse) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.Body != nil {
		// net/http's RoundTripper contract: it «must always close the body, including on
		// errors».
		_ = r.Body.Close()
	}
	return nil, errors.New("none: Classify sends nothing")
}
