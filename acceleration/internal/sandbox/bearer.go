package sandbox

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrNotBearer is a connection whose scheme sends no bearer token, which is the only
// credential a sandbox is given: Daytona replaces a placeholder only where a header carries
// it verbatim (https://www.daytona.io/docs/en/secrets.md, «Substitution scope»).
var ErrNotBearer = fmt.Errorf("%w: the connection's scheme sends no bearer token", ErrNoCredential)

// BearerToken is the token of the Authorization header scheme's Wrap puts on a request to
// host with credential: what code in the sandbox sends in its place.
func BearerToken(ctx context.Context, scheme core.Scheme, credential core.AccessCredential, host string) (string, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, "https://"+strings.TrimPrefix(host, "*.")+"/", nil)
	if err != nil {
		return "", stack.Wrap(err)
	}
	var sent string
	_, err = scheme.Wrap(captured(func(r *http.Request) { sent = r.Header.Get("Authorization") }), credential).RoundTrip(request)
	if err != nil {
		return "", err
	}
	token, bearer := strings.CutPrefix(sent, "Bearer ")
	if !bearer || token == "" {
		return "", stack.Wrap(ErrNotBearer)
	}
	return token, nil
}

// captured is a RoundTripper that hands see the request and sends nothing.
type captured func(*http.Request)

func (see captured) RoundTrip(r *http.Request) (*http.Response, error) {
	see(r)
	return &http.Response{StatusCode: http.StatusNoContent, Body: http.NoBody, Request: r}, nil
}
