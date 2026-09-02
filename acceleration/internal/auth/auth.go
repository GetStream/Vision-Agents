// Package auth decides who a request is from.
//
// Two modes, because the two deployments have different shapes. Behind a proxy that has
// already authenticated the caller there is nothing left for the router to verify, so it
// reads the principal the proxy names. Standing on its own it verifies an API key and a
// token signed with that key's secret.
package auth

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

// Mode names how a request proves who it is.
type Mode string

const (
	// NoAuth trusts the headers a proxy in front of the router sets, and is only safe
	// when nothing else can reach the router. The proxy authenticates, rate limits, and
	// overwrites the two headers below so a caller cannot name itself.
	NoAuth Mode = "noauth"
	// APIKey verifies an API key and a token signed with the secret belonging to it.
	APIKey Mode = "api_key"
)

// ParseMode reads a mode from configuration.
func ParseMode(value string) (Mode, error) {
	switch Mode(strings.TrimSpace(value)) {
	case "", NoAuth:
		return NoAuth, nil
	case APIKey:
		return APIKey, nil
	default:
		return "", fmt.Errorf("auth: unknown mode %q, want %q or %q", value, NoAuth, APIKey)
	}
}

const (
	// OrganizationHeader and AppHeader are what the proxy names the caller in. They are
	// read in NoAuth mode and ignored entirely in APIKey mode, where the key decides who
	// the caller is and a header would only be a way around it.
	OrganizationHeader = "X-Stream-Organization-Id"
	AppHeader          = "X-Stream-App-Id"
	// APIKeyHeader carries the public half of the credential.
	APIKeyHeader = "X-Api-Key"
	// AuthTypeHeader names which kind of credential the caller believes it is presenting.
	// It has no query parameter counterpart on purpose: a browser WebSocket cannot set a
	// header, so there is no way for one to claim to be a backend.
	AuthTypeHeader = "Stream-Auth-Type"
	// APIKeyParam and TokenParam carry the same two values on a socket, because a browser
	// WebSocket cannot set a header.
	APIKeyParam = "api_key"
	TokenParam  = "token"
	// CustomerHeader and CustomerParam name the tenant directly, with no organization
	// around it. They are what a local deployment running without a proxy and without keys
	// uses, and they are read only in NoAuth mode.
	CustomerHeader = "X-Customer-Id"
	CustomerParam  = "customer_id"
)

const (
	// AuthTypeServer is a request an integration makes for itself, from a process the
	// customer runs. AuthTypeJWT is one it makes on an end user's behalf.
	//
	// What a caller naming neither is taken to be depends on the mode, and each way round
	// is right for its own: APIKey verifies, so it assumes the weaker of the two, while
	// NoAuth believes what it is told and a deployment with no proxy in front of it is a
	// local one where every caller is a backend.
	AuthTypeServer = "server"
	AuthTypeJWT    = "jwt"
)

// ErrUnauthenticated is every authentication failure. It is one error rather than several
// because telling a caller that the key was real but the token was not is a way to find
// out which keys exist.
var ErrUnauthenticated = errors.New("auth: unauthenticated")

// Principal is who a request is from. AppID is the tenant: it is what rows are keyed by,
// and what CustomerFrom reports.
type Principal struct {
	OrganizationID string
	AppID          string
	// ServerSide is whether the caller is a process the customer runs rather than an end
	// user's device. It is what the paths that configure an agent ask for: a browser
	// holding a token its own backend minted may hold a conversation, and may not rewrite
	// the agent holding it or replace what the agent knows.
	ServerSide bool
}

// App is what an API key resolves to.
type App struct {
	OrganizationID string
	AppID          string
	// Secret signs the caller's token, so it is held recoverably rather than hashed.
	Secret string
}

// Lookup resolves the public half of a credential to the app holding it. It returns
// ErrUnauthenticated when there is no such key, or it has been revoked or has expired.
type Lookup func(ctx context.Context, key string) (App, error)

// Authenticator resolves the principal a request carries.
type Authenticator interface {
	Authenticate(ctx context.Context, r *http.Request) (Principal, error)
}

// New returns the authenticator for a mode. APIKey needs somewhere to look keys up, which
// is Postgres, so a deployment without a store cannot run it.
func New(mode Mode, lookup Lookup) (Authenticator, error) {
	switch mode {
	case NoAuth:
		return proxied{}, nil
	case APIKey:
		if lookup == nil {
			return nil, fmt.Errorf("auth: %s needs a store to look keys up in", APIKey)
		}
		return keyed{lookup: lookup}, nil
	default:
		return nil, fmt.Errorf("auth: unknown mode %q", mode)
	}
}

// proxied reads the principal a proxy named.
type proxied struct{}

func (proxied) Authenticate(_ context.Context, r *http.Request) (Principal, error) {
	app := strings.TrimSpace(r.Header.Get(AppHeader))
	if app == "" {
		app = strings.TrimSpace(r.Header.Get(CustomerHeader))
	}
	if app == "" {
		app = strings.TrimSpace(r.URL.Query().Get(CustomerParam))
	}
	if app == "" {
		return Principal{}, ErrUnauthenticated
	}
	return Principal{
		OrganizationID: strings.TrimSpace(r.Header.Get(OrganizationHeader)),
		AppID:          app,
		// The auth type is believed for the same reason the app id is: the proxy has
		// already verified the credential and overwrites this header rather than
		// forwarding the caller's own. Saying nothing means server-side, because a
		// deployment with no proxy in front is a local one where every caller is, and
		// the alternative is a local router that refuses to accept an agent config.
		ServerSide: !strings.EqualFold(authTypeOf(r), AuthTypeJWT),
	}, nil
}

// keyed verifies an API key and the token signed with its secret.
type keyed struct {
	lookup Lookup
}

func (k keyed) Authenticate(ctx context.Context, r *http.Request) (Principal, error) {
	key, token := credentials(r)
	if key == "" || token == "" {
		return Principal{}, ErrUnauthenticated
	}

	app, err := k.lookup(ctx, key)
	if err != nil {
		return Principal{}, ErrUnauthenticated
	}

	// The method is pinned rather than taken from the token, because a token is allowed to
	// name its own algorithm and "none" is one of the names.
	claims := jwt.MapClaims{}
	_, err = jwt.ParseWithClaims(token, claims,
		func(*jwt.Token) (any, error) { return []byte(app.Secret), nil },
		jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Alg()}),
		jwt.WithExpirationRequired(),
	)
	if err != nil {
		return Principal{}, ErrUnauthenticated
	}

	return Principal{
		OrganizationID: app.OrganizationID,
		AppID:          app.AppID,
		// Both halves have to agree, and they fail closed in opposite directions. The
		// header cannot promote a request on its own because nothing signs it; the claim
		// cannot either, so a server token handed to a browser that sends the client
		// header is treated as the client it is.
		ServerSide: strings.EqualFold(authTypeOf(r), AuthTypeServer) && serverToken(claims),
	}, nil
}

// authTypeOf reads the caller's declaration, which is absent from every request written
// before there was anything to declare.
func authTypeOf(r *http.Request) string {
	return strings.TrimSpace(r.Header.Get(AuthTypeHeader))
}

// serverToken reads Stream's own marking of a server-side token: `server` set, and no
// `user_id`, since a token naming a user is one minted for that user to hold. The flag is
// written as a boolean by some SDKs and as the string "true" by others, so both are read.
func serverToken(claims jwt.MapClaims) bool {
	if _, named := claims["user_id"]; named {
		return false
	}
	switch flag := claims["server"].(type) {
	case bool:
		return flag
	case string:
		return strings.EqualFold(flag, "true")
	default:
		return false
	}
}

// credentials pulls the key and the token off a request. The query string is the socket's
// way in, since a browser WebSocket carries no headers of its own.
func credentials(r *http.Request) (key, token string) {
	key = strings.TrimSpace(r.Header.Get(APIKeyHeader))
	if header := strings.TrimSpace(r.Header.Get("Authorization")); header != "" {
		if rest, found := strings.CutPrefix(header, "Bearer "); found {
			token = strings.TrimSpace(rest)
		}
	}
	if key == "" {
		key = strings.TrimSpace(r.URL.Query().Get(APIKeyParam))
	}
	if token == "" {
		token = strings.TrimSpace(r.URL.Query().Get(TokenParam))
	}
	return key, token
}
