// Package auth decides who a request is from.
//
// Four modes, because four deployments have four different answers already available to
// them. Standing on its own the router verifies an API key and a token signed with that
// key's secret. Behind a proxy that has already authenticated the caller there is nothing
// left to verify, so it reads the principal the proxy names. On a laptop there is nobody
// to authenticate and nothing worth protecting, so it asks for nothing. And a deployment
// with an answer of its own supplies an Authenticator instead of choosing between these.
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
	// APIKey verifies an API key and a token signed with the secret belonging to it.
	//
	// It is the default. A default that trusts whatever reaches it is only correct when
	// something else guarantees that nothing does, and that guarantee is a NetworkPolicy
	// in one deployment and nothing at all in the next: it cannot be what a deployment
	// gets by saying nothing. This one fails closed instead, refusing to start without
	// the store and the key encryption key rather than refusing every request later.
	APIKey Mode = "api_key"
	// Proxy trusts the headers something in front of the router sets, and is only safe
	// when nothing else can reach the router. The proxy authenticates, rate limits, and
	// overwrites those headers so a caller cannot name itself.
	Proxy Mode = "proxy"
	// NoAuth asks for nothing. The tenant comes from the customer header and every caller
	// is that customer's own backend, because there is nothing here that could tell one
	// from anything else. It is for a laptop.
	NoAuth Mode = "noauth"
	// Custom is an Authenticator the deployment supplies itself. New does not build one:
	// there is nothing to build, and the mode exists so that configuration naming it is
	// understood rather than rejected.
	Custom Mode = "custom"
)

// ParseMode reads a mode from configuration.
func ParseMode(value string) (Mode, error) {
	switch mode := Mode(strings.TrimSpace(value)); mode {
	case "":
		return APIKey, nil
	case APIKey, Proxy, NoAuth, Custom:
		return mode, nil
	default:
		return "", fmt.Errorf("auth: unknown mode %q, want %q, %q, %q or %q",
			value, APIKey, Proxy, NoAuth, Custom)
	}
}

const (
	// OrganizationHeader and AppHeader are what the proxy names the caller in. They are
	// read in Proxy mode and ignored entirely in APIKey mode, where the key decides who
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
	// uses, and they are read in NoAuth and Proxy modes.
	CustomerHeader = "X-Customer-Id"
	CustomerParam  = "customer_id"
	// UserHeader and UserParam name the end user the request is for, which is what a daily
	// limit is counted against and who owns the sessions opened for them.
	//
	// In Proxy mode they are whatever the proxy decided. In APIKey mode the token's
	// user_id claim is the answer for an end user, since a header a caller writes itself
	// is a limit a caller can reset — but a server-side caller is read from the header,
	// because one that holds the secret could mint a token for any user it liked and so
	// has nothing to gain by lying. They are not read at all in NoAuth mode, where there
	// is no credential behind them to make them worth anything.
	UserHeader = "X-Stream-User-Id"
	UserParam  = "user_id"
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
	// AuthTypeAnonymous is Stream's own name for a caller presenting no token at all.
	// Such a caller may still go by a name, and nothing has checked it.
	AuthTypeAnonymous = "anonymous"
)

// ErrUnauthenticated is every authentication failure. It is one error rather than several
// because telling a caller that the key was real but the token was not is a way to find
// out which keys exist.
var ErrUnauthenticated = errors.New("auth: unauthenticated")

// ErrLevelRefused says the caller proved who it is and the app does not admit that sort of
// user. It is kept apart from ErrUnauthenticated because the two are opposite advice: one
// caller should go and find a credential, and the other has a perfectly good one and needs
// to be told that this app does not take guests. Nothing is leaked by the distinction,
// since it is only ever reached by a caller that already authenticated.
var ErrLevelRefused = errors.New("auth: this app does not admit this kind of user")

// Kind is what sort of caller a request is from. It is recorded alongside the user id
// wherever one person's things have to be kept from another's, because the id on its own
// does not say whether anybody checked it.
type Kind string

const (
	// KindServer is a process the customer runs, acting for itself.
	KindServer Kind = "server"
	// KindAuthenticated is an end user whose token was verified.
	KindAuthenticated Kind = "authenticated"
	// KindGuest is an end user whose token was verified and who Stream issued as a guest:
	// a real user with a temporary account rather than one the customer knows. It is
	// treated exactly as KindAuthenticated is, and kept apart from it only so a caller
	// can be told which it was.
	KindGuest Kind = "guest"
	// KindAnonymous is an end user who presented no token. Whatever name such a caller
	// goes by is a claim nobody checked, which is why the kind travels with it.
	KindAnonymous Kind = "anonymous"
)

// Verified reports whether a token was checked for this kind of caller, which is what
// makes the user id worth anything.
func (k Kind) Verified() bool {
	return k == KindServer || k == KindAuthenticated || k == KindGuest
}

// Principal is who a request is from. AppID is the tenant: it is what rows are keyed by,
// and what CustomerFrom reports.
type Principal struct {
	OrganizationID string
	AppID          string
	// UserID is the end user the caller is acting for, which is who a daily limit is
	// counted against and who owns the sessions they open. It is empty for a request a
	// backend makes for itself, since there is no user behind one.
	UserID string
	// Kind is what sort of caller this is. It qualifies UserID: an anonymous caller may
	// name any user id it likes, so the name alone cannot be what one person's session is
	// kept from another by.
	Kind Kind
	// ServerSide is whether the caller is a process the customer runs rather than an end
	// user's device. It is what every operation not marked client-accessible asks for: a
	// browser holding a token its own backend minted may hold a conversation, and may not
	// rewrite the agent holding it or replace what the agent knows.
	ServerSide bool
}

// App is what an API key resolves to.
type App struct {
	OrganizationID string
	AppID          string
	// Secret signs the caller's token, so it is held recoverably rather than hashed.
	Secret string
	// Levels is which sorts of end user this app turns away.
	Levels Levels
}

// Levels says which sorts of end user an app refuses.
//
// It is written as refusals rather than permissions so that the zero value admits
// everybody. Every mode but APIKey resolves no app at all, and an app that has never been
// configured has no settings document, so the zero value is what most callers are measured
// against: if it denied, turning authentication on would lock out every end user of every
// deployment that had not yet written a row.
//
// A backend is never refused. It holds the secret, so it could mint a token for any user
// it liked, and a switch it can turn off for itself is not a control.
type Levels struct {
	NoAnonymous bool
	NoGuest     bool
}

// Admits reports whether an app takes this sort of caller.
func (l Levels) Admits(kind Kind) bool {
	switch kind {
	case KindAnonymous:
		return !l.NoAnonymous
	case KindGuest:
		return !l.NoGuest
	default:
		return true
	}
}

// Lookup resolves the public half of a credential to the app holding it. It returns
// ErrUnauthenticated when there is no such key, or it has been revoked or has expired.
type Lookup func(ctx context.Context, key string) (App, error)

// Authenticator resolves the principal a request carries.
type Authenticator interface {
	Authenticate(ctx context.Context, r *http.Request) (Principal, error)
}

// Func adapts a plain function to an Authenticator, which is the whole of what a
// deployment answering the question its own way has to write.
type Func func(ctx context.Context, r *http.Request) (Principal, error)

// Authenticate calls f.
func (f Func) Authenticate(ctx context.Context, r *http.Request) (Principal, error) {
	return f(ctx, r)
}

// New returns the authenticator for a mode. APIKey needs somewhere to look keys up, which
// is Postgres, so a deployment without a store cannot run it. Custom has nothing to build:
// the authenticator is the deployment's, and it is passed to the server directly.
func New(mode Mode, lookup Lookup) (Authenticator, error) {
	switch mode {
	case NoAuth:
		return open{}, nil
	case Proxy:
		return proxied{}, nil
	case APIKey:
		if lookup == nil {
			return nil, fmt.Errorf("auth: %s needs a store to look keys up in", APIKey)
		}
		return keyed{lookup: lookup}, nil
	case Custom:
		return nil, fmt.Errorf("auth: %s builds no authenticator; supply one with api.WithAuthenticator", Custom)
	default:
		return nil, fmt.Errorf("auth: unknown mode %q", mode)
	}
}

// open asks for nothing and takes every caller for the customer's own backend.
//
// The user and the auth type are deliberately not read. In Proxy mode they are worth
// something because something verified them and overwrote them; here nothing did, so
// reading them would only be a way for a caller to name itself anything it liked, and the
// name it chose would then be what one person's sessions were kept from another's by. A
// deployment that wants end users to be told apart has a mode for that.
type open struct{}

func (open) Authenticate(_ context.Context, r *http.Request) (Principal, error) {
	app := strings.TrimSpace(r.Header.Get(CustomerHeader))
	if app == "" {
		app = strings.TrimSpace(r.URL.Query().Get(CustomerParam))
	}
	if app == "" {
		return Principal{}, ErrUnauthenticated
	}
	return Principal{AppID: app, Kind: KindServer, ServerSide: true}, nil
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
	// The auth type is believed for the same reason the app id is: the proxy has already
	// verified the credential and overwrites this header rather than forwarding the
	// caller's own. Saying nothing means server-side, because a proxy that has classified
	// a caller says so, and one that says nothing is a deployment whose callers are all
	// backends.
	declared := authTypeOf(r)
	kind := KindServer
	switch {
	case strings.EqualFold(declared, AuthTypeAnonymous):
		kind = KindAnonymous
	case strings.EqualFold(declared, AuthTypeJWT):
		kind = KindAuthenticated
	}
	return Principal{
		OrganizationID: strings.TrimSpace(r.Header.Get(OrganizationHeader)),
		AppID:          app,
		UserID:         user(r),
		Kind:           kind,
		ServerSide:     kind == KindServer,
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

	// Both halves have to agree, and they fail closed in opposite directions. The header
	// cannot promote a request on its own because nothing signs it; the claim cannot
	// either, so a server token handed to a browser that sends the client header is
	// treated as the client it is.
	serverSide := strings.EqualFold(authTypeOf(r), AuthTypeServer) && serverToken(claims)

	// A verified token naming a user is the only thing that makes a user id worth
	// anything. One that names nobody leaves the caller anonymous, free to go by the
	// unverified name in the header the way a proxied caller does, and kept apart from
	// every verified user by the kind rather than by the name.
	principal := Principal{
		OrganizationID: app.OrganizationID,
		AppID:          app.AppID,
		UserID:         userClaim(claims),
		Kind:           KindAuthenticated,
		ServerSide:     serverSide,
	}
	switch {
	case serverSide:
		principal.Kind = KindServer
		// A server token names no user, by definition: one that named a user would be a
		// token minted for that user to hold. The header is how a backend says which of
		// its users it is acting for, and it is worth believing here for the same reason
		// it would be pointless to lie in — a caller holding the secret can mint a token
		// for anyone. What it buys is that the sessions it opens belong to that user, so
		// the user's own device can reach them afterwards.
		principal.UserID = user(r)
	case principal.UserID == "":
		principal.Kind = KindAnonymous
		principal.UserID = user(r)
	case guestToken(claims):
		principal.Kind = KindGuest
	}

	// Checked after the kind is settled, because which levels an app admits is a question
	// about the kind rather than about the credential. A backend is never refused.
	//
	// The principal goes back beside the error, unlike every other failure here, so that
	// the refusal can be logged as the level it was. Nothing is leaked by that: this is
	// the one failure reached by a caller that has already proved who it is.
	if !app.Levels.Admits(principal.Kind) {
		return principal, ErrLevelRefused
	}
	return principal, nil
}

// authTypeOf reads the caller's declaration, which is absent from every request written
// before there was anything to declare.
func authTypeOf(r *http.Request) string {
	return strings.TrimSpace(r.Header.Get(AuthTypeHeader))
}

// user reads the end user a proxy named, from the header or from the socket's query string.
func user(r *http.Request) string {
	if named := strings.TrimSpace(r.Header.Get(UserHeader)); named != "" {
		return named
	}
	return strings.TrimSpace(r.URL.Query().Get(UserParam))
}

// userClaim reads the user a token was minted for. Only a string is accepted: a claim of
// some other shape is a token that does not name a user, not a user with an unusual name.
func userClaim(claims jwt.MapClaims) string {
	named, _ := claims["user_id"].(string)
	return strings.TrimSpace(named)
}

// guestToken reads whether the user this token names is one Stream issued as a guest. It
// is the role claim a guest token carries. A guest is a verified user with a temporary
// account and reaches exactly what a permanent one does; the kind is kept apart so a
// caller can be told which it was talking to, and so an app can decline to take them.
func guestToken(claims jwt.MapClaims) bool {
	role, _ := claims["role"].(string)
	return strings.EqualFold(strings.TrimSpace(role), string(KindGuest))
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
