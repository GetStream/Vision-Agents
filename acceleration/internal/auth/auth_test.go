package auth

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/require"
)

// signed returns a token for the secret, valid for the duration given.
func signed(t *testing.T, secret string, ttl time.Duration) string {
	t.Helper()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.RegisteredClaims{
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(ttl)),
	}).SignedString([]byte(secret))
	require.NoError(t, err)
	return token
}

// signedClaims returns a token carrying claims of its own, which is how a caller says
// whether it is a backend or an end user.
func signedClaims(t *testing.T, secret string, claims jwt.MapClaims) string {
	t.Helper()
	claims["exp"] = time.Now().Add(time.Hour).Unix()
	token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(secret))
	require.NoError(t, err)
	return token
}

// lookupOf resolves one key to one app and rejects everything else.
func lookupOf(key string, app App) Lookup {
	return func(_ context.Context, presented string) (App, error) {
		if presented != key {
			return App{}, ErrUnauthenticated
		}
		return app, nil
	}
}

func TestParseMode(t *testing.T) {
	// Naming nothing gets the mode that verifies something. A default that trusts
	// whatever reaches it is only correct when something else guarantees nothing does,
	// which is not a guarantee a deployment should get by saying nothing.
	for _, value := range []string{"", "api_key"} {
		mode, err := ParseMode(value)
		require.NoError(t, err)
		require.Equal(t, APIKey, mode)
	}

	for value, want := range map[string]Mode{"proxy": Proxy, "noauth": NoAuth, "custom": Custom} {
		mode, err := ParseMode(value)
		require.NoError(t, err)
		require.Equal(t, want, mode)
	}

	_, err := ParseMode("open")
	require.Error(t, err)
}

func TestCustomBuildsNothing(t *testing.T) {
	// The mode is understood so a deployment naming it is not rejected at startup, but
	// the authenticator is the deployment's own and arrives another way.
	_, err := New(Custom, nil)
	require.ErrorContains(t, err, "WithAuthenticator")
}

func TestNoAuth(t *testing.T) {
	authenticator, err := New(NoAuth, nil)
	require.NoError(t, err)

	t.Run("takes the customer header and calls it a backend", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(CustomerHeader, "examples")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, Principal{
			AppID: "examples", Kind: KindServer, ServerSide: true,
		}, principal)
	})

	t.Run("reads the customer query parameter for a socket", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/dispatch?customer_id=examples", nil)

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, "examples", principal.AppID)
	})

	t.Run("names nobody when the request names nobody", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)

		_, err := authenticator.Authenticate(context.Background(), r)
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("ignores the headers a proxy would have set", func(t *testing.T) {
		// This is the whole reason the mode was split out of the one behind a proxy.
		// Nothing here verified any of these, so believing them would let a caller on
		// the same laptop call itself any user of any app it liked, and a user id is
		// what one person's conversations are kept from another's by.
		r := httptest.NewRequest(http.MethodGet, "/v1/calls?user_id=user-2", nil)
		r.Header.Set(CustomerHeader, "examples")
		r.Header.Set(AppHeader, "somebody-elses-app")
		r.Header.Set(OrganizationHeader, "somebody-elses-org")
		r.Header.Set(UserHeader, "user-1")
		r.Header.Set(AuthTypeHeader, AuthTypeJWT)

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, Principal{
			AppID: "examples", Kind: KindServer, ServerSide: true,
		}, principal)
	})
}

func TestProxy(t *testing.T) {
	authenticator, err := New(Proxy, nil)
	require.NoError(t, err)

	t.Run("reads the principal the proxy named", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(AppHeader, "app-1")
		r.Header.Set(OrganizationHeader, "org-1")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, Principal{
			OrganizationID: "org-1", AppID: "app-1", Kind: KindServer, ServerSide: true,
		}, principal)
	})

	t.Run("believes the auth type the proxy named", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(AppHeader, "app-1")
		r.Header.Set(AuthTypeHeader, AuthTypeJWT)

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.False(t, principal.ServerSide)
	})

	t.Run("treats a caller that names no auth type as a backend", func(t *testing.T) {
		// A proxy that has classified a caller says so, so one that says nothing is a
		// deployment whose callers are all backends.
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(CustomerHeader, "examples")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.True(t, principal.ServerSide)
	})

	t.Run("falls back to the customer header", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(CustomerHeader, "examples")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, "examples", principal.AppID)
		require.Empty(t, principal.OrganizationID)
	})

	t.Run("falls back to the customer query parameter for a socket", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/dispatch?customer_id=examples", nil)

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, "examples", principal.AppID)
	})

	t.Run("names nobody when the request names nobody", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)

		_, err := authenticator.Authenticate(context.Background(), r)
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("reads the end user the proxy named", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(AppHeader, "app-1")
		r.Header.Set(UserHeader, "user-1")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, "user-1", principal.UserID)
	})

	t.Run("reads the end user from the query parameter for a socket", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/llm/stream?customer_id=examples&user_id=user-1", nil)

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, "user-1", principal.UserID)
	})
}

func TestAPIKey(t *testing.T) {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"
	app := App{OrganizationID: "org-1", AppID: "app-1", Secret: secret}

	authenticator, err := New(APIKey, lookupOf(key, app))
	require.NoError(t, err)

	request := func(key, token string) *http.Request {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(APIKeyHeader, key)
		r.Header.Set("Authorization", "Bearer "+token)
		return r
	}

	// serverSide is a request that claims to be a backend in both of the two places a
	// caller can say so.
	serverSide := func(claims jwt.MapClaims) *http.Request {
		r := request(key, signedClaims(t, secret, claims))
		r.Header.Set(AuthTypeHeader, AuthTypeServer)
		return r
	}

	t.Run("resolves the app the key belongs to", func(t *testing.T) {
		// A token naming nobody and claiming nothing leaves the caller anonymous: the
		// key says which app it is, and nothing says which person.
		principal, err := authenticator.Authenticate(context.Background(),
			request(key, signed(t, secret, time.Hour)))
		require.NoError(t, err)
		require.Equal(t,
			Principal{OrganizationID: "org-1", AppID: "app-1", Kind: KindAnonymous}, principal)
	})

	t.Run("names the end user the token was minted for", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			request(key, signedClaims(t, secret, jwt.MapClaims{"user_id": "user-1"})))
		require.NoError(t, err)
		require.Equal(t, "user-1", principal.UserID)
	})

	t.Run("ignores the user header, since only the token names a user here", func(t *testing.T) {
		// A caller who could name their own user could reset their own limit by picking a
		// name nobody has spent anything under.
		r := request(key, signedClaims(t, secret, jwt.MapClaims{"user_id": "user-1"}))
		r.Header.Set(UserHeader, "somebody-else")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, "user-1", principal.UserID)
	})

	t.Run("names no user for a claim that is not a name", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			request(key, signedClaims(t, secret, jwt.MapClaims{"user_id": 42})))
		require.NoError(t, err)
		require.Empty(t, principal.UserID)
	})

	t.Run("names a backend when the header and the token agree", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			serverSide(jwt.MapClaims{"server": true}))
		require.NoError(t, err)
		require.True(t, principal.ServerSide)
	})

	t.Run("reads the flag written as a string, which is what Stream mints", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			serverSide(jwt.MapClaims{"server": "true"}))
		require.NoError(t, err)
		require.True(t, principal.ServerSide)
	})

	t.Run("lets a backend say which of its users it is acting for", func(t *testing.T) {
		// A server token names no user by definition, so the header is the only place a
		// backend can say whose session it is opening. Believing it costs nothing: a
		// caller holding the secret could mint a token for that user instead.
		r := serverSide(jwt.MapClaims{"server": true})
		r.Header.Set(UserHeader, "user-1")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, KindServer, principal.Kind)
		require.True(t, principal.ServerSide)
		require.Equal(t, "user-1", principal.UserID)
	})

	t.Run("keeps a token naming a user client-side, whatever the header says", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			serverSide(jwt.MapClaims{"server": true, "user_id": "user-1"}))
		require.NoError(t, err)
		require.False(t, principal.ServerSide)
	})

	t.Run("will not promote a caller on the header alone", func(t *testing.T) {
		// Nothing signs the header, so a client that sets it is still a client.
		principal, err := authenticator.Authenticate(context.Background(),
			serverSide(jwt.MapClaims{"user_id": "user-1"}))
		require.NoError(t, err)
		require.False(t, principal.ServerSide)
	})

	t.Run("will not promote a caller on the token alone", func(t *testing.T) {
		// A server token pasted into a browser is used by a browser, and says so.
		principal, err := authenticator.Authenticate(context.Background(),
			request(key, signedClaims(t, secret, jwt.MapClaims{"server": true})))
		require.NoError(t, err)
		require.False(t, principal.ServerSide)
	})

	t.Run("gives a socket no way to claim to be a backend", func(t *testing.T) {
		// The auth type has no query parameter, and a browser WebSocket sets no headers.
		token := signedClaims(t, secret, jwt.MapClaims{"server": true})
		target := "/v1/dispatch?api_key=" + key + "&token=" + token +
			"&stream_auth_type=server&" + AuthTypeHeader + "=server"

		principal, err := authenticator.Authenticate(context.Background(),
			httptest.NewRequest(http.MethodGet, target, nil))
		require.NoError(t, err)
		require.False(t, principal.ServerSide)
	})

	t.Run("takes the key and token off a socket's query string", func(t *testing.T) {
		target := "/v1/dispatch?api_key=" + key + "&token=" + signed(t, secret, time.Hour)
		principal, err := authenticator.Authenticate(context.Background(),
			httptest.NewRequest(http.MethodGet, target, nil))
		require.NoError(t, err)
		require.Equal(t, "app-1", principal.AppID)
	})

	t.Run("rejects a token signed with another secret", func(t *testing.T) {
		_, err := authenticator.Authenticate(context.Background(),
			request(key, signed(t, "vas_live_wrong", time.Hour)))
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("rejects an expired token", func(t *testing.T) {
		_, err := authenticator.Authenticate(context.Background(),
			request(key, signed(t, secret, -time.Minute)))
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("rejects a token with no expiry", func(t *testing.T) {
		token, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.RegisteredClaims{}).
			SignedString([]byte(secret))
		require.NoError(t, err)

		_, err = authenticator.Authenticate(context.Background(), request(key, token))
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("rejects an unsigned token", func(t *testing.T) {
		token, err := jwt.NewWithClaims(jwt.SigningMethodNone, jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
		}).SignedString(jwt.UnsafeAllowNoneSignatureType)
		require.NoError(t, err)

		_, err = authenticator.Authenticate(context.Background(), request(key, token))
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("rejects an unknown key", func(t *testing.T) {
		_, err := authenticator.Authenticate(context.Background(),
			request("vak_live_ffffffffffffffff00000000", signed(t, secret, time.Hour)))
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("ignores the headers a proxy would set", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(AppHeader, "app-2")
		r.Header.Set(OrganizationHeader, "org-2")
		r.Header.Set(CustomerHeader, "app-3")

		_, err := authenticator.Authenticate(context.Background(), r)
		require.ErrorIs(t, err, ErrUnauthenticated)
	})

	t.Run("needs somewhere to look keys up", func(t *testing.T) {
		_, err := New(APIKey, nil)
		require.Error(t, err)
	})
}

func TestLevelsDefaultToAdmittingEverybody(t *testing.T) {
	// The zero value is what every caller in every mode but api_key is measured against,
	// and what an app nobody has configured gets. If it denied, turning authentication on
	// would lock out every end user of every deployment that had not written a row yet.
	var none Levels
	for _, kind := range []Kind{KindServer, KindAuthenticated, KindGuest, KindAnonymous} {
		require.True(t, none.Admits(kind), string(kind))
	}
}

func TestAnAppTurnsAwayTheLevelsItDoesNotWant(t *testing.T) {
	const key, secret = "vak_live_0123456789abcdef00000000", "vas_live_s3cret"

	// An app taking neither of the two levels below a signed-in user.
	closed := App{
		OrganizationID: "org-1", AppID: "app-1", Secret: secret,
		Levels: Levels{NoAnonymous: true, NoGuest: true},
	}
	authenticator, err := New(APIKey, lookupOf(key, closed))
	require.NoError(t, err)

	presenting := func(claims jwt.MapClaims) *http.Request {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(APIKeyHeader, key)
		r.Header.Set("Authorization", "Bearer "+signedClaims(t, secret, claims))
		return r
	}

	t.Run("refuses an anonymous caller", func(t *testing.T) {
		_, err := authenticator.Authenticate(context.Background(), presenting(jwt.MapClaims{}))
		require.ErrorIs(t, err, ErrLevelRefused)
	})

	t.Run("refuses a guest", func(t *testing.T) {
		_, err := authenticator.Authenticate(context.Background(),
			presenting(jwt.MapClaims{"user_id": "user-1", "role": "guest"}))
		require.ErrorIs(t, err, ErrLevelRefused)
	})

	t.Run("says which level it turned away", func(t *testing.T) {
		// The one failure here that comes with a principal, because it is the one
		// reached by a caller that has already proved who it is: there is nothing left
		// to withhold, and an operator wants to know which level is being refused.
		refused, err := authenticator.Authenticate(context.Background(), presenting(jwt.MapClaims{}))
		require.ErrorIs(t, err, ErrLevelRefused)
		require.Equal(t, KindAnonymous, refused.Kind)
	})

	t.Run("still admits a signed-in user", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			presenting(jwt.MapClaims{"user_id": "user-1"}))
		require.NoError(t, err)
		require.Equal(t, KindAuthenticated, principal.Kind)
	})

	t.Run("never refuses the customer's own backend", func(t *testing.T) {
		// A backend holds the secret, so a switch it could turn off for itself is not a
		// control, and refusing it would only lock the customer out of their own app.
		r := presenting(jwt.MapClaims{"server": true})
		r.Header.Set(AuthTypeHeader, AuthTypeServer)

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, KindServer, principal.Kind)
	})
}

func TestCredential(t *testing.T) {
	t.Run("mints a key that validates and a secret that does not repeat", func(t *testing.T) {
		key, secret, err := NewCredential(Live)
		require.NoError(t, err)
		require.True(t, ValidKey(key))
		require.Equal(t, secret[len(secret)-4:], Last4(secret))

		other, otherSecret, err := NewCredential(Live)
		require.NoError(t, err)
		require.NotEqual(t, key, other)
		require.NotEqual(t, secret, otherSecret)
	})

	t.Run("carries the environment so a live secret is visible as one", func(t *testing.T) {
		key, secret, err := NewCredential(Test)
		require.NoError(t, err)
		require.Contains(t, key, "_test_")
		require.Contains(t, secret, "_test_")
	})

	t.Run("rejects an unknown environment", func(t *testing.T) {
		_, _, err := NewCredential("staging")
		require.Error(t, err)
	})

	t.Run("rejects a truncated or corrupted key", func(t *testing.T) {
		key, _, err := NewCredential(Live)
		require.NoError(t, err)

		require.False(t, ValidKey(key[:len(key)-1]))
		require.False(t, ValidKey(key[:len(key)-1]+"0"))
		require.False(t, ValidKey("vak_live_zzzzzzzzzzzzzzzz00000000"))
		require.False(t, ValidKey("nonsense"))
	})
}

func TestSealer(t *testing.T) {
	sealer, err := NewSealer("a passphrase")
	require.NoError(t, err)

	t.Run("returns what it was given", func(t *testing.T) {
		sealed, err := sealer.Seal("vas_live_s3cret")
		require.NoError(t, err)
		require.NotContains(t, string(sealed), "s3cret")

		opened, err := sealer.Open(sealed)
		require.NoError(t, err)
		require.Equal(t, "vas_live_s3cret", opened)
	})

	t.Run("seals the same secret differently each time", func(t *testing.T) {
		first, err := sealer.Seal("vas_live_s3cret")
		require.NoError(t, err)
		second, err := sealer.Seal("vas_live_s3cret")
		require.NoError(t, err)
		require.NotEqual(t, first, second)
	})

	t.Run("refuses a secret sealed under another key", func(t *testing.T) {
		other, err := NewSealer("a different passphrase")
		require.NoError(t, err)

		sealed, err := other.Seal("vas_live_s3cret")
		require.NoError(t, err)

		_, err = sealer.Open(sealed)
		require.Error(t, err)
	})

	t.Run("refuses a tampered ciphertext", func(t *testing.T) {
		sealed, err := sealer.Seal("vas_live_s3cret")
		require.NoError(t, err)
		sealed[len(sealed)-1] ^= 0xff

		_, err = sealer.Open(sealed)
		require.Error(t, err)
	})

	t.Run("needs a key encryption key", func(t *testing.T) {
		_, err := NewSealer("")
		require.Error(t, err)
	})
}
