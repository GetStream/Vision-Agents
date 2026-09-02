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
	for _, value := range []string{"", "noauth"} {
		mode, err := ParseMode(value)
		require.NoError(t, err)
		require.Equal(t, NoAuth, mode)
	}

	mode, err := ParseMode("api_key")
	require.NoError(t, err)
	require.Equal(t, APIKey, mode)

	_, err = ParseMode("open")
	require.Error(t, err)
}

func TestNoAuth(t *testing.T) {
	authenticator, err := New(NoAuth, nil)
	require.NoError(t, err)

	t.Run("reads the principal the proxy named", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodGet, "/v1/calls", nil)
		r.Header.Set(AppHeader, "app-1")
		r.Header.Set(OrganizationHeader, "org-1")

		principal, err := authenticator.Authenticate(context.Background(), r)
		require.NoError(t, err)
		require.Equal(t, Principal{OrganizationID: "org-1", AppID: "app-1"}, principal)
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

	t.Run("resolves the app the key belongs to", func(t *testing.T) {
		principal, err := authenticator.Authenticate(context.Background(),
			request(key, signed(t, secret, time.Hour)))
		require.NoError(t, err)
		require.Equal(t, Principal{OrganizationID: "org-1", AppID: "app-1"}, principal)
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
