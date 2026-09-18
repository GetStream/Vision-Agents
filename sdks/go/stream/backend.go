// Package stream is the client half of the acceleration backend: a whole voice or text
// pipeline that runs there rather than here, and the phone numbers it answers on.
package stream

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

const (
	// URLEnv names the router this SDK talks to.
	URLEnv = "STREAM_ACCELERATION_URL"
	// CustomerEnv names who the work is billed to.
	CustomerEnv = "STREAM_ACCELERATION_CUSTOMER_ID"
	// AuthenticateEnv switches on the app credential, for a router reached through the
	// authenticating proxy. It is opt-in rather than inferred from the presence of a
	// credential, because a Stream key and secret are in the environment for plenty of
	// reasons that have nothing to do with how this router is reached.
	AuthenticateEnv = "STREAM_ACCELERATION_AUTHENTICATE"
	// APIKeyEnv and APISecretEnv are the Stream app a hosted router is reached as. The
	// proxy in front of one authenticates the app rather than trusting the header below.
	APIKeyEnv    = "STREAM_API_KEY"
	APISecretEnv = "STREAM_API_SECRET"
	// CustomerHeader carries the identity every request and every cost row is keyed by.
	CustomerHeader = "X-Customer-Id"
	// APIKeyHeader names the Stream app a request is made for.
	APIKeyHeader = "api_key"
	// AuthTypeHeader says how the token is meant to be read. AuthTypeJWT is the only answer
	// the proxy accepts; anonymous access is refused outright.
	//
	// It is only ever sent to the proxy, which verifies the token and then strips this
	// header before the router sees it. Sent to a router directly it would mean the
	// opposite of what is wanted: with nothing in front to have checked it, a router reads
	// this as a caller acting for an end user's device, and a device may hold a
	// conversation but not rewrite the agent holding it.
	AuthTypeHeader = "stream-auth-type"
	AuthTypeJWT    = "jwt"
	// AuthorizationHeader carries the token signed with the app's secret.
	AuthorizationHeader = "authorization"
	// DefaultURL is where a router started with no address of its own listens.
	DefaultURL = "http://localhost:8080"
)

// tokenValidity is how long a minted token lasts. Short, because it is minted per request
// and a stolen one should stop working sooner than the credential behind it.
const tokenValidity = time.Hour

var (
	// ErrNoCustomer is returned when nothing says who is calling: neither a customer id
	// for a router that trusts one, nor a credential naming the app it belongs to.
	ErrNoCustomer = errors.New("stream: who is calling is not set; set " + CustomerEnv +
		", or " + AuthenticateEnv + " for a router behind the proxy")
	// ErrNoCredential is returned when authentication is asked for without the credential
	// it needs. Falling back to the customer header instead would send a request the proxy
	// refuses, and report it as whatever the proxy says rather than as what it is.
	ErrNoCredential = errors.New("stream: " + AuthenticateEnv + " needs " + APIKeyEnv +
		" and " + APISecretEnv)
)

// Backend is where the acceleration router is, and who is calling it.
type Backend struct {
	// URL is the router's base URL. Empty falls back to STREAM_ACCELERATION_URL.
	URL string
	// CustomerID is the identity every request and every cost row is keyed by. Empty falls
	// back to STREAM_ACCELERATION_CUSTOMER_ID. A router behind the authenticating proxy
	// works this out from the credential instead and ignores what it is told.
	CustomerID string
	// Authenticate says the router is reached through the proxy, so requests carry the app
	// credential below rather than naming a customer. False falls back to
	// STREAM_ACCELERATION_AUTHENTICATE.
	Authenticate bool
	// APIKey and APISecret are the Stream app to authenticate as, read only when
	// Authenticate is set. Empty falls back to STREAM_API_KEY and STREAM_API_SECRET.
	APIKey    string
	APISecret string
	// UserID is the end user this SDK is acting for, if it is acting for one. Empty speaks
	// for the app itself, which is what a backend does, and is what keeps the per-user
	// daily limits out of it.
	UserID string
	// Token is a credential somebody else minted for UserID to hold, used instead of
	// signing one here.
	//
	// It is what a process without the secret has: a worker handed a user's token, or a test
	// standing in for a device. With it, APISecret is not needed at all, which is the point
	// -- a token is the whole credential and the secret behind it could mint any other.
	Token string
	// HTTPClient is used for both the REST calls and the socket handshake. Nil uses the
	// default client.
	HTTPClient *http.Client
}

// Resolve fills in whatever the environment knows and refuses a backend nobody is billed for.
func (b Backend) Resolve() (Backend, error) {
	if b.URL == "" {
		b.URL = os.Getenv(URLEnv)
	}
	if b.URL == "" {
		b.URL = DefaultURL
	}
	b.URL = strings.TrimSuffix(b.URL, "/")

	if b.CustomerID == "" {
		b.CustomerID = os.Getenv(CustomerEnv)
	}

	if !b.Authenticate {
		if set := os.Getenv(AuthenticateEnv); set != "" {
			on, err := strconv.ParseBool(set)
			if err != nil {
				return b, fmt.Errorf("stream: %s=%q is not a true or false: %w", AuthenticateEnv, set, err)
			}
			b.Authenticate = on
		}
	}
	if b.Authenticate {
		if b.APIKey == "" {
			b.APIKey = os.Getenv(APIKeyEnv)
		}
		if b.APISecret == "" {
			b.APISecret = os.Getenv(APISecretEnv)
		}
		// A token handed in stands in for the secret, since it is already the thing the
		// secret would have been used to make.
		if b.APIKey == "" || (b.APISecret == "" && b.Token == "") {
			return b, ErrNoCredential
		}
		// The credential names the app it belongs to and the proxy strips whatever this end
		// claims, so there is nothing left for a customer id to answer.
		return b, nil
	}
	if b.CustomerID == "" {
		return b, ErrNoCustomer
	}
	return b, nil
}

// token mints what the proxy verifies: a token signed with the app's secret, naming a user
// only when this SDK is acting for one.
//
// A token that names nobody speaks for the app itself, which is Stream's `server: true`, and
// is what a backend holds. That is also what leaves the per-user daily limits out of it.
func (b Backend) token() (string, error) {
	if b.Token != "" {
		return b.Token, nil
	}
	now := time.Now()
	claims := jwt.MapClaims{
		"iat": jwt.NewNumericDate(now),
		"exp": jwt.NewNumericDate(now.Add(tokenValidity)),
	}
	if b.UserID == "" {
		claims["server"] = true
	} else {
		claims["user_id"] = b.UserID
	}
	signed, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(b.APISecret))
	if err != nil {
		return "", fmt.Errorf("stream: signing a token for %s: %w", b.APIKey, err)
	}
	return signed, nil
}

// Credentials are what every request to the router carries: the app and a token signed for
// it when going through the proxy, and otherwise who is billed.
func (b Backend) Credentials() (http.Header, error) {
	header := http.Header{}
	if !b.Authenticate {
		if b.CustomerID != "" {
			header.Set(CustomerHeader, b.CustomerID)
		}
		return header, nil
	}
	token, err := b.token()
	if err != nil {
		return nil, err
	}
	header.Set(APIKeyHeader, b.APIKey)
	header.Set(AuthTypeHeader, AuthTypeJWT)
	header.Set(AuthorizationHeader, token)
	return header, nil
}

// Client is an HTTP client for the generated API, already carrying the credentials.
func (b Backend) Client() (*acceleration.ClientWithResponses, error) {
	resolved, err := b.Resolve()
	if err != nil {
		return nil, err
	}

	options := []acceleration.ClientOption{
		acceleration.WithRequestEditorFn(func(_ context.Context, request *http.Request) error {
			// Minted per request, so a client left idle longer than a token lasts does not
			// wake up holding an expired one.
			credentials, err := resolved.Credentials()
			if err != nil {
				return err
			}
			for name, values := range credentials {
				request.Header[name] = values
			}
			return nil
		}),
	}
	if resolved.HTTPClient != nil {
		options = append(options, acceleration.WithHTTPClient(resolved.HTTPClient))
	}
	return acceleration.NewClientWithResponses(resolved.URL, options...)
}

// SocketURL is the WebSocket URL for a path on the router.
func (b Backend) SocketURL(path string) string {
	switch {
	case strings.HasPrefix(b.URL, "https://"):
		return "wss://" + strings.TrimPrefix(b.URL, "https://") + path
	case strings.HasPrefix(b.URL, "http://"):
		return "ws://" + strings.TrimPrefix(b.URL, "http://") + path
	default:
		return b.URL + path
	}
}
