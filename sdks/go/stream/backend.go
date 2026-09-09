// Package stream is the client half of the acceleration backend: a whole voice or text
// pipeline that runs there rather than here, and the phone numbers it answers on.
package stream

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

const (
	// URLEnv names the router this SDK talks to.
	URLEnv = "STREAM_ACCELERATION_URL"
	// CustomerEnv names who the work is billed to.
	CustomerEnv = "STREAM_ACCELERATION_CUSTOMER_ID"
	// CustomerHeader carries the identity every request and every cost row is keyed by.
	CustomerHeader = "X-Customer-Id"
	// DefaultURL is where a router started with no address of its own listens.
	DefaultURL = "http://localhost:8080"
)

// ErrNoCustomer is returned when neither the option nor the environment says who is calling.
var ErrNoCustomer = errors.New("stream: a customer id is required; set " + CustomerEnv)

// Backend is where the acceleration router is, and who is calling it.
type Backend struct {
	// URL is the router's base URL. Empty falls back to STREAM_ACCELERATION_URL.
	URL string
	// CustomerID is the identity every request and every cost row is keyed by. Empty falls
	// back to STREAM_ACCELERATION_CUSTOMER_ID.
	CustomerID string
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
	if b.CustomerID == "" {
		return b, ErrNoCustomer
	}
	return b, nil
}

// Headers are what every request to the router carries.
func (b Backend) Headers() http.Header {
	return http.Header{CustomerHeader: []string{b.CustomerID}}
}

// Client is an HTTP client for the generated API, already carrying the customer header.
func (b Backend) Client() (*acceleration.ClientWithResponses, error) {
	resolved, err := b.Resolve()
	if err != nil {
		return nil, err
	}

	options := []acceleration.ClientOption{
		acceleration.WithRequestEditorFn(func(_ context.Context, request *http.Request) error {
			request.Header.Set(CustomerHeader, resolved.CustomerID)
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
