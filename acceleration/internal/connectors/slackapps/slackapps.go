// Package slackapps creates, updates and deletes a customer's Slack app with Slack's app
// manifest API, and rotates the app configuration token those calls take. It is the client
// the router's provider app endpoint (internal/api/connector_provider_apps.go) uses for a
// managed Slack app (core.ClientManaged, T54).
package slackapps

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// defaultBaseURL is where Slack's Web API methods are: every method page names
// POST https://slack.com/api/<method>
// (https://docs.slack.dev/reference/methods/apps.manifest.create, opened October 6, 2026).
const defaultBaseURL = "https://slack.com/api/"

// Method names, each a page under https://docs.slack.dev/reference/methods/.
const (
	methodCreate = "apps.manifest.create"
	methodUpdate = "apps.manifest.update"
	methodDelete = "apps.manifest.delete"
	methodRotate = "tooling.tokens.rotate"
)

// maxResponseBytes caps what is read of one answer. The answers here are a few hundred bytes
// of JSON (the examples on the four method pages); 1 MiB is the cap core reads a provider's
// answer with (Scheme.Classify, «first 1 MiB», internal/connectors/core/AGENTS.md).
const maxResponseBytes = 1 << 20

// Error is Slack's answer {"ok": false, "error": code} to one method: «For failure results,
// the error property will contain a short machine-readable error code»
// (https://docs.slack.dev/apis/web-api/). A code is never a secret, so it is safe to show.
type Error struct {
	Method string
	Code   string
}

func (e *Error) Error() string {
	return fmt.Sprintf("slackapps: %s answered %s", e.Method, e.Code)
}

// Is matches a sentinel below by its code alone, whatever the method.
func (e *Error) Is(target error) bool {
	var sentinel *Error
	return errors.As(target, &sentinel) && sentinel.Method == "" && sentinel.Code == e.Code
}

// The codes a caller acts on, each from the error list of the method page that names it.
var (
	// ErrAppNotFound: apps.manifest.update and .delete, «app_not_found».
	ErrAppNotFound = &Error{Code: "app_not_found"}
	// ErrInvalidRefreshToken: tooling.tokens.rotate, «invalid_refresh_token».
	ErrInvalidRefreshToken = &Error{Code: "invalid_refresh_token"}
	// ErrTokenExpired: every method here, «token_expired».
	ErrTokenExpired = &Error{Code: "token_expired"}
	// ErrInvalidAuth: every method here, «invalid_auth».
	ErrInvalidAuth = &Error{Code: "invalid_auth"}
	// ErrRateLimited: «ratelimited» in the method pages' lists, and an HTTP 429, which Slack
	// answers a rate-limited call with (https://docs.slack.dev/apis/web-api/rate-limits).
	ErrRateLimited = &Error{Code: "ratelimited"}
)

// Config is how the client reaches Slack.
type Config struct {
	// HTTP sends every call. In the router it is egress.NewClient(timeout, nil), so each call
	// dials only a checked public address; a test passes fakeprovider's Client.
	HTTP *http.Client
	// BaseURL is defaultBaseURL when empty. A test points it at fakeprovider's
	// URL + fakeprovider.PathSlackAPI.
	BaseURL string
}

// Client calls Slack's app manifest API and rotates configuration tokens. It holds no token:
// each call is given the one it spends.
type Client struct {
	http *http.Client
	base string
}

// New builds a client.
func New(cfg Config) (*Client, error) {
	if cfg.HTTP == nil {
		return nil, stack.Wrap(errors.New("slackapps: an HTTP client is required"))
	}
	base := cfg.BaseURL
	if base == "" {
		base = defaultBaseURL
	}
	if !strings.HasSuffix(base, "/") {
		base += "/"
	}
	return &Client{http: cfg.HTTP, base: base}, nil
}

// Credentials are what apps.manifest.create answers for a new app: its id and, under
// credentials, its client id, client secret and signing secret. verification_token is
// deprecated in favour of the signing secret
// (https://docs.slack.dev/authentication/verifying-requests-from-slack) and is not kept.
type Credentials struct {
	AppID         string
	ClientID      string
	ClientSecret  string
	SigningSecret string
}

// String names the app and hides both secrets, as core.StoredCredentials does.
func (c Credentials) String() string {
	return fmt.Sprintf("slackapps.Credentials{AppID: %s, ClientID: %s, ClientSecret: [redacted], SigningSecret: [redacted]}", c.AppID, c.ClientID)
}

// GoString is String, so %#v hides the secrets too.
func (c Credentials) GoString() string { return c.String() }

// LogValue keeps the secrets out of slog.
func (c Credentials) LogValue() slog.Value {
	return slog.GroupValue(slog.String("app_id", c.AppID), slog.String("client_id", c.ClientID))
}

// ConfigToken is an app configuration token and the refresh token tooling.tokens.rotate
// takes. ExpiresAt is the answer's exp: «Each app configuration token will expire 12 hours
// after it has been generated»
// (https://docs.slack.dev/app-manifests/configuring-apps-with-app-manifests#config-tokens).
type ConfigToken struct {
	Token        string
	RefreshToken string
	ExpiresAt    time.Time
}

// String hides both tokens.
func (t ConfigToken) String() string {
	return fmt.Sprintf("slackapps.ConfigToken{Token: [redacted], RefreshToken: [redacted], ExpiresAt: %s}", t.ExpiresAt.Format(time.RFC3339))
}

// GoString is String, so %#v hides the tokens too.
func (t ConfigToken) GoString() string { return t.String() }

// LogValue keeps the tokens out of slog.
func (t ConfigToken) LogValue() slog.Value {
	return slog.GroupValue(slog.Time("expires_at", t.ExpiresAt))
}

// Create makes a new app from manifest with apps.manifest.create, spending the configuration
// token token. Slack takes no idempotency key there, so a call whose answer is lost may have
// made an app; the caller serializes creates and keeps one app per customer.
func (c *Client) Create(ctx context.Context, token string, manifest Manifest) (Credentials, error) {
	encoded, err := json.Marshal(manifest)
	if err != nil {
		return Credentials{}, stack.Wrap(fmt.Errorf("slackapps: encode the manifest: %w", err))
	}
	var answer struct {
		AppID       string `json:"app_id"`
		Credentials struct {
			ClientID      string `json:"client_id"`
			ClientSecret  string `json:"client_secret"`
			SigningSecret string `json:"signing_secret"`
		} `json:"credentials"`
	}
	if err := c.call(ctx, methodCreate, token, url.Values{"manifest": {string(encoded)}}, &answer); err != nil {
		return Credentials{}, err
	}
	created := Credentials{
		AppID:         answer.AppID,
		ClientID:      answer.Credentials.ClientID,
		ClientSecret:  answer.Credentials.ClientSecret,
		SigningSecret: answer.Credentials.SigningSecret,
	}
	if created.AppID == "" || created.ClientID == "" || created.ClientSecret == "" || created.SigningSecret == "" {
		return Credentials{}, stack.Wrap(fmt.Errorf("slackapps: %s answered ok without an app id, a client id, a client secret or a signing secret", methodCreate))
	}
	return created, nil
}

// Update replaces the app's manifest with apps.manifest.update.
func (c *Client) Update(ctx context.Context, token, appID string, manifest Manifest) error {
	encoded, err := json.Marshal(manifest)
	if err != nil {
		return stack.Wrap(fmt.Errorf("slackapps: encode the manifest: %w", err))
	}
	return c.call(ctx, methodUpdate, token, url.Values{"app_id": {appID}, "manifest": {string(encoded)}}, nil)
}

// Delete removes the app with apps.manifest.delete, which «permanently removes an app created
// through app manifests». An app already gone is ErrAppNotFound.
func (c *Client) Delete(ctx context.Context, token, appID string) error {
	return c.call(ctx, methodDelete, token, url.Values{"app_id": {appID}}, nil)
}

// Rotate trades refreshToken for a new configuration token and a new refresh token with
// tooling.tokens.rotate. Whether the refresh token spent here keeps working is not on Slack's
// pages, so a caller saves the new one before anything else and never sends the old one again.
func (c *Client) Rotate(ctx context.Context, refreshToken string) (ConfigToken, error) {
	var answer struct {
		Token        string `json:"token"`
		RefreshToken string `json:"refresh_token"`
		// exp is seconds since the epoch: the page's example is iat 1633095660 and exp
		// 1633138860, 43,200 s (12 h) apart.
		Exp int64 `json:"exp"`
	}
	if err := c.call(ctx, methodRotate, "", url.Values{"refresh_token": {refreshToken}}, &answer); err != nil {
		return ConfigToken{}, err
	}
	if answer.Token == "" || answer.RefreshToken == "" || answer.Exp <= 0 {
		return ConfigToken{}, stack.Wrap(fmt.Errorf("slackapps: %s answered ok without a token, a refresh token or an expiry", methodRotate))
	}
	return ConfigToken{Token: answer.Token, RefreshToken: answer.RefreshToken, ExpiresAt: time.Unix(answer.Exp, 0).UTC()}, nil
}

// call posts form to method and decodes an ok answer into into. A token is sent as a bearer
// token: «We prefer tokens to be sent in the Authorization HTTP header»
// (https://docs.slack.dev/apis/web-api/). The body is form-encoded, which every method here
// accepts (each page's «Content types»).
func (c *Client) call(ctx context.Context, method, token string, form url.Values, into any) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+method, strings.NewReader(form.Encode()))
	if err != nil {
		return stack.Wrap(fmt.Errorf("slackapps: %s: %w", method, err))
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := c.http.Do(request)
	if err != nil {
		return stack.Wrap(fmt.Errorf("slackapps: %s: %w", method, err))
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusTooManyRequests {
		return stack.Wrap(&Error{Method: method, Code: ErrRateLimited.Code})
	}
	if response.StatusCode != http.StatusOK {
		return stack.Wrap(fmt.Errorf("slackapps: %s answered HTTP %d", method, response.StatusCode))
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes))
	if err != nil {
		return stack.Wrap(fmt.Errorf("slackapps: %s: %w", method, err))
	}
	var status struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &status); err != nil {
		return stack.Wrap(fmt.Errorf("slackapps: %s answered something that is not JSON", method))
	}
	if !status.OK {
		return stack.Wrap(&Error{Method: method, Code: status.Error})
	}
	if into == nil {
		return nil
	}
	if err := json.Unmarshal(body, into); err != nil {
		return stack.Wrap(fmt.Errorf("slackapps: %s answered an ok it could not read", method))
	}
	return nil
}
