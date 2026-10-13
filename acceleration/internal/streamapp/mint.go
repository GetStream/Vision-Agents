package streamapp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

// mintTimeout is how long one request to the gateway's mint endpoint may take.
const mintTimeout = 10 * time.Second

// Mint asks the gateway in front of the router for tokens in an app the router holds no
// secret of: a server token for an app that has sent nothing for a while, and the user
// tokens an agent joins a call with or a guest is handed.
type Mint struct {
	url    string
	bearer string
	client *http.Client
}

// NewMint is the gateway's mint endpoint at url, called with the bearer token it expects.
func NewMint(url, bearer string) (*Mint, error) {
	if strings.TrimSpace(url) == "" || strings.TrimSpace(bearer) == "" {
		return nil, errors.New("streamapp: the gateway's mint endpoint needs its url and the bearer it expects")
	}
	return &Mint{url: strings.TrimRight(url, "/"), bearer: bearer, client: &http.Client{Timeout: mintTimeout}}, nil
}

// Minted is a token the gateway minted, and when it stops being accepted.
type Minted struct {
	Token     Secret
	ExpiresAt time.Time
}

// ServerToken is a server token for an app, by one of the app's keys.
func (m *Mint) ServerToken(ctx context.Context, app int64, apiKey string) (Minted, error) {
	return m.mint(ctx, app, "server-token", map[string]any{"api_key": apiKey})
}

// UserToken is a token for a user of an app, lasting ttl.
func (m *Mint) UserToken(ctx context.Context, app int64, apiKey, userID string, ttl time.Duration) (Minted, error) {
	return m.mint(ctx, app, "user-token", map[string]any{
		"api_key": apiKey, "user_id": userID, "ttl_seconds": int64(ttl / time.Second),
	})
}

// ErrCannotMint is an identity with neither a secret to sign a user's token with nor a
// gateway to ask for one.
var ErrCannotMint = errors.New("streamapp: nothing can mint a user token in this app")

// MintUserToken is a token for a user of a bound identity's app, lasting ttl. An identity
// holding its app's secret signs it here; one acting with a forwarded token has the
// gateway sign it, since a server token cannot.
func (c *Clients) MintUserToken(ctx context.Context, bound Bound, userID string, ttl time.Duration) (string, error) {
	if !bound.Identity.Secret.Empty() {
		return bound.Client.CreateToken(userID, getstream.WithExpiration(ttl))
	}
	forwarded, ok := c.Forwarded()
	if !ok || forwarded.Mint() == nil || bound.Identity.Token.Empty() {
		return "", ErrCannotMint
	}
	minted, err := forwarded.Mint().UserToken(ctx, bound.Identity.StreamApp, bound.Identity.APIKey, userID, ttl)
	if err != nil {
		return "", err
	}
	return minted.Token.Reveal(), nil
}

func (m *Mint) mint(ctx context.Context, app int64, kind string, body map[string]any) (Minted, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return Minted{}, err
	}
	url := m.url + "/internal/v1/apps/" + strconv.FormatInt(app, 10) + "/" + kind
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(encoded))
	if err != nil {
		return Minted{}, err
	}
	request.Header.Set("Authorization", "Bearer "+m.bearer)
	request.Header.Set("Content-Type", "application/json")
	response, err := m.client.Do(request)
	if err != nil {
		return Minted{}, fmt.Errorf("streamapp: minting a %s for app %d: %w", kind, app, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 1<<10))
		return Minted{}, fmt.Errorf("streamapp: the gateway minted no %s for app %d: %s: %s",
			kind, app, response.Status, strings.TrimSpace(string(detail)))
	}
	var minted struct {
		Token     string    `json:"token"`
		ExpiresAt time.Time `json:"expires_at"`
	}
	if err := json.NewDecoder(response.Body).Decode(&minted); err != nil {
		return Minted{}, fmt.Errorf("streamapp: reading the %s minted for app %d: %w", kind, app, err)
	}
	if minted.Token == "" {
		return Minted{}, fmt.Errorf("streamapp: the gateway minted an empty %s for app %d", kind, app)
	}
	return Minted{Token: NewSecret(minted.Token), ExpiresAt: minted.ExpiresAt}, nil
}
