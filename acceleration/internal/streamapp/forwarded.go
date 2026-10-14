package streamapp

import (
	"cmp"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// tokenPurpose names a sealed server token, so it opens as nothing else the keyring
	// seals, and nothing else opens as one.
	tokenPurpose = "accelerate:stream-app-token:v1"
	// refreshBefore is how long before a held token expires the gateway is asked for
	// another, so nothing is started with a token about to stop working.
	refreshBefore = time.Hour
)

// ForwardedOptions configures Forwarded. Store and Sealer are required.
type ForwardedOptions struct {
	Store  *store.Store
	Sealer *auth.Sealer
	// Mint is the gateway's mint endpoint, asked for a token when the one held runs out.
	// Without it an app is acted in only while a token it forwarded lasts.
	Mint *Mint
	// BaseURL is the Stream API every app is reached at. Empty is Stream's default.
	BaseURL string
	Logger  *slog.Logger
	// Now is the clock, for tests.
	Now func() time.Time
}

// Forwarded acts in each customer's own Stream app with the server token the gateway in
// front minted for it, rather than with secrets the customer registered. The gateway
// already holds every app's secret to check its callers, and forwards a token with each
// request it vouches for; the newest is kept sealed in Postgres, so every replica and a
// restart act with it.
type Forwarded struct {
	store   *store.Store
	sealer  *auth.Sealer
	mint    *Mint
	baseURL string
	logger  *slog.Logger
	now     func() time.Time

	mu   sync.Mutex
	held map[int64]heldToken
}

// heldToken is a token opened from the store, or kept from a request.
type heldToken struct {
	customer string
	apiKey   string
	token    Secret
	expires  time.Time
}

// NewForwarded returns the source of forwarded tokens.
func NewForwarded(options ForwardedOptions) (*Forwarded, error) {
	if options.Store == nil || options.Sealer == nil {
		return nil, errors.New("streamapp: forwarded tokens need the store and the keyring to keep them")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &Forwarded{
		store: options.Store, sealer: options.Sealer, mint: options.Mint, baseURL: options.BaseURL,
		logger: cmp.Or(options.Logger, slog.Default()), now: now,
		held: map[int64]heldToken{},
	}, nil
}

// Mint is the gateway's mint endpoint, nil when there is none.
func (f *Forwarded) Mint() *Mint { return f.mint }

// Keep holds a token the gateway forwarded for a customer's own app, unless the one held
// already outlasts it, and reports whether it was kept. A customer that is not an app's
// own has no token to keep.
func (f *Forwarded) Keep(ctx context.Context, customer, apiKey, token string) (bool, error) {
	app, ok := ParseAppID(customer)
	if !ok || apiKey == "" || token == "" {
		return false, nil
	}
	f.mu.Lock()
	held, ok := f.held[app]
	f.mu.Unlock()
	if ok && held.token.Reveal() == token {
		return false, nil
	}
	expires, err := expiryOf(token)
	if err != nil {
		return false, err
	}
	if ok && !held.expires.Before(expires) {
		return false, nil
	}
	return f.keep(ctx, app, heldToken{customer: customer, apiKey: apiKey, token: NewSecret(token), expires: expires})
}

// keep writes a token down unless the store holds one that outlasts it, and reports
// whether it was written.
func (f *Forwarded) keep(ctx context.Context, app int64, token heldToken) (bool, error) {
	sealed, err := f.sealer.SealWithAAD(token.token.Reveal(), boundTo(tokenPurpose, token.customer, app, token.apiKey))
	if err != nil {
		return false, err
	}
	kept, err := f.store.KeepStreamAppToken(ctx, store.StreamAppToken{
		StreamAppPK: app, CustomerID: token.customer, APIKey: token.apiKey,
		Sealed: sealed, KEKVersion: f.sealer.CurrentVersion(), ExpiresAt: token.expires,
	})
	if err != nil || !kept {
		return false, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	if held, ok := f.held[app]; !ok || held.expires.Before(token.expires) {
		f.held[app] = token
	}
	return true, nil
}

// identity is the customer's own app acted in with the token held for it, refreshed
// through the gateway when it is about to run out. ErrNoIdentity is an app with no token
// held and none to be had.
func (f *Forwarded) identity(ctx context.Context, customer string, app int64) (Identity, error) {
	if customer != CustomerOf(app) {
		return Identity{}, ErrNoIdentity
	}
	token, err := f.read(ctx, app)
	if err != nil {
		return Identity{}, err
	}
	if token.expires.Sub(f.now()) < refreshBefore && f.mint != nil {
		minted, err := f.mint.ServerToken(ctx, app, token.apiKey)
		switch {
		case err == nil:
			token = heldToken{customer: customer, apiKey: token.apiKey, token: minted.Token, expires: minted.ExpiresAt}
			if _, err := f.keep(ctx, app, token); err != nil {
				f.logger.Warn("stream: could not keep a token the gateway minted", "stream_app", app, "error", err)
			}
		case token.expires.After(f.now()):
			f.logger.Warn("stream: could not refresh an app's token, acting with the one held",
				"stream_app", app, "expires_at", token.expires, "error", err)
		default:
			return Identity{}, fmt.Errorf("%w: %w", ErrNoIdentity, err)
		}
	}
	if !token.expires.After(f.now()) {
		return Identity{}, ErrNoIdentity
	}
	return Identity{
		CustomerID: customer, StreamApp: app, APIKey: token.apiKey, Token: token.token, TokenExpires: token.expires,
		BaseURL: f.baseURL, Registered: true,
	}, nil
}

// read is the token held for an app, from memory or else from the store.
func (f *Forwarded) read(ctx context.Context, app int64) (heldToken, error) {
	f.mu.Lock()
	held, ok := f.held[app]
	f.mu.Unlock()
	if ok && held.expires.Sub(f.now()) >= refreshBefore {
		return held, nil
	}
	row, err := f.store.StreamAppToken(ctx, app)
	if errors.Is(err, store.ErrNoStreamAppToken) {
		return heldToken{}, ErrNoIdentity
	}
	if err != nil {
		return heldToken{}, err
	}
	if !row.ExpiresAt.After(f.now()) {
		// Nothing to open: an expired row, or a placeholder seeded for an app that had
		// registered its keys, is only the key the gateway is asked to mint for.
		return heldToken{customer: row.CustomerID, apiKey: row.APIKey, expires: row.ExpiresAt}, nil
	}
	opened, err := f.sealer.OpenWithAADVersion(row.Sealed, boundTo(tokenPurpose, row.CustomerID, app, row.APIKey), row.KEKVersion)
	if err != nil {
		return heldToken{}, fmt.Errorf("streamapp: opening the token held for app %d: %w", app, err)
	}
	read := heldToken{customer: row.CustomerID, apiKey: row.APIKey, token: NewSecret(opened), expires: row.ExpiresAt}
	f.mu.Lock()
	defer f.mu.Unlock()
	if held, ok := f.held[app]; !ok || held.expires.Before(read.expires) {
		f.held[app] = read
	}
	return f.held[app], nil
}

// boundTo is what something sealed for an app's key is bound to: what it is, whose it is,
// which app and which key. A sealed value moved to any other row does not open there. Each
// part is prefixed with its length, so no two different rows can run together into the
// same bytes.
func boundTo(purpose, customer string, app int64, apiKey string) []byte {
	var aad []byte
	for _, part := range []string{purpose, customer, strconv.FormatInt(app, 10), apiKey} {
		aad = binary.BigEndian.AppendUint32(aad, uint32(len(part)))
		aad = append(aad, part...)
	}
	return aad
}

// expiryOf is when a token the gateway minted expires. Its signature is not checked: the
// router holds no secret to check it with, and believes the gateway as it believes the
// app it names. A token that never expires is refused, since it could never be replaced.
func expiryOf(token string) (time.Time, error) {
	claims := jwt.RegisteredClaims{}
	if _, _, err := jwt.NewParser().ParseUnverified(token, &claims); err != nil {
		return time.Time{}, fmt.Errorf("streamapp: reading a forwarded token: %w", err)
	}
	if claims.ExpiresAt == nil {
		return time.Time{}, errors.New("streamapp: a forwarded token names no expiry")
	}
	return claims.ExpiresAt.Time, nil
}
