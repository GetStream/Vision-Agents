// Package demoauth contains credential helpers used by local demos. It is not used by
// the production router, which continues to use Google ADC or an explicitly configured
// identity-token file.
package demoauth

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
)

const (
	commandTimeout = 10 * time.Second
	commandWait    = 2 * time.Second
	maxTokenBytes  = 16 * 1024
	refreshLead    = 5 * time.Minute
	refreshMin     = time.Second
	retryMin       = 5 * time.Second
	retryMax       = 30 * time.Second
)

// GCloudTokenSource caches an identity token minted from the logged-in local gcloud
// account. It refreshes ahead of expiry on a background goroutine; Token never starts a
// subprocess. Use this only for the demo's fixed, trusted Cloud Run endpoint.
type GCloudTokenSource struct {
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	mu     sync.RWMutex
	token  *oauth2.Token
	mint   func(context.Context) (*oauth2.Token, error)
	closed bool

	closeOnce sync.Once
	policy    refreshPolicy
}

type refreshPolicy struct {
	lead     time.Duration
	minimum  time.Duration
	retryMin time.Duration
	retryMax time.Duration
}

var defaultRefreshPolicy = refreshPolicy{
	lead:     refreshLead,
	minimum:  refreshMin,
	retryMin: retryMin,
	retryMax: retryMax,
}

// NewGCloudTokenSource uses the existing user login from `gcloud auth login`. It does
// not write credentials to disk or accept an audience override. Cloud Run's developer
// user-token flow uses the command's default audience; production callers should keep
// using ADC instead.
func NewGCloudTokenSource(ctx context.Context) (*GCloudTokenSource, error) {
	if ctx == nil {
		return nil, errors.New("demoauth: context is required")
	}
	path, err := exec.LookPath("gcloud")
	if err != nil {
		return nil, errors.New("demoauth: gcloud is unavailable; install it and run gcloud auth login")
	}
	return newGCloudTokenSource(ctx, func(ctx context.Context) (*oauth2.Token, error) {
		return mintGCloudToken(ctx, path)
	})
}

func newGCloudTokenSource(parent context.Context, mint func(context.Context) (*oauth2.Token, error)) (*GCloudTokenSource, error) {
	return newGCloudTokenSourceWithPolicy(parent, mint, defaultRefreshPolicy)
}

func newGCloudTokenSourceWithPolicy(parent context.Context, mint func(context.Context) (*oauth2.Token, error), policy refreshPolicy) (*GCloudTokenSource, error) {
	if parent == nil || mint == nil {
		return nil, errors.New("demoauth: token source configuration is invalid")
	}
	if policy.lead <= 0 || policy.minimum <= 0 || policy.retryMin <= 0 || policy.retryMax < policy.retryMin {
		return nil, errors.New("demoauth: token refresh policy is invalid")
	}
	ctx, cancel := context.WithCancel(parent)
	token, err := mint(ctx)
	if err != nil {
		cancel()
		return nil, err
	}
	if !usableToken(token, time.Now()) {
		cancel()
		return nil, errors.New("demoauth: gcloud returned an invalid identity token")
	}
	source := &GCloudTokenSource{
		ctx:    ctx,
		cancel: cancel,
		done:   make(chan struct{}),
		token:  cloneToken(token),
		mint:   mint,
		policy: policy,
	}
	go source.refreshLoop()
	return source, nil
}

// Token returns the current cached credential. It never invokes gcloud. Once the cache
// expires or the source is closed, callers receive an error and can use their existing
// semantic fallback.
func (source *GCloudTokenSource) Token() (*oauth2.Token, error) {
	if source == nil {
		return nil, errors.New("demoauth: identity token is unavailable")
	}
	source.mu.RLock()
	defer source.mu.RUnlock()
	if source.closed || source.ctx.Err() != nil || !usableToken(source.token, time.Now()) {
		return nil, errors.New("demoauth: cached identity token is unavailable")
	}
	return cloneToken(source.token), nil
}

// Close stops and joins the background refresh loop. It is safe to call more than once.
func (source *GCloudTokenSource) Close() error {
	if source == nil {
		return nil
	}
	source.closeOnce.Do(func() {
		source.mu.Lock()
		source.closed = true
		source.mu.Unlock()
		source.cancel()
	})
	<-source.done
	return nil
}

func (source *GCloudTokenSource) refreshLoop() {
	defer close(source.done)
	for {
		current := source.currentToken()
		if current == nil {
			return
		}
		if !waitFor(source.ctx, refreshDelay(current.Expiry, time.Now(), source.policy)) {
			return
		}
		for {
			fresh, err := source.mint(source.ctx)
			if err == nil && usableToken(fresh, time.Now()) {
				source.mu.Lock()
				if source.ctx.Err() == nil && !source.closed {
					source.token = cloneToken(fresh)
				}
				source.mu.Unlock()
				break
			}
			if !waitFor(source.ctx, retryDelay(current.Expiry, time.Now(), source.policy)) {
				return
			}
		}
	}
}

func (source *GCloudTokenSource) currentToken() *oauth2.Token {
	source.mu.RLock()
	defer source.mu.RUnlock()
	return cloneToken(source.token)
}

func refreshDelay(expiry, now time.Time, policy refreshPolicy) time.Duration {
	remaining := expiry.Sub(now)
	if remaining <= 0 {
		return policy.minimum
	}
	lead := policy.lead
	if tenth := remaining / 10; tenth < lead {
		lead = tenth
	}
	delay := remaining - lead
	if delay < policy.minimum {
		return policy.minimum
	}
	return delay
}

func retryDelay(expiry, now time.Time, policy refreshPolicy) time.Duration {
	remaining := expiry.Sub(now)
	delay := policy.retryMax
	if remaining > 0 && remaining/3 < delay {
		delay = remaining / 3
	}
	if delay < policy.retryMin {
		return policy.retryMin
	}
	return delay
}

func waitFor(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return ctx.Err() == nil
	case <-ctx.Done():
		return false
	}
}

func mintGCloudToken(parent context.Context, path string) (*oauth2.Token, error) {
	ctx, cancel := context.WithTimeout(parent, commandTimeout)
	defer cancel()
	command := exec.CommandContext(ctx, path, "auth", "print-identity-token", "--quiet")
	command.WaitDelay = commandWait
	stdout := &limitedBuffer{limit: maxTokenBytes}
	command.Stdout = stdout
	command.Stderr = io.Discard
	if err := command.Run(); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, errors.New("demoauth: gcloud identity-token command timed out")
		}
		return nil, errors.New("demoauth: gcloud identity-token command failed")
	}
	if stdout.overflow {
		return nil, errors.New("demoauth: gcloud identity token exceeded the size limit")
	}
	return parseIdentityToken(string(stdout.data))
}

type limitedBuffer struct {
	data     []byte
	limit    int
	overflow bool
}

func (buffer *limitedBuffer) Write(data []byte) (int, error) {
	n := len(data)
	remaining := buffer.limit - len(buffer.data)
	if remaining <= 0 {
		buffer.overflow = true
		return n, nil
	}
	if len(data) > remaining {
		buffer.data = append(buffer.data, data[:remaining]...)
		buffer.overflow = true
		return n, nil
	}
	buffer.data = append(buffer.data, data...)
	return n, nil
}

func parseIdentityToken(raw string) (*oauth2.Token, error) {
	value := strings.TrimSpace(raw)
	if value == "" || len(value) > maxTokenBytes || strings.ContainsAny(value, "\r\n \t") {
		return nil, errors.New("demoauth: gcloud returned an invalid identity token")
	}
	parts := strings.Split(value, ".")
	if len(parts) != 3 || parts[1] == "" {
		return nil, errors.New("demoauth: gcloud returned an invalid identity token")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil || len(payload) > maxTokenBytes {
		return nil, errors.New("demoauth: gcloud returned an invalid identity token")
	}
	var claims struct {
		Expiry json.Number `json:"exp"`
	}
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := decoder.Decode(&claims); err != nil {
		return nil, errors.New("demoauth: gcloud returned an invalid identity token")
	}
	expirySeconds, err := claims.Expiry.Int64()
	if err != nil {
		return nil, errors.New("demoauth: gcloud returned an invalid identity token")
	}
	expiry := time.Unix(expirySeconds, 0)
	if !expiry.After(time.Now().Add(10*time.Second)) || expiry.After(time.Now().Add(24*time.Hour)) {
		return nil, errors.New("demoauth: gcloud returned an expired or implausible identity token")
	}
	return &oauth2.Token{AccessToken: value, TokenType: "Bearer", Expiry: expiry}, nil
}

func usableToken(token *oauth2.Token, now time.Time) bool {
	return token != nil && token.AccessToken != "" && token.Expiry.After(now)
}

func cloneToken(token *oauth2.Token) *oauth2.Token {
	if token == nil {
		return nil
	}
	copy := *token
	return &copy
}
