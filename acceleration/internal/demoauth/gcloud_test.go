package demoauth

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/oauth2"
)

func TestGCloudTokenSourceCachesAndRefreshesInBackground(t *testing.T) {
	var calls atomic.Int32
	refreshStarted := make(chan struct{})
	releaseRefresh := make(chan struct{})
	policy := refreshPolicy{lead: 200 * time.Millisecond, minimum: 5 * time.Millisecond, retryMin: 10 * time.Millisecond, retryMax: 40 * time.Millisecond}
	firstToken := testIdentityToken(time.Now().Add(800 * time.Millisecond))
	secondToken := testIdentityToken(time.Now().Add(time.Hour))
	source, err := newGCloudTokenSourceWithPolicy(context.Background(), func(ctx context.Context) (*oauth2.Token, error) {
		switch calls.Add(1) {
		case 1:
			return firstToken, nil
		case 2:
			close(refreshStarted)
			select {
			case <-releaseRefresh:
				return secondToken, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		default:
			return nil, fmt.Errorf("unexpected refresh")
		}
	}, policy)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = source.Close() })

	for range 32 {
		token, err := source.Token()
		if err != nil || token.AccessToken != firstToken.AccessToken {
			t.Fatalf("cached token = %#v, %v", token, err)
		}
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("Token invoked the command %d times before refresh", got)
	}
	select {
	case <-refreshStarted:
	case <-time.After(time.Second):
		t.Fatal("background token refresh did not start")
	}
	start := time.Now()
	token, err := source.Token()
	if err != nil || token.AccessToken != firstToken.AccessToken {
		t.Fatalf("Token while refresh is pending = %#v, %v", token, err)
	}
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("Token waited for background refresh: %s", elapsed)
	}
	close(releaseRefresh)
	deadline := time.Now().Add(time.Second)
	refreshed := false
	for time.Now().Before(deadline) {
		token, err = source.Token()
		if err == nil && token.AccessToken == secondToken.AccessToken {
			refreshed = true
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !refreshed {
		t.Fatalf("background refresh did not publish new token: token=%#v err=%v", token, err)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("gcloud refresh calls = %d, want 2", got)
	}
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := source.Token(); err == nil {
		t.Fatal("closed token source returned a credential")
	}
}

func TestGCloudTokenSourceKeepsUsableTokenAndRetriesBoundedly(t *testing.T) {
	var calls atomic.Int32
	policy := refreshPolicy{lead: 200 * time.Millisecond, minimum: 5 * time.Millisecond, retryMin: 10 * time.Millisecond, retryMax: 40 * time.Millisecond}
	refreshed := make(chan struct{})
	firstToken := testIdentityToken(time.Now().Add(800 * time.Millisecond))
	secondToken := testIdentityToken(time.Now().Add(time.Hour))
	source, err := newGCloudTokenSourceWithPolicy(context.Background(), func(context.Context) (*oauth2.Token, error) {
		switch calls.Add(1) {
		case 1:
			return firstToken, nil
		case 2:
			return nil, fmt.Errorf("private command failure")
		case 3:
			close(refreshed)
			return secondToken, nil
		default:
			return nil, fmt.Errorf("unexpected refresh")
		}
	}, policy)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = source.Close() })
	select {
	case <-refreshed:
	case <-time.After(time.Second):
		t.Fatal("bounded refresh retry did not recover")
	}
	deadline := time.Now().Add(time.Second)
	refreshedToken := false
	var token *oauth2.Token
	for time.Now().Before(deadline) {
		token, err = source.Token()
		if err == nil && token.AccessToken == secondToken.AccessToken {
			refreshedToken = true
			break
		}
		time.Sleep(time.Millisecond)
	}
	if !refreshedToken {
		t.Fatalf("retry did not publish refreshed token: token=%#v err=%v", token, err)
	}
	if got := calls.Load(); got != 3 {
		t.Fatalf("gcloud refresh calls = %d, want 3", got)
	}
}

func TestGCloudTokenSourceCloseCancelsAndJoinsRefresh(t *testing.T) {
	var calls atomic.Int32
	refreshStarted := make(chan struct{})
	policy := refreshPolicy{lead: 200 * time.Millisecond, minimum: 5 * time.Millisecond, retryMin: 10 * time.Millisecond, retryMax: 40 * time.Millisecond}
	firstToken := testIdentityToken(time.Now().Add(800 * time.Millisecond))
	source, err := newGCloudTokenSourceWithPolicy(context.Background(), func(ctx context.Context) (*oauth2.Token, error) {
		if calls.Add(1) == 1 {
			return firstToken, nil
		}
		close(refreshStarted)
		<-ctx.Done()
		return nil, ctx.Err()
	}, policy)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-refreshStarted:
	case <-time.After(time.Second):
		t.Fatal("background refresh did not start")
	}
	start := time.Now()
	if err := source.Close(); err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("Close did not join the canceled refresh promptly")
	}
	if err := source.Close(); err != nil {
		t.Fatalf("repeated Close: %v", err)
	}
}

func TestParseIdentityTokenRequiresValidExpiryAndBoundsOutput(t *testing.T) {
	valid := identityToken(time.Now().Add(time.Hour))
	if token, err := parseIdentityToken(valid); err != nil || token.AccessToken != valid || token.TokenType != "Bearer" {
		t.Fatalf("parse valid token = %#v, %v", token, err)
	}
	for _, invalid := range []string{
		"",
		"not-a-jwt",
		identityToken(time.Now().Add(-time.Minute)),
		identityToken(time.Now().Add(48 * time.Hour)),
		"header.%%% .signature",
		strings.Repeat("x", maxTokenBytes+1),
	} {
		if _, err := parseIdentityToken(invalid); err == nil {
			t.Fatalf("parseIdentityToken(%q) succeeded", invalid)
		}
	}
	buffer := &limitedBuffer{limit: 3}
	if n, err := buffer.Write([]byte("abcdef")); err != nil || n != 6 || !buffer.overflow || string(buffer.data) != "abc" {
		t.Fatalf("limited writer n=%d err=%v overflow=%v data=%q", n, err, buffer.overflow, buffer.data)
	}
}

func TestMintGCloudTokenBoundsCommandOutputAndContext(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fake gcloud scripts use the platform shell")
	}
	dir := t.TempDir()
	oversized := filepath.Join(dir, "oversized-gcloud")
	requireNoError(t, os.WriteFile(oversized, []byte("#!/bin/sh\nprintf '%020000d' 0\n"), 0o700))
	if _, err := mintGCloudToken(context.Background(), oversized); err == nil || !strings.Contains(err.Error(), "size limit") {
		t.Fatalf("oversized command output error = %v", err)
	}

	slow := filepath.Join(dir, "slow-gcloud")
	requireNoError(t, os.WriteFile(slow, []byte("#!/bin/sh\nexec sleep 5\n"), 0o700))
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := mintGCloudToken(ctx, slow); err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Fatalf("canceled command error = %v", err)
	}
}

func requireNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func testIdentityToken(expiry time.Time) *oauth2.Token {
	return &oauth2.Token{AccessToken: identityToken(expiry), TokenType: "Bearer", Expiry: expiry}
}

func identityToken(expiry time.Time) string {
	payload := []byte(fmt.Sprintf(`{"exp":%d}`, expiry.Unix()))
	return "header." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
}
