// Package streamapp says which Stream app the router acts in for a customer, and with
// which credential.
//
// Everything the router does in Stream, writing a conversation, joining a call, creating a
// SIP trunk, minting a token or checking a hook, is done in one app with one key. Which app
// that is used to be the deployment's own, whoever was calling. A Source answers it per
// customer instead, from the customer id alone, because much of that work happens with no
// request behind it: an outbox delivering after the caller left, a conversation recovered
// at startup, a campaign ringing on a schedule, a hook Stream sends on its own.
//
// Work already written records the app it was written in, its pin, so it is read and
// finished there even after the customer's app has changed.
package streamapp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
)

// Identity is one customer's Stream app and the credential the router acts in it with.
type Identity struct {
	// CustomerID is the router tenant the identity was resolved for.
	CustomerID string
	// StreamApp is the pin a record made with this identity carries: the Stream app's id,
	// or zero for the deployment's own app while the router runs in deployment mode,
	// which is how every record written before apps had identities reads.
	StreamApp int64
	// APIKey is the app's public key.
	APIKey string
	// Secret signs server-side requests, tokens and hook checks. It never prints.
	Secret Secret
	// UserToken is the deployment's raw STREAM_USER_TOKEN, which a voice edge prefers to
	// minting its own. Only the deployment identity carries one.
	UserToken string
	// BaseURL is the Stream API the app is reached at. Empty is Stream's default.
	BaseURL string
}

// Fingerprint names the credential without revealing it, for caches and logs.
func (i Identity) Fingerprint() string {
	sum := sha256.Sum256([]byte(i.APIKey + "\x00" + i.Secret.Reveal() + "\x00" + i.BaseURL))
	return hex.EncodeToString(sum[:8])
}

// Source answers which identity the router acts with for a customer.
type Source interface {
	// For is the identity new work for a customer is done with.
	For(ctx context.Context, customer string) (Identity, error)
	// ForApp is the identity work already pinned to an app is finished with: the app it
	// was written in, never wherever the customer is now.
	ForApp(ctx context.Context, customer string, app int64) (Identity, error)
}

var (
	// ErrNoIdentity is a customer the router has no Stream app to act in for.
	ErrNoIdentity = errors.New("streamapp: no Stream app is configured for this customer")
	// ErrStreamAppMoved is work pinned to an app the customer no longer acts in. It is
	// parked rather than delivered anywhere else.
	ErrStreamAppMoved = errors.New("streamapp: this was written in another Stream app")
	// ErrStreamAppDisconnected is a customer whose app was disconnected or blocked.
	ErrStreamAppDisconnected = errors.New("streamapp: this customer's Stream app is disconnected")
	// ErrReadOnly is work kept in the deployment's shared app for a customer that may no
	// longer write there. It can be read, not added to.
	ErrReadOnly = errors.New("streamapp: this is kept in the shared app and can only be read")
	// ErrDeploymentAppUnknown is work pinned to an app while the router does not yet know
	// which app its own is. It waits rather than guesses.
	ErrDeploymentAppUnknown = errors.New("streamapp: the deployment's own Stream app is not known yet")
)
