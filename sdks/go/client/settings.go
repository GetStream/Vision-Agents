package client

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// AppSettings is what the router does for this app.
type AppSettings = acceleration.AppSettings

// Settings is what the router does for this app, as far as the app can see it.
type Settings struct {
	client *Client
}

// Settings is what the router does for this app.
func (c *Client) Settings() *Settings { return &Settings{client: c} }

// App reads this app's settings: which Stream app its conversations, transcripts, calls
// and phone lines are written into, and whether that app holds the agent channel and call
// types they need. Only a backend may ask.
func (s *Settings) App(ctx context.Context) (*AppSettings, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}

	read, err := api.GetAppSettingsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("client: reading the app's settings: %w", err)
	}
	if read.JSON200 == nil {
		return nil, failure("reading the app's settings", read.HTTPResponse, read.Body)
	}
	return read.JSON200, nil
}

// StreamKey is one key of the app's own Stream app and its secret.
type StreamKey struct {
	APIKey    string
	APISecret string
	// CreatedAt is when Stream made the key, zero when not known.
	CreatedAt time.Time
}

// StreamCredentials is every key the router acts in the app's own Stream app with.
type StreamCredentials struct {
	// Keys replace those the router held. Each is checked with Stream.
	Keys []StreamKey
	// PrimaryKey mints tokens. Empty is the first key.
	PrimaryKey string
	// Revision is the revision last read from App, 0 for an app never registered.
	Revision int64
	// AllowGuests lets guests be made in the app. Nil keeps what was set.
	AllowGuests *bool
}

// RegisterStream registers the app's own Stream app with the router, which acts in it from
// then on with the keys given: every key must belong to this app. Keys left out are
// dropped. Only a backend may ask, and only of a router running in app mode.
func (s *Settings) RegisterStream(ctx context.Context, credentials StreamCredentials) (*AppSettings, error) {
	if len(credentials.Keys) == 0 {
		return nil, errors.New("client: registering a Stream app needs a key; DisconnectStream takes the app back")
	}
	body := acceleration.StreamCredentials{
		ExpectedRevision: credentials.Revision, AllowGuests: credentials.AllowGuests,
		PrimaryKey: pointer(credentials.PrimaryKey),
	}
	keys := streamKeys(credentials.Keys)
	body.Keys = &keys
	return s.putStream(ctx, "registering the app's Stream app", body)
}

// DisconnectStream takes the app's own Stream app back: the router deletes every key and
// writes nothing for the app anywhere until it registers again. proof is one of the app's
// keys and its secret, which Stream checks and the router keeps nowhere.
func (s *Settings) DisconnectStream(ctx context.Context, revision int64, proof StreamKey) (*AppSettings, error) {
	if proof.APIKey == "" || proof.APISecret == "" {
		return nil, errors.New("client: disconnecting a Stream app needs one of its keys and its secret")
	}
	proved := streamKeys([]StreamKey{proof})[0]
	none := []acceleration.StreamKeyInput{}
	body := acceleration.StreamCredentials{ExpectedRevision: revision, Keys: &none, Proof: &proved}
	return s.putStream(ctx, "disconnecting the app's Stream app", body)
}

// CheckStream asks Stream again about the app's own Stream app, and says which numbers
// still have their lines in another app.
func (s *Settings) CheckStream(ctx context.Context) (*acceleration.StreamCheck, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}
	checked, err := api.CheckAppStreamCredentialsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("client: checking the app's Stream app: %w", err)
	}
	if checked.JSON200 == nil {
		return nil, failure("checking the app's Stream app", checked.HTTPResponse, checked.Body)
	}
	return checked.JSON200, nil
}

func (s *Settings) putStream(ctx context.Context, what string, body acceleration.StreamCredentials) (*AppSettings, error) {
	api, err := s.client.api()
	if err != nil {
		return nil, err
	}
	written, err := api.UpdateAppStreamCredentialsWithResponse(ctx, body)
	if err != nil {
		return nil, fmt.Errorf("client: %s: %w", what, err)
	}
	if written.JSON200 == nil {
		return nil, failure(what, written.HTTPResponse, written.Body)
	}
	return written.JSON200, nil
}

func streamKeys(keys []StreamKey) []acceleration.StreamKeyInput {
	sent := make([]acceleration.StreamKeyInput, 0, len(keys))
	for _, key := range keys {
		secret := key.APISecret
		sent = append(sent, acceleration.StreamKeyInput{ApiKey: key.APIKey, ApiSecret: &secret, CreatedAt: pointer(key.CreatedAt)})
	}
	return sent
}
