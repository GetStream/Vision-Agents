// Package edge is the Stream call an agent is asked to join.
//
// The acceleration backend joins a call that already exists, so what is needed here is the
// creating of one and a link a person can open to be on the other end of it. No media
// crosses this package: the conversation happens in the backend.
package edge

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

const (
	// APIKeyEnv and APISecretEnv are the Stream credentials calls are created with.
	APIKeyEnv    = "STREAM_API_KEY"
	APISecretEnv = "STREAM_API_SECRET"
	// MonitorBaseURLEnv points the monitoring link at another deployment of the demo.
	MonitorBaseURLEnv = "EXAMPLE_BASE_URL"
	// DefaultMonitorBaseURL is Stream's hosted video demo, the same page the Python
	// examples open when they join a call.
	DefaultMonitorBaseURL = "https://getstream.io/video/demos"
	// DefaultCallType is the Stream call type used when none is named.
	DefaultCallType = "agent"
)

// monitorTokenValidity is how long the browser's token lasts. A call outliving it is a call
// nobody is still on.
const monitorTokenValidity = time.Hour

// User is somebody a call is created by or watched as.
type User struct {
	ID   string
	Name string
}

// Call is one Stream call, named the way the backend needs it named.
type Call struct {
	ID   string
	Type string
}

// Options are the credentials and the demo deployment. Empty fields read the environment.
type Options struct {
	APIKey         string
	APISecret      string
	MonitorBaseURL string
}

// Edge creates calls and mints the tokens for watching them.
type Edge struct {
	apiKey     string
	apiSecret  string
	monitorURL string
	client     *getstream.Stream
}

// New reads whatever the options leave empty out of the environment.
func New(options Options) (*Edge, error) {
	apiKey := options.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(APIKeyEnv)
	}
	apiSecret := options.APISecret
	if apiSecret == "" {
		apiSecret = os.Getenv(APISecretEnv)
	}
	if apiKey == "" || apiSecret == "" {
		return nil, fmt.Errorf("edge: %s and %s are required to create a call", APIKeyEnv, APISecretEnv)
	}

	monitor := options.MonitorBaseURL
	if monitor == "" {
		monitor = os.Getenv(MonitorBaseURLEnv)
	}
	if monitor == "" {
		monitor = DefaultMonitorBaseURL
	}

	client, err := getstream.NewClient(apiKey, apiSecret)
	if err != nil {
		return nil, fmt.Errorf("edge: %w", err)
	}
	return &Edge{apiKey: apiKey, apiSecret: apiSecret, monitorURL: monitor, client: client}, nil
}

// CreateCall creates the call the backend will join, or returns the one already under that
// id.
//
// An empty id names a new call after a random one, which is what a one-off conversation
// wants. An empty type is the default one.
func (e *Edge) CreateCall(ctx context.Context, call Call, createdBy User) (Call, error) {
	if call.Type == "" {
		call.Type = DefaultCallType
	}
	if call.ID == "" {
		id, err := randomID()
		if err != nil {
			return call, err
		}
		call.ID = id
	}
	if createdBy.ID == "" {
		return call, errors.New("edge: a call needs somebody to have created it")
	}

	request := &getstream.GetOrCreateCallRequest{
		Data: &getstream.CallRequest{CreatedByID: &createdBy.ID},
	}
	if _, err := e.client.Video().Call(call.Type, call.ID).GetOrCreate(ctx, request); err != nil {
		return call, fmt.Errorf("edge: creating call %s:%s: %w", call.Type, call.ID, err)
	}
	return call, nil
}

// MonitorURL is a link a person can open to join a call from a browser and hear the agent.
//
// The token is minted here rather than fetched, so this makes no network calls: the
// coordinator registers the browser as a user when it connects, the same way it does for
// the agent.
func (e *Edge) MonitorURL(call Call, user User) (string, error) {
	if call.ID == "" {
		return "", errors.New("edge: there is no call to watch")
	}
	if user.ID == "" {
		return "", errors.New("edge: a monitoring user id is required")
	}
	if user.Name == "" {
		user.Name = user.ID
	}

	token, err := e.client.CreateToken(user.ID, getstream.WithExpiration(monitorTokenValidity))
	if err != nil {
		return "", fmt.Errorf("edge: minting a monitoring token: %w", err)
	}

	query := url.Values{
		"api_key":    {e.apiKey},
		"token":      {token},
		"skip_lobby": {"true"},
		"user_name":  {user.Name},
	}
	return fmt.Sprintf("%s/join/%s?%s",
		strings.TrimSuffix(e.monitorURL, "/"), url.PathEscape(call.ID), query.Encode()), nil
}

func randomID() (string, error) {
	raw := make([]byte, 8)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("edge: naming a call: %w", err)
	}
	return hex.EncodeToString(raw), nil
}
