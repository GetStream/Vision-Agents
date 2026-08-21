package streamrtc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

const (
	apiKeyEnvVar    = "STREAM_API_KEY"
	apiSecretEnvVar = "STREAM_API_SECRET"
	userTokenEnvVar = "STREAM_USER_TOKEN"
	defaultCallType = "default"
	defaultUserID   = "voicebench-caller"

	// opusNegotiatedChannels is the channel count Stream negotiates for the published track.
	opusNegotiatedChannels = 2
)

// Options join a Stream call as the scripted caller.
type Options struct {
	CallID    string
	CallType  string
	UserID    string
	UserName  string
	APIKey    string
	APISecret string
	UserToken string
	Logger    *slog.Logger
}

// Resolve fills the credentials from the STREAM_* environment and validates them.
func (o *Options) Resolve() error {
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APISecret == "" {
		o.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if o.UserToken == "" {
		o.UserToken = os.Getenv(userTokenEnvVar)
	}
	if o.APIKey == "" {
		return fmt.Errorf("streamrtc: %s is not set", apiKeyEnvVar)
	}
	if o.APISecret == "" && o.UserToken == "" {
		return fmt.Errorf("streamrtc: set %s or %s", userTokenEnvVar, apiSecretEnvVar)
	}
	return nil
}

// ErrDisabled is returned when this binary was built without `-tags webrtc` (and cgo/libopus).
var ErrDisabled = errors.New("streamrtc: rebuild with -tags webrtc, CGO_ENABLED=1, and libopus")

// Join connects to a Stream call as the benchmark caller.
func Join(ctx context.Context, options Options) (transport.Media, error) {
	return join(ctx, options)
}
