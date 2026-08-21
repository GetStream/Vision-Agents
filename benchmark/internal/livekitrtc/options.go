package livekitrtc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"

	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

const (
	defaultIdentity = "voicebench-caller"
	urlEnvVar       = "LIVEKIT_URL"
	apiKeyEnvVar    = "LIVEKIT_API_KEY"
	apiSecretEnvVar = "LIVEKIT_API_SECRET"
)

// Options join a LiveKit room as the scripted caller.
type Options struct {
	URL       string
	APIKey    string
	APISecret string
	Room      string
	Identity  string
	Logger    *slog.Logger
}

// Resolve fills the connection details from the LIVEKIT_* environment and validates them.
func (o *Options) Resolve() error {
	if o.URL == "" {
		o.URL = os.Getenv(urlEnvVar)
	}
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APISecret == "" {
		o.APISecret = os.Getenv(apiSecretEnvVar)
	}
	for _, missing := range []struct {
		value string
		name  string
	}{
		{o.URL, urlEnvVar},
		{o.APIKey, apiKeyEnvVar},
		{o.APISecret, apiSecretEnvVar},
	} {
		if missing.value == "" {
			return fmt.Errorf("livekitrtc: %s is not set", missing.name)
		}
	}
	return nil
}

// ErrDisabled is returned when this binary was built without `-tags webrtc` (and cgo/libopus).
var ErrDisabled = errors.New("livekitrtc: rebuild with -tags webrtc, CGO_ENABLED=1, and libopus")

// Join connects to a LiveKit room as the benchmark caller.
func Join(ctx context.Context, options Options) (transport.Media, error) {
	return join(ctx, options)
}
