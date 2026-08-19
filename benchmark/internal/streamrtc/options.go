package streamrtc

import (
	"context"
	"errors"
	"log/slog"

	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

const (
	apiKeyEnvVar    = "STREAM_API_KEY"
	apiSecretEnvVar = "STREAM_API_SECRET"
	userTokenEnvVar = "STREAM_USER_TOKEN"
	defaultCallType = "default"
	defaultUserID   = "voicebench-caller"
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

// ErrDisabled is returned when this binary was built without `-tags webrtc` (and cgo/libopus).
var ErrDisabled = errors.New("streamrtc: rebuild with -tags webrtc, CGO_ENABLED=1, and libopus")

// Join connects to a Stream call as the benchmark caller.
func Join(ctx context.Context, options Options) (transport.Media, error) {
	return join(ctx, options)
}
