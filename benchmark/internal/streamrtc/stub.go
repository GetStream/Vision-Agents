//go:build !cgo || !webrtc

package streamrtc

import (
	"context"

	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

func join(ctx context.Context, options Options) (transport.Media, error) {
	_ = ctx
	_ = options
	return nil, ErrDisabled
}
