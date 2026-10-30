//go:build !webrtc

package streamrtc

import (
	"context"

	"github.com/GetStream/Vision-Agents/performance_benchmark/internal/rtc/transport"
)

func join(ctx context.Context, options Options) (transport.Media, error) {
	_ = ctx
	_ = options
	return nil, ErrDisabled
}
