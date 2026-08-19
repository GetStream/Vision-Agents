//go:build !cgo || !webrtc

package streamrtc

import (
	"context"

	"github.com/GetStream/Vision-Agents/benchmark/internal/telephony"
)

func join(ctx context.Context, options Options) (telephony.Media, error) {
	_ = ctx
	_ = options
	return nil, ErrDisabled
}
