//go:build !cgo || !webrtc

package livekitrtc

import (
	"context"

	"github.com/GetStream/Vision-Agents/benchmark/rtc/transport"
)

func join(context.Context, Options) (transport.Media, error) { return nil, ErrDisabled }
