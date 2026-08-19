//go:build !cgo || !webrtc

package run

import (
	"context"

	"github.com/GetStream/Vision-Agents/benchmark/internal/caller"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/streamrtc"
)

func runWebRTC(ctx context.Context, cfg Config, sc scenario.Scenario, audioMap map[string][]int16, trial int) (caller.Result, error) {
	_ = ctx
	_ = cfg
	_ = sc
	_ = audioMap
	_ = trial
	return caller.Result{}, streamrtc.ErrDisabled
}
