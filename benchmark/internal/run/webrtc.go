package run

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/caller"
	"github.com/GetStream/Vision-Agents/benchmark/internal/livekitrtc"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/streamrtc"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

const agentReadyTimeout = 45 * time.Second

type targetFailure struct {
	err error
}

func (e targetFailure) Error() string { return e.err.Error() }

func (e targetFailure) Unwrap() error { return e.err }

func runWebRTC(ctx context.Context, cfg Config, sc scenario.Scenario, audioMap map[string][]int16, trial int) (caller.Result, error) {
	callID := webrtcCallID(cfg, sc, trial)
	callType := cfg.CallType
	if callType == "" {
		callType = "default"
	}

	media, err := dial(ctx, cfg, callID, callType)
	if err != nil {
		return caller.Result{}, err
	}
	defer media.Close()

	if cfg.Target != nil {
		stop, err := cfg.Target.StartCall(ctx, callID, callType)
		if err != nil {
			return caller.Result{}, targetFailure{err: err}
		}
		defer stop()
	}
	readyCtx, cancelReady := context.WithTimeout(ctx, agentReadyTimeout)
	defer cancelReady()
	if err := media.WaitForAgent(readyCtx); err != nil {
		return caller.Result{}, targetFailure{err: err}
	}

	eng := caller.Engine{Audio: audioMap, Logger: cfg.Logger}
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(max(sc.MaxDurationS, 60))*time.Second)
	defer cancel()
	recording, err := eng.Play(callCtx, sc, media)
	if errors.Is(err, context.DeadlineExceeded) {
		return recording, targetFailure{err: err}
	}
	return recording, err
}

func dial(ctx context.Context, cfg Config, callID string, callType string) (transport.Media, error) {
	switch cfg.Transport {
	case "", transportStream:
		return streamrtc.Join(ctx, streamrtc.Options{
			CallID:   callID,
			CallType: callType,
			UserID:   cfg.UserID,
			Logger:   cfg.Logger,
		})
	case transportLiveKit:
		return livekitrtc.Join(ctx, livekitrtc.Options{
			Room:     callID,
			Identity: cfg.UserID,
			URL:      cfg.TargetURL,
			Logger:   cfg.Logger,
		})
	default:
		return nil, fmt.Errorf("run: unknown transport %s", cfg.Transport)
	}
}
