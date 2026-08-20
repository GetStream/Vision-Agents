//go:build cgo && webrtc

package run

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/caller"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/streamrtc"
)

func runWebRTC(ctx context.Context, cfg Config, sc scenario.Scenario, audioMap map[string][]int16, trial int) (caller.Result, error) {
	callID := webrtcCallID(cfg, sc, trial)
	callType := cfg.CallType
	if callType == "" {
		callType = "default"
	}

	media, err := streamrtc.Join(ctx, streamrtc.Options{
		CallID:   callID,
		CallType: callType,
		UserID:   cfg.UserID,
		Logger:   cfg.Logger,
	})
	if err != nil {
		return caller.Result{}, err
	}
	defer media.Close()

	if cfg.AccelURL != "" {
		stop, err := startAccelSession(ctx, cfg, callID, callType)
		if err != nil {
			return caller.Result{}, err
		}
		defer stop()
	}

	if cfg.AgentURL != "" {
		sessionID, err := startAgentSession(ctx, cfg.AgentURL, callID, callType)
		if err != nil {
			return caller.Result{}, err
		}
		defer func() {
			closeCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			closeAgentSession(closeCtx, cfg.Logger, cfg.AgentURL, callID, sessionID)
		}()
	}

	eng := caller.Engine{Audio: audioMap, Logger: cfg.Logger}
	callCtx, cancel := context.WithTimeout(ctx, time.Duration(max(sc.MaxDurationS, 60))*time.Second)
	defer cancel()
	return eng.Play(callCtx, sc, media)
}
