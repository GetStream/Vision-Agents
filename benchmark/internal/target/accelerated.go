package target

import (
	"context"
	"os"
	"strings"
)

const defaultAccelURL = "http://127.0.0.1:8080"

// As-shipped acceleration pipeline, matching examples/agents/customer_support.
const (
	DefaultAcceleratedSTT      = "gemini/gemini-3.5-transcribe-live"
	DefaultAcceleratedTTS      = "inworld/inworld-tts-2-flash"
	DefaultAcceleratedModel    = "gemini/gemini-3.8-flash"
	DefaultAcceleratedSubagent = "openai/gpt-5.6-sol"
)

// Accelerated is the Python SDK plus stream.Accelerated. Function calling stays
// in Python. An optional --bin spawns the router the SDK talks to.
type Accelerated struct {
	Python
	Bin string
}

func (a *Accelerated) Prepare(ctx context.Context) (func(), error) {
	a.Pipeline = "accelerated"
	a.Env = append(a.Env, acceleratedPipelineEnv()...)
	var stops []func()
	combine := func() {
		for i := len(stops) - 1; i >= 0; i-- {
			stops[i]()
		}
	}
	if a.Spawn && (a.Bin != "" || os.Getenv("ACCEL_ROUTER") != "") {
		routerURL := a.routerURL()
		stopRouter, err := StartRouter(ctx, a.Bin, routerURL)
		if err != nil {
			return nil, err
		}
		stops = append(stops, stopRouter)
		a.Env = append(a.Env, "STREAM_ACCELERATION_URL="+routerURL)
		a.logger().Info("spawned accel router for accelerated target", "url", routerURL)
	}
	stopPython, err := a.Python.Prepare(ctx)
	if err != nil {
		combine()
		return nil, err
	}
	stops = append(stops, stopPython)
	return combine, nil
}

func acceleratedPipelineEnv() []string {
	return []string{
		"VOICEBENCH_MODEL=" + envOr("VOICEBENCH_MODEL", DefaultAcceleratedModel),
		"VOICEBENCH_STT=" + envOr("VOICEBENCH_STT", DefaultAcceleratedSTT),
		"VOICEBENCH_TTS=" + envOr("VOICEBENCH_TTS", DefaultAcceleratedTTS),
		"VOICEBENCH_SUBAGENT=" + envOr("VOICEBENCH_SUBAGENT", DefaultAcceleratedSubagent),
		"STREAM_ACCELERATION_CUSTOMER_ID=" + envOr("STREAM_ACCELERATION_CUSTOMER_ID", "voicebench"),
	}
}

func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func (a *Accelerated) routerURL() string {
	if u := os.Getenv("STREAM_ACCELERATION_URL"); u != "" {
		return strings.TrimRight(u, "/")
	}
	return defaultAccelURL
}
