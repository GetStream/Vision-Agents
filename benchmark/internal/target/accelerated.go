package target

import (
	"context"
	"os"
	"strings"
)

const defaultAccelURL = "http://127.0.0.1:8080"

// The acceleration pipeline the bench runs by default. The subagent is named as
// thinking_llm in each pack's agents/accelerated/{pack}/agent.yaml; this records it.
const (
	DefaultAcceleratedSTT      = "deepgram/flux-general-en"
	DefaultAcceleratedTTS      = "elevenlabs/eleven_v4_turbo"
	DefaultAcceleratedModel    = "gemma/gemma-4-26B-A4B-it"
	DefaultAcceleratedSubagent = "openai/gpt-6.1-sol"
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
	env := []string{
		"VOICEBENCH_MODEL=" + envOr("VOICEBENCH_MODEL", DefaultAcceleratedModel),
		"VOICEBENCH_STT=" + envOr("VOICEBENCH_STT", DefaultAcceleratedSTT),
		"VOICEBENCH_TTS=" + envOr("VOICEBENCH_TTS", DefaultAcceleratedTTS),
		"VOICEBENCH_SUBAGENT=" + envOr("VOICEBENCH_SUBAGENT", DefaultAcceleratedSubagent),
	}
	// A hosted router sits behind Stream's proxy and is reached with STREAM_API_KEY and
	// STREAM_API_SECRET. A customer id would make the SDK drop that credential, so it is set
	// empty rather than left out: the agent loads .env, which may name one.
	if authenticatesToRouter() {
		return append(env, "STREAM_ACCELERATION_CUSTOMER_ID=")
	}
	return append(env, "STREAM_ACCELERATION_CUSTOMER_ID="+envOr("STREAM_ACCELERATION_CUSTOMER_ID", "voicebench"))
}

// authenticatesToRouter reads STREAM_ACCELERATION_AUTHENTICATE the way the Python SDK does.
func authenticatesToRouter() bool {
	switch strings.ToLower(os.Getenv("STREAM_ACCELERATION_AUTHENTICATE")) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
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
