package target

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"

	"github.com/GetStream/Vision-Agents/benchmark/internal/livekitrtc"
)

type liveKitDispatchMetadata struct {
	CallID       string      `json:"call_id"`
	Pack         string      `json:"pack"`
	WorldURL     string      `json:"world_url"`
	Instructions string      `json:"instructions"`
	Tools        []AccelTool `json:"tools"`
}

// LiveKitWorkerAgentName is the agent name the spawned reference worker registers under.
const LiveKitWorkerAgentName = "voicebench"

// LiveKit creates an agent dispatch for a LiveKit room.
type LiveKit struct {
	Root         string
	Pack         string
	URL          string
	APIKey       string
	APISecret    string
	AgentName    string
	Deployment   string
	WorldURL     string
	Spawn        bool
	WorkerPort   int
	Instructions string
	Tools        []AccelTool
	Logger       *slog.Logger
}

func (l *LiveKit) Prepare(ctx context.Context) (func(), error) {
	if l.Instructions == "" || len(l.Tools) == 0 {
		instructions, tools, err := LoadPackContract(l.Root, l.Pack)
		if err != nil {
			return nil, err
		}
		l.Instructions = instructions
		l.Tools = tools
	}
	if l.AgentName == "" && l.Spawn {
		l.AgentName = LiveKitWorkerAgentName
	}
	if l.AgentName == "" {
		l.AgentName = os.Getenv("LIVEKIT_AGENT_NAME")
	}
	if l.AgentName == "" {
		return nil, fmt.Errorf("run: LIVEKIT_AGENT_NAME or --livekit-agent is required with --target livekit")
	}
	if l.Deployment == "" {
		l.Deployment = os.Getenv("LIVEKIT_AGENT_DEPLOYMENT")
	}
	if !l.Spawn && isLoopbackURL(l.WorldURL) {
		return nil, fmt.Errorf("run: --target livekit without --spawn needs a world server the worker can reach, but the world URL is %s. Pass --world-url with an address reachable from the worker, or use --spawn to run the reference worker locally", l.WorldURL)
	}
	options := livekitrtc.Options{URL: l.URL, APIKey: l.APIKey, APISecret: l.APISecret}
	if err := options.Resolve(); err != nil {
		return nil, err
	}
	l.URL, l.APIKey, l.APISecret = options.URL, options.APIKey, options.APISecret
	l.logger().Info("livekit cloud", "url", l.URL, "region", options.Region)
	if !l.Spawn {
		return func() {}, nil
	}
	if l.WorkerPort <= 0 {
		l.WorkerPort = 8081
	}
	pipelineEnv := liveKitPipelineEnv()
	stop, err := StartProcess(ctx, Process{
		Command: "uv",
		Args:    []string{"run", "python", "worker.py", "start"},
		Dir:     filepath.Join(l.Root, "agents-livekit"),
		Env: append([]string{
			"VOICEBENCH_WORLD_URL=" + l.WorldURL,
			"VOICEBENCH_WORKER_PORT=" + strconv.Itoa(l.WorkerPort),
			"LIVEKIT_AGENT_NAME=" + l.AgentName,
			"LIVEKIT_URL=" + l.URL,
			"LIVEKIT_API_KEY=" + l.APIKey,
			"LIVEKIT_API_SECRET=" + l.APISecret,
		}, pipelineEnv...),
		DropEnv: []string{"VOICEBENCH_WORLD_URL=", "VOICEBENCH_WORKER_PORT=", "LIVEKIT_AGENT_NAME=",
			"LIVEKIT_URL=", "LIVEKIT_API_KEY=", "LIVEKIT_API_SECRET=",
			"VOICEBENCH_LIVEKIT_PIPELINE=", "VOICEBENCH_LIVEKIT_MODEL=", "VOICEBENCH_LIVEKIT_STT=",
			"VOICEBENCH_LIVEKIT_TTS=", "VOICEBENCH_LIVEKIT_VOICE="},
		ReadyURL:     fmt.Sprintf("http://127.0.0.1:%d/", l.WorkerPort),
		ReadyTimeout: 120 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("run: spawn livekit worker: %w", err)
	}
	l.logger().Info("spawned livekit worker", "agent", l.AgentName, "port", l.WorkerPort)
	return stop, nil
}

func isLoopbackURL(raw string) bool {
	parsed, err := url.Parse(raw)
	if err != nil {
		return false
	}
	host := parsed.Hostname()
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func (l *LiveKit) StartCall(ctx context.Context, callID string, _ string) (func(), error) {
	metadata, err := json.Marshal(liveKitDispatchMetadata{
		CallID:       callID,
		Pack:         l.Pack,
		WorldURL:     l.WorldURL,
		Instructions: l.Instructions,
		Tools:        l.Tools,
	})
	if err != nil {
		return nil, err
	}

	client := lksdk.NewAgentDispatchServiceClient(l.URL, l.APIKey, l.APISecret)
	created, err := client.CreateDispatch(ctx, &livekit.CreateAgentDispatchRequest{
		AgentName:  l.AgentName,
		Room:       callID,
		Metadata:   string(metadata),
		Deployment: l.Deployment,
		Attributes: map[string]string{
			"voicebench.pack": l.Pack,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("run: create livekit dispatch: %w", err)
	}
	if created.Id == "" {
		return nil, fmt.Errorf("run: create livekit dispatch: missing id")
	}
	// The caller waits on transport.Media.WaitForAgent for the worker to actually
	// join the room, which is strictly stronger evidence than dispatch assignment.
	dispatchID := created.Id
	l.logger().Info("livekit dispatch created", "dispatch", dispatchID, "room", callID)
	return func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, err := client.DeleteDispatch(closeCtx, &livekit.DeleteAgentDispatchRequest{DispatchId: dispatchID, Room: callID})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			l.logger().Warn("delete livekit dispatch", "err", err)
		}
	}, nil
}

func (l *LiveKit) logger() *slog.Logger {
	if l.Logger == nil {
		return slog.Default()
	}
	return l.Logger
}

const (
	DefaultLiveKitPipeline      = "inference"
	DefaultLiveKitSTT           = "google/gemini-3.5-transcribe-live"
	DefaultLiveKitModel         = "google/gemini-3.5-flash-lite"
	DefaultLiveKitTTS           = "inworld/inworld-tts-2-flash"
	DefaultLiveKitVoice         = "Ashley"
	DefaultLiveKitRealtimeModel = "gpt-realtime-2"
	DefaultLiveKitRealtimeVoice = "marin"
)

func liveKitPipelineEnv() []string {
	pipeline := envOr("VOICEBENCH_LIVEKIT_PIPELINE", DefaultLiveKitPipeline)
	if pipeline == "realtime" {
		return []string{
			"VOICEBENCH_LIVEKIT_PIPELINE=realtime",
			"VOICEBENCH_LIVEKIT_MODEL=" + envOr("VOICEBENCH_LIVEKIT_MODEL", DefaultLiveKitRealtimeModel),
			"VOICEBENCH_LIVEKIT_VOICE=" + envOr("VOICEBENCH_LIVEKIT_VOICE", DefaultLiveKitRealtimeVoice),
		}
	}
	return []string{
		"VOICEBENCH_LIVEKIT_PIPELINE=" + pipeline,
		"VOICEBENCH_LIVEKIT_MODEL=" + envOr("VOICEBENCH_LIVEKIT_MODEL", DefaultLiveKitModel),
		"VOICEBENCH_LIVEKIT_STT=" + envOr("VOICEBENCH_LIVEKIT_STT", DefaultLiveKitSTT),
		"VOICEBENCH_LIVEKIT_TTS=" + envOr("VOICEBENCH_LIVEKIT_TTS", DefaultLiveKitTTS),
		"VOICEBENCH_LIVEKIT_VOICE=" + envOr("VOICEBENCH_LIVEKIT_VOICE", DefaultLiveKitVoice),
	}
}
