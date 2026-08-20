package target

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Python starts and controls the Python Vision Agents benchmark server.
type Python struct {
	Root     string
	Pack     string
	URL      string
	Spawn    bool
	Port     int
	WorldURL string
	Logger   *slog.Logger
}

func (p *Python) Prepare(ctx context.Context) (func(), error) {
	if !p.Spawn {
		return func() {}, nil
	}
	if p.Port <= 0 {
		p.Port = 8000
	}
	if p.URL == "" {
		p.URL = fmt.Sprintf("http://127.0.0.1:%d", p.Port)
	}
	stop, err := StartProcess(ctx, Process{
		Command: "uv",
		Args: []string{"run", "python", "-m", "voicebench_agents", p.Pack,
			"--host", "127.0.0.1", "--port", strconv.Itoa(p.Port)},
		Dir:          filepath.Join(p.Root, "agents"),
		Env:          []string{"VOICEBENCH_WORLD_URL=" + p.WorldURL},
		DropEnv:      []string{"VOICEBENCH_WORLD_URL=", "WORLD_URL="},
		ReadyURL:     fmt.Sprintf("http://127.0.0.1:%d/ready", p.Port),
		ReadyTimeout: 120 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("run: spawn agent: %w", err)
	}
	p.logger().Info("spawned agent ready", "url", p.URL)
	return stop, nil
}

func (p *Python) StartCall(ctx context.Context, callID string, callType string) (func(), error) {
	sessionID, err := startAgentSession(ctx, p.URL, callID, callType)
	if err != nil {
		return nil, err
	}
	return func() {
		closeCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		closeAgentSession(closeCtx, p.logger(), p.URL, callID, sessionID)
	}, nil
}

func (p *Python) logger() *slog.Logger {
	if p.Logger == nil {
		return slog.Default()
	}
	return p.Logger
}

func startAgentSession(ctx context.Context, agentURL, callID, callType string) (string, error) {
	base := strings.TrimRight(agentURL, "/")
	body, _ := json.Marshal(map[string]string{"call_type": callType})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/calls/"+callID+"/sessions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("run: start agent session: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return "", fmt.Errorf("run: start agent session: HTTP %d", resp.StatusCode)
	}
	var parsed struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("run: start agent session: %w", err)
	}
	if parsed.SessionID == "" {
		return "", fmt.Errorf("run: start agent session: missing session_id")
	}
	return parsed.SessionID, nil
}

func closeAgentSession(ctx context.Context, logger *slog.Logger, agentURL, callID, sessionID string) {
	if sessionID == "" {
		return
	}
	if logger == nil {
		logger = slog.Default()
	}
	base := strings.TrimRight(agentURL, "/")
	url := base + "/calls/" + callID + "/sessions/" + sessionID
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, url, nil)
	if err != nil {
		logger.Warn("close agent session", "err", err)
		return
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		logger.Warn("close agent session", "err", err)
		return
	}
	_ = resp.Body.Close()

	client := &http.Client{Timeout: time.Second}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusNotFound {
				return
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	logger.Warn("agent session still open after close", "session", sessionID, "call", callID)
}
