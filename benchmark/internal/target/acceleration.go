package target

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"gopkg.in/yaml.v3"
)

const accelCustomer = "voicebench"

type AccelTool struct {
	Name        string         `yaml:"name" json:"name"`
	Description string         `yaml:"description" json:"description"`
	Parameters  map[string]any `yaml:"parameters,omitempty" json:"parameters,omitempty"`
}

type accelToolsFile struct {
	Tools []AccelTool `yaml:"tools"`
}

type accelSessionRequest struct {
	CallID       string      `json:"call_id"`
	CallType     string      `json:"call_type,omitempty"`
	UserID       string      `json:"user_id,omitempty"`
	Instructions string      `json:"instructions,omitempty"`
	Greeting     string      `json:"greeting,omitempty"`
	Tools        []AccelTool `json:"tools"`
}

type accelSession struct {
	ID     string `json:"id"`
	CallID string `json:"call_id"`
}

type accelConn struct {
	conn  *websocket.Conn
	write sync.Mutex
	done  chan struct{}
}

// Acceleration starts and controls the Go acceleration router.
type Acceleration struct {
	Root         string
	Pack         string
	URL          string
	Spawn        bool
	Bin          string
	WorldURL     string
	Instructions string
	Tools        []AccelTool
	Logger       *slog.Logger
}

func (a *Acceleration) Prepare(ctx context.Context) (func(), error) {
	if a.Instructions == "" || len(a.Tools) == 0 {
		instructions, tools, err := LoadPackContract(a.Root, a.Pack)
		if err != nil {
			return nil, err
		}
		a.Instructions = instructions
		a.Tools = tools
	}
	if !a.Spawn {
		if a.URL == "" {
			return nil, fmt.Errorf("run: --target-url is required for an acceleration target without --spawn")
		}
		return func() {}, nil
	}
	if a.URL == "" {
		a.URL = "http://127.0.0.1:8080"
	}
	if a.Bin == "" {
		a.Bin = os.Getenv("ACCEL_ROUTER")
	}
	if a.Bin == "" {
		return nil, fmt.Errorf("run: --bin or ACCEL_ROUTER is required with --target acceleration --spawn")
	}
	stop, err := StartRouter(ctx, a.Bin, a.URL)
	if err != nil {
		return nil, err
	}
	a.logger().Info("spawned accel router", "url", a.URL)
	return stop, nil
}

// StartRouter launches the acceleration router and waits until /health succeeds.
func StartRouter(ctx context.Context, bin, baseURL string) (func(), error) {
	if bin == "" {
		bin = os.Getenv("ACCEL_ROUTER")
	}
	if bin == "" {
		return nil, fmt.Errorf("run: --bin or ACCEL_ROUTER is required to spawn the router")
	}
	if baseURL == "" {
		baseURL = "http://127.0.0.1:8080"
	}
	addr := "127.0.0.1:8080"
	if parsed, err := url.Parse(baseURL); err == nil && parsed.Host != "" {
		addr = parsed.Host
	}
	stop, err := StartProcess(ctx, Process{
		Command:      bin,
		Env:          []string{"ROUTER_ADDR=" + addr},
		DropEnv:      []string{"ROUTER_ADDR="},
		ReadyURL:     strings.TrimRight(baseURL, "/") + "/health",
		ReadyTimeout: 120 * time.Second,
	})
	if err != nil {
		return nil, fmt.Errorf("run: spawn accel router: %w", err)
	}
	return stop, nil
}

func (a *Acceleration) StartCall(ctx context.Context, callID string, callType string) (func(), error) {
	body, err := json.Marshal(accelSessionRequest{
		CallID:       callID,
		CallType:     callType,
		UserID:       "accel-agent",
		Instructions: a.Instructions,
		Greeting:     "Hello, how can I help?",
		Tools:        a.Tools,
	})
	if err != nil {
		return nil, err
	}

	base := strings.TrimRight(a.URL, "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/v1/agents/sessions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Customer-Id", accelCustomer)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("run: create accel session: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("run: create accel session: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var created accelSession
	if err := json.Unmarshal(raw, &created); err != nil {
		return nil, fmt.Errorf("run: create accel session: %w", err)
	}
	if created.ID == "" {
		return nil, fmt.Errorf("run: create accel session: missing id")
	}

	wsURL, err := AccelEventsURL(base, created.ID)
	if err != nil {
		closeAccelSession(context.Background(), a, created.ID)
		return nil, err
	}
	header := http.Header{}
	header.Set("X-Customer-Id", accelCustomer)
	conn, _, err := websocket.DefaultDialer.DialContext(ctx, wsURL, header)
	if err != nil {
		closeAccelSession(context.Background(), a, created.ID)
		return nil, fmt.Errorf("run: accel events socket: %w", err)
	}

	watchCtx, cancel := context.WithCancel(ctx)
	session := &accelConn{conn: conn, done: make(chan struct{})}
	go session.serveTools(watchCtx, a.WorldURL)

	a.logger().Info("accel session ready", "session", created.ID, "call", callID)
	return func() {
		cancel()
		_ = session.writeJSON(map[string]any{"type": "close"})
		_ = conn.Close()
		select {
		case <-session.done:
		case <-time.After(2 * time.Second):
		}
		closeCtx, closeCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer closeCancel()
		closeAccelSession(closeCtx, a, created.ID)
	}, nil
}

func (a *Acceleration) logger() *slog.Logger {
	if a.Logger == nil {
		return slog.Default()
	}
	return a.Logger
}

func LoadPackContract(root, pack string) (instructions string, tools []AccelTool, err error) {
	promptRaw, err := os.ReadFile(filepath.Join(root, "agents", "contracts", pack+".prompt"))
	if err != nil {
		return "", nil, fmt.Errorf("run: accel instructions: %w", err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "agents", "contracts", pack+".tools.yaml"))
	if err != nil {
		return "", nil, fmt.Errorf("run: accel tools: %w", err)
	}
	var file accelToolsFile
	if err := yaml.Unmarshal(raw, &file); err != nil {
		return "", nil, fmt.Errorf("run: accel tools: %w", err)
	}
	if len(file.Tools) == 0 {
		return "", nil, fmt.Errorf("run: accel tools: %s.tools.yaml has no tools", pack)
	}
	return strings.TrimSpace(string(promptRaw)), file.Tools, nil
}

func (s *accelConn) serveTools(ctx context.Context, worldURL string) {
	defer close(s.done)
	for {
		var frame map[string]any
		if err := s.conn.ReadJSON(&frame); err != nil {
			return
		}
		if ctx.Err() != nil {
			return
		}
		if fmt.Sprint(frame["type"]) != "tool_call" {
			continue
		}
		id := fmt.Sprint(frame["id"])
		name := fmt.Sprint(frame["name"])
		args := "{}"
		switch v := frame["arguments"].(type) {
		case string:
			if strings.TrimSpace(v) != "" {
				args = v
			}
		default:
			if v != nil {
				raw, err := json.Marshal(v)
				if err == nil {
					args = string(raw)
				}
			}
		}
		output, fail := CallWorldTool(ctx, worldURL, name, args)
		result := map[string]any{"type": "tool_result", "tool_call_id": id}
		if fail != "" {
			result["error"] = fail
		} else {
			result["output"] = output
		}
		if err := s.writeJSON(result); err != nil {
			return
		}
	}
}

func (s *accelConn) writeJSON(v any) error {
	s.write.Lock()
	defer s.write.Unlock()
	_ = s.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
	return s.conn.WriteJSON(v)
}

func CallWorldTool(ctx context.Context, worldURL, name, args string) (string, string) {
	url := strings.TrimRight(worldURL, "/") + "/v1/session/tools/" + name
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(args))
	if err != nil {
		return "", err.Error()
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err.Error()
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", err.Error()
	}
	body := strings.TrimSpace(string(raw))
	if resp.StatusCode >= 300 {
		if body == "" {
			body = resp.Status
		}
		return "", fmt.Sprintf("HTTP %d: %s", resp.StatusCode, body)
	}
	if body == "" {
		return "{}", ""
	}
	return body, ""
}

func closeAccelSession(ctx context.Context, cfg *Acceleration, id string) {
	if id == "" || cfg.URL == "" {
		return
	}
	base := strings.TrimRight(cfg.URL, "/")
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, base+"/v1/agents/sessions/"+id, nil)
	if err != nil {
		cfg.logger().Warn("close accel session", "err", err)
		return
	}
	req.Header.Set("X-Customer-Id", accelCustomer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		cfg.logger().Warn("close accel session", "err", err)
		return
	}
	_ = resp.Body.Close()
}

func AccelEventsURL(httpBase, sessionID string) (string, error) {
	parsed, err := url.Parse(httpBase)
	if err != nil {
		return "", err
	}
	switch parsed.Scheme {
	case "https":
		parsed.Scheme = "wss"
	default:
		parsed.Scheme = "ws"
	}
	parsed.Path = "/v1/agents/sessions/" + sessionID + "/events"
	parsed.RawQuery = ""
	return parsed.String(), nil
}
