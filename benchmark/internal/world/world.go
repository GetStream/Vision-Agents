package world

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
)

// ToolCall is one agent tool invocation.
type ToolCall struct {
	Name       string         `json:"name"`
	Args       map[string]any `json:"args"`
	Result     any            `json:"result"`
	Error      string         `json:"error,omitempty"`
	Started    time.Time      `json:"started"`
	Ended      time.Time      `json:"ended"`
	DurationMS int            `json:"duration_ms"`
}

type handler func(s *Session, args map[string]any) (any, error)

// Session is one seeded world.
type Session struct {
	ID         string
	ScenarioID string
	Pack       string
	State      map[string]any
	Tools      []ToolCall
	Delays     map[string]time.Duration
}

// Server holds the active session and HTTP routes.
type Server struct {
	mu       sync.Mutex
	session  *Session
	handlers map[string]handler
	http     *http.Server
	Addr     string
	Logger   *slog.Logger
}

// New builds a world server with pack handlers registered.
func New(logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	s := &Server{handlers: map[string]handler{}, Logger: logger}
	registerRestaurant(s)
	registerHealthcare(s)
	registerTelecom(s)
	return s
}

func (s *Server) handle(name string, fn handler) {
	s.handlers[name] = fn
}

// Seed installs the active session from a scenario.
func (s *Server) Seed(sc scenario.Scenario) *Session {
	state := cloneMap(sc.Seed)
	if state == nil {
		state = map[string]any{}
	}
	delays := map[string]time.Duration{}
	for name, ms := range sc.ToolDelayMS {
		delays[name] = time.Duration(ms) * time.Millisecond
	}
	sess := &Session{
		ID:         sc.ID + "-" + time.Now().UTC().Format("150405"),
		ScenarioID: sc.ID,
		Pack:       sc.Pack,
		State:      state,
		Delays:     delays,
	}
	s.mu.Lock()
	s.session = sess
	s.mu.Unlock()
	return sess
}

// Snapshot returns a copy of the active session.
func (s *Server) Snapshot() *Session {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.session == nil {
		return nil
	}
	copySess := *s.session
	copySess.State = cloneMap(s.session.State)
	copySess.Tools = append([]ToolCall(nil), s.session.Tools...)
	return &copySess
}

// ListenAndServe starts the HTTP API.
func (s *Server) ListenAndServe(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("GET /v1/active", s.getActive)
	mux.HandleFunc("GET /v1/session/state", s.getState)
	mux.HandleFunc("GET /v1/session/tools", s.getTools)
	mux.HandleFunc("POST /v1/session/tools/", s.postTool)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	s.Addr = listener.Addr().String()
	s.http = &http.Server{Handler: mux}
	go func() { _ = s.http.Serve(listener) }()
	return nil
}

func (s *Server) Close() error {
	if s.http == nil {
		return nil
	}
	return s.http.Close()
}

func (s *Server) getActive(w http.ResponseWriter, r *http.Request) {
	sess := s.Snapshot()
	if sess == nil {
		http.Error(w, "no active session", http.StatusNotFound)
		return
	}
	writeJSON(w, map[string]any{"id": sess.ID, "scenario": sess.ScenarioID, "pack": sess.Pack})
}

func (s *Server) getState(w http.ResponseWriter, r *http.Request) {
	sess := s.Snapshot()
	if sess == nil {
		http.Error(w, "no active session", http.StatusNotFound)
		return
	}
	writeJSON(w, sess.State)
}

func (s *Server) getTools(w http.ResponseWriter, r *http.Request) {
	sess := s.Snapshot()
	if sess == nil {
		http.Error(w, "no active session", http.StatusNotFound)
		return
	}
	writeJSON(w, sess.Tools)
}

func (s *Server) postTool(w http.ResponseWriter, r *http.Request) {
	name := strings.TrimPrefix(r.URL.Path, "/v1/session/tools/")
	name = strings.Trim(name, "/")
	body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	args := map[string]any{}
	if len(body) > 0 {
		if err := json.Unmarshal(body, &args); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}

	s.mu.Lock()
	sess := s.session
	fn, ok := s.handlers[name]
	s.mu.Unlock()
	if sess == nil {
		http.Error(w, "no active session", http.StatusNotFound)
		return
	}
	if !ok {
		http.Error(w, "unknown tool "+name, http.StatusNotFound)
		return
	}
	started := time.Now()
	if d := sess.Delays[name]; d > 0 {
		time.Sleep(d)
	}
	result, err := fn(sess, args)
	ended := time.Now()
	call := ToolCall{
		Name:       name,
		Args:       args,
		Result:     result,
		Started:    started,
		Ended:      ended,
		DurationMS: int(ended.Sub(started).Milliseconds()),
	}
	if err != nil {
		call.Error = err.Error()
		call.Result = map[string]any{"error": err.Error()}
	}
	s.mu.Lock()
	sess.Tools = append(sess.Tools, call)
	s.mu.Unlock()
	writeJSON(w, call.Result)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}
	raw, _ := json.Marshal(in)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return out
}

func getMap(state map[string]any, key string) map[string]any {
	v, _ := state[key].(map[string]any)
	if v == nil {
		v = map[string]any{}
		state[key] = v
	}
	return v
}

func getList(state map[string]any, key string) []any {
	v, _ := state[key].([]any)
	if v == nil {
		v = []any{}
		state[key] = v
	}
	return v
}

func strArg(args map[string]any, key string) string {
	v, _ := args[key].(string)
	return v
}

func boolArg(args map[string]any, key string) bool {
	v, _ := args[key].(bool)
	return v
}

func numArg(args map[string]any, key string) float64 {
	switch v := args[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case json.Number:
		n, _ := v.Float64()
		return n
	default:
		return 0
	}
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	default:
		return fmt.Sprint(t)
	}
}

var (
	clockRe  = regexp.MustCompile(`(?i)(\d{1,2})(?::(\d{2}))\s*(a\.?m\.?|p\.?m\.?)?`)
	dayWords = []string{
		"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
		"today", "tomorrow", "tonight", "this",
	}
)

// normalizeTime maps spoken clock strings onto the scenario seed form (h:mm 12-hour).
func normalizeTime(raw string) string {
	s := strings.ToLower(strings.TrimSpace(raw))
	if s == "" {
		return ""
	}
	for _, w := range dayWords {
		s = strings.ReplaceAll(s, w, " ")
	}
	s = strings.ReplaceAll(s, " at ", " ")
	s = strings.Join(strings.Fields(s), " ")
	m := clockRe.FindStringSubmatch(s)
	if m == nil {
		return s
	}
	hour, _ := strconv.Atoi(m[1])
	min, _ := strconv.Atoi(m[2])
	if hour >= 13 && hour <= 23 {
		hour -= 12
	}
	if hour == 0 {
		hour = 12
	}
	return fmt.Sprintf("%d:%02d", hour, min)
}

// Lookup walks a dotted path in a nested map/list document.
func Lookup(state map[string]any, path string) (any, bool) {
	cur := any(state)
	for _, part := range strings.Split(path, ".") {
		switch node := cur.(type) {
		case map[string]any:
			next, ok := node[part]
			if !ok {
				return nil, false
			}
			cur = next
		case []any:
			idx := -1
			fmt.Sscanf(part, "%d", &idx)
			if idx < 0 || idx >= len(node) {
				return nil, false
			}
			cur = node[idx]
		default:
			return nil, false
		}
	}
	return cur, true
}

// CheckAssertions evaluates end-state rules.
func CheckAssertions(state map[string]any, assertions []scenario.Assertion) []string {
	var fails []string
	for _, a := range assertions {
		got, ok := Lookup(state, a.Path)
		if a.Exists {
			if !ok {
				fails = append(fails, a.Path+" missing")
			}
			continue
		}
		if a.Neq != nil {
			if ok && fmt.Sprint(got) == fmt.Sprint(a.Neq) {
				fails = append(fails, a.Path+" equals forbidden "+fmt.Sprint(a.Neq))
			}
			continue
		}
		if a.Eq != nil {
			gotNorm := normalize(got)
			want := normalize(a.Eq)
			if !ok || fmt.Sprint(gotNorm) != fmt.Sprint(want) {
				fails = append(fails, fmt.Sprintf("%s want %v got %v", a.Path, a.Eq, got))
			}
		}
	}
	return fails
}

func normalize(v any) any {
	switch t := v.(type) {
	case int:
		return float64(t)
	case int64:
		return float64(t)
	default:
		return v
	}
}

// CheckToolOrder fails if After was called without Before earlier.
func CheckToolOrder(tools []ToolCall, rules []scenario.OrderConstraint) []string {
	index := map[string]int{}
	for i, t := range tools {
		if _, ok := index[t.Name]; !ok {
			index[t.Name] = i
		}
	}
	var fails []string
	for _, rule := range rules {
		before, bok := index[rule.Before]
		after, aok := index[rule.After]
		if !aok {
			continue
		}
		if !bok || before > after {
			fails = append(fails, fmt.Sprintf("%s must run before %s", rule.Before, rule.After))
		}
	}
	return fails
}

// EntityInTools reports which required entities never appeared in tool args.
func EntityInTools(tools []ToolCall, entities []scenario.Entity) []string {
	var fails []string
	blob, _ := json.Marshal(tools)
	text := string(blob)
	for _, e := range entities {
		if !e.InTools {
			continue
		}
		if !scenario.MatchValue(text, e.Value) {
			fails = append(fails, e.Name+"="+e.Value)
		}
	}
	return fails
}
