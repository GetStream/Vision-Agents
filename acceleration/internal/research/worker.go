package research

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"sync"
	"time"
)

func cursorKey() string { return os.Getenv("CURSOR_API_KEY") }

type Worker struct {
	Profile Profile
	Index   *Index
	Token   string
	ACP     *ACP
	// Only the backend queues work. Refuse overlapping direct worker requests.
	closed bool
	busy   chan struct{}
	mu     sync.Mutex
}

func NewWorker(ctx context.Context, p Profile, root, token string) (*Worker, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	for _, r := range p.Repositories {
		if !sha.MatchString(r.Revision) {
			return nil, errors.New("repository revision missing")
		}
	}
	idx, e := BuildIndex(root, p.Repositories)
	if e != nil {
		return nil, e
	}
	a, e := StartACP(ctx, root, p.Model)
	if e != nil {
		return nil, e
	}
	return &Worker{Profile: p, Index: idx, Token: token, ACP: a, busy: make(chan struct{}, 1)}, nil
}
func (w *Worker) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.closed = true
	if w.ACP != nil {
		w.ACP.Close()
	}
}
func (w *Worker) ServeHTTP(out http.ResponseWriter, r *http.Request) {
	if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")), []byte("Bearer "+w.Token)) != 1 {
		http.Error(out, "unauthorized", 401)
		return
	}
	if r.URL.Path == "/health" {
		w.mu.Lock()
		alive := !w.closed && w.ACP != nil && w.ACP.Alive()
		w.mu.Unlock()
		if !alive {
			select {
			case w.busy <- struct{}{}:
				recoverCtx, end := context.WithTimeout(r.Context(), 15*time.Second)
				w.mu.Lock()
				var err error
				if w.closed {
					err = errors.New("worker closed")
				} else {
					w.ACP, err = StartACP(recoverCtx, w.Index.Root, w.Profile.Model)
				}
				alive = err == nil
				w.mu.Unlock()
				end()
				<-w.busy
			default:
			}
		}
		if alive {
			out.WriteHeader(204)
		} else {
			http.Error(out, "Cursor unavailable", 503)
		}
		return
	}
	if r.URL.Path != "/research" || r.Method != "POST" {
		http.NotFound(out, r)
		return
	}
	var in Request
	if json.NewDecoder(http.MaxBytesReader(out, r.Body, 8192)).Decode(&in) != nil {
		http.Error(out, "invalid request", 400)
		return
	}
	repos, e := w.Profile.Select(in)
	if e != nil {
		http.Error(out, e.Error(), 400)
		return
	}
	select {
	case w.busy <- struct{}{}:
		defer func() { <-w.busy }()
	default:
		http.Error(out, "busy", 429)
		return
	}
	started := time.Now()
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	out.Header().Set("Content-Type", "application/x-ndjson")
	out.Header().Set("Cache-Control", "no-store")
	encoder := json.NewEncoder(out)
	emit := func(phase string) {
		_ = encoder.Encode(Frame{Progress: &Progress{Phase: phase, ElapsedMS: time.Since(started).Milliseconds()}})
		if f, ok := out.(http.Flusher); ok {
			f.Flush()
		}
	}
	w.mu.Lock()
	if w.closed {
		e = errors.New("worker closed")
	} else if w.ACP == nil || !w.ACP.Alive() {
		w.ACP, e = StartACP(ctx, w.Index.Root, w.Profile.Model)
	}
	active := w.ACP
	w.mu.Unlock()
	var text string
	if e == nil {
		emit("researching")
		text, e = active.Research(ctx, in, repos, w.Index.Context(in.Question, repos), emit)
	}
	result := Result{Status: "research_failed", Code: "cursor_command_failed"}
	if e == nil {
		emit("verifying_citations")
		result, e = Verify(text, w.Index.Root, repos)
		if e != nil {
			result = Result{Status: "research_failed", Code: e.Error()}
		}
	} else if errors.Is(e, context.Canceled) {
		result.Code = "cursor_cancelled"
	} else if errors.Is(e, context.DeadlineExceeded) {
		result.Code = "cursor_timeout"
	}
	result.ElapsedMS = time.Since(started).Milliseconds()
	_ = encoder.Encode(Frame{Result: &result})
}
