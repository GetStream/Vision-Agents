// Package managed owns named research workspaces for the backend lifetime.
package managed

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/daytona/clients/sdk-go/pkg/daytona"
	sdkerrors "github.com/daytona/clients/sdk-go/pkg/errors"
	"github.com/daytona/clients/sdk-go/pkg/options"
	"github.com/daytona/clients/sdk-go/pkg/types"
	"golang.org/x/sync/errgroup"
	"golang.org/x/sys/unix"

	"github.com/GetStream/Vision-Agents/acceleration/internal/research"
)

type Manager struct {
	Profiles map[string]*Workspace
	cancel   context.CancelFunc
	stopped  chan struct{}
	lock     *os.File
	client   *daytona.Client
}
type Workspace struct {
	Profile   research.Profile
	client    *daytona.Client
	box       *daytona.Sandbox
	token     string
	link      *types.PreviewLink
	slots     chan struct{}
	exclusive chan struct{}
	mu        sync.Mutex
	closed    bool
	life      context.Context
	Timings   map[string]int64
}

func Start(ctx context.Context, path string) (*Manager, error) {
	profiles, e := research.Load(path)
	if e != nil {
		return nil, e
	}
	if len(profiles) == 0 {
		return nil, errors.New("research: no profiles configured")
	}
	deployment := os.Getenv("RESEARCH_DEPLOYMENT_ID")
	if deployment == "" || strings.ContainsAny(deployment, "/\\") {
		return nil, errors.New("research: deployment ID required without path separators")
	}
	lock, e := os.OpenFile(filepath.Join(os.TempDir(), "stream-research-"+deployment+".lock"), os.O_CREATE|os.O_RDWR, 0600)
	if e != nil {
		return nil, e
	}
	if e = unix.Flock(int(lock.Fd()), unix.LOCK_EX|unix.LOCK_NB); e != nil {
		lock.Close()
		return nil, errors.New("research: deployment already running on this host")
	}
	client, e := daytona.NewClient()
	if e != nil {
		lock.Close()
		return nil, errors.New("research: Daytona credentials unavailable")
	}
	m := &Manager{Profiles: map[string]*Workspace{}, stopped: make(chan struct{}), lock: lock, client: client}
	life, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	if e = reconcile(ctx, client, deployment); e != nil {
		cancel()
		close(m.stopped)
		return m, errors.Join(e, m.Close())
	}
	for _, p := range profiles {
		w := &Workspace{Profile: p, client: client, slots: make(chan struct{}, 9), exclusive: make(chan struct{}, 1), life: life, Timings: map[string]int64{}}
		m.Profiles[p.Name] = w
		if e = w.prepare(ctx); e != nil {
			cancel()
			close(m.stopped)
			cleanup := m.Close()
			return m, errors.Join(e, cleanup)
		}
	}
	go func() {
		defer close(m.stopped)
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-life.Done():
				return
			case <-ticker.C:
				for _, w := range m.Profiles {
					c, end := context.WithTimeout(life, 60*time.Second)
					w.heartbeat(c)
					end()
				}
			}
		}
	}()
	return m, nil
}
func (m *Manager) Close() error {
	m.cancel()
	<-m.stopped
	var all []error
	for _, w := range m.Profiles {
		w.mu.Lock()
		w.closed = true
		if w.box != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			e := w.box.DeleteAndWait(ctx, 30*time.Second)
			if errors.Is(e, sdkerrors.ErrNotFound) {
				e = nil
			}
			cancel()
			if e != nil {
				all = append(all, errors.New("research: workspace deletion failed; identity retained"))
			} else {
				w.box = nil
			}
		}
		w.mu.Unlock()
	}
	if len(all) == 0 && m.client != nil {
		ctx, end := context.WithTimeout(context.Background(), 5*time.Second)
		all = append(all, m.client.Close(ctx))
		end()
	}
	if errors.Join(all...) == nil && m.lock != nil {
		_ = m.lock.Close()
		m.lock = nil
	}
	return errors.Join(all...)
}
func (m *Manager) Find(name, customer, agent string) (*Workspace, error) {
	w := m.Profiles[name]
	if w == nil || w.Profile.CustomerID != customer || w.Profile.AgentID != agent {
		return nil, errors.New("research: profile not available to this agent")
	}
	return w, nil
}
func quote(s string) string { return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'" }
func (w *Workspace) command(ctx context.Context, command string) (string, error) {
	r, e := w.box.Process.ExecuteCommand(ctx, command, options.WithExecuteTimeout(120*time.Second))
	if e != nil || r.ExitCode != 0 {
		return "", errors.New("research: workspace preparation command failed")
	}
	return r.Result, nil
}
func (w *Workspace) prepare(ctx context.Context) error {
	started := time.Now()
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	deployment := os.Getenv("RESEARCH_DEPLOYMENT_ID")
	if deployment == "" {
		return errors.New("research: RESEARCH_DEPLOYMENT_ID is required")
	}
	labels := map[string]string{"stream-research-deployment": deployment, "stream-research-profile": w.Profile.Name, "stream-research-customer": w.Profile.CustomerID, "stream-research-agent": w.Profile.AgentID}
	var image any = w.Profile.Image
	if w.Profile.Image == "build" {
		binary := os.Getenv("RESEARCH_WORKER_BINARY")
		if binary == "" {
			return errors.New("research: RESEARCH_WORKER_BINARY is required for image build")
		}
		image = daytona.Base("python:3.12-slim-bookworm").AptGet([]string{"git", "curl", "ca-certificates", "util-linux"}).Run("useradd --uid 10001 --create-home support-reader && mkdir -p /opt/cursor /opt/support /opt/repositories").Run("curl -fsSL --max-time 120 https://downloads.cursor.com/lab/2026.09.08-6caf4ff/linux/x64/agent-cli-package.tar.gz -o /tmp/cursor.tar.gz && tar xzf /tmp/cursor.tar.gz --strip-components=1 -C /opt/cursor && rm /tmp/cursor.tar.gz").AddLocalFile(binary, "/opt/support/research-worker").Run("chmod 755 /opt/support/research-worker && chmod -R a+rX,go-w /opt/cursor /opt/support").Entrypoint([]string{"sleep", "infinity"})
	}
	interval := 5
	box, e := w.client.Create(ctx, types.ImageParams{SandboxBaseParams: types.SandboxBaseParams{User: "root", Labels: labels, Public: false, AutoStopInterval: &interval}, Image: image, Resources: &types.Resources{CPU: 2, Memory: 4, Disk: 10}}, options.WithTimeout(10*time.Minute), options.WithWaitForStart(false))
	if e != nil {
		return fmt.Errorf("research: Daytona creation failed: %w", e)
	}
	w.box = box
	if e = box.WaitForStart(ctx, 10*time.Minute); e != nil {
		return fmt.Errorf("research: Daytona readiness failed: %w", e)
	}
	w.Timings["vm_start_ms"] = time.Since(started).Milliseconds()
	clones, cloneCtx := errgroup.WithContext(ctx)
	clones.SetLimit(3)
	for i := range w.Profile.Repositories {
		clones.Go(func() error {
			cloneStarted := time.Now()
			r := &w.Profile.Repositories[i]
			args := "GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git -c core.hooksPath=/dev/null clone --depth 1 --single-branch --no-tags "
			if r.Ref != "" {
				args += "--branch " + quote(r.Ref) + " "
			}
			args += "-- " + quote(r.URL) + " " + quote(research.Root+"/"+r.ID)
			if _, err := w.command(cloneCtx, args); err != nil {
				return fmt.Errorf("research: clone %s: %w", r.ID, err)
			}
			rev, err := w.command(cloneCtx, "git -C "+quote(research.Root+"/"+r.ID)+" rev-parse HEAD")
			if err != nil {
				return err
			}
			r.Revision = strings.TrimSpace(rev)
			depth, err := w.command(cloneCtx, "git -C "+quote(research.Root+"/"+r.ID)+" rev-list --count HEAD")
			if err != nil || strings.TrimSpace(depth) != "1" {
				return fmt.Errorf("research: %s clone is not shallow", r.ID)
			}
			elapsed := time.Since(cloneStarted).Milliseconds()
			w.mu.Lock()
			w.Timings["clone_"+r.ID+"_ms"] = elapsed
			w.mu.Unlock()
			slog.Info("research repository prepared", "repository", r.ID, "revision", r.Revision, "clone_ms", elapsed)
			return nil
		})
	}
	if err := clones.Wait(); err != nil {
		return err
	}
	data, _ := json.Marshal(w.Profile)
	secret := make([]byte, 32)
	if _, e = rand.Read(secret); e != nil {
		return e
	}
	w.token = hex.EncodeToString(secret)
	for name, value := range map[string][]byte{"profile.json": data, "worker-token": []byte(w.token), "cursor-key": []byte(os.Getenv("CURSOR_API_KEY"))} {
		if len(value) == 0 {
			return errors.New("research: Cursor key required")
		}
		if e = w.box.FileSystem.UploadFile(ctx, value, "/opt/support/"+name); e != nil {
			return errors.New("research: failed to upload worker configuration")
		}
	}
	// No repo-provided configuration, hooks, symlinks or hidden files reach Cursor.
	preparation := research.PrepareScript
	if _, e = w.command(ctx, "set -eu\n"+preparation); e != nil {
		return e
	}
	cursorStarted := time.Now()
	if e = w.startWorker(ctx); e != nil {
		return e
	}
	w.link, e = w.box.GetPreviewLink(ctx, 8081)
	if e != nil {
		return errors.New("research: worker endpoint unavailable")
	}
	for {
		r, e := w.request(ctx, "GET", "/health", nil)
		if e == nil {
			_ = r.Body.Close()
			if r.StatusCode == 204 {
				w.Timings["index_cursor_init_ms"] = time.Since(cursorStarted).Milliseconds()
				w.Timings["startup_ms"] = time.Since(started).Milliseconds()
				slog.Info("research workspace ready", "profile", w.Profile.Name, "sandbox", w.box.ID, "timings_ms", w.Timings, "repositories", w.Profile.Repositories)
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return errors.New("research: worker readiness timeout")
		case <-time.After(time.Second):
		}
	}
}
func (w *Workspace) request(ctx context.Context, method, path string, body io.Reader) (*http.Response, error) {
	r, e := http.NewRequestWithContext(ctx, method, strings.TrimRight(w.link.URL, "/")+path, body)
	if e != nil {
		return nil, e
	}
	r.Header.Set("Authorization", "Bearer "+w.token)
	r.Header.Set("X-Daytona-Preview-Token", w.link.Token)
	r.Header.Set("Content-Type", "application/json")
	return http.DefaultClient.Do(r)
}
func (w *Workspace) Research(ctx context.Context, in research.Request, progress func(research.Progress)) research.Result {
	ctx, stop := context.WithCancel(ctx)
	defer stop()
	unregister := context.AfterFunc(w.life, stop)
	defer unregister()
	started := time.Now()
	failure := func(code string) research.Result {
		return research.Result{Status: "research_failed", Code: code, ElapsedMS: time.Since(started).Milliseconds()}
	}
	if _, e := w.Profile.Select(in); e != nil {
		return failure(e.Error())
	}
	select {
	case w.slots <- struct{}{}:
		defer func() { <-w.slots }()
	default:
		return failure("queue_full")
	}
	progress(research.Progress{Phase: "queued"})
	wait, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	select {
	case w.exclusive <- struct{}{}:
		defer func() { <-w.exclusive }()
	case <-wait.Done():
		if errors.Is(wait.Err(), context.DeadlineExceeded) {
			return failure("queue_timeout")
		}
		return failure("queue_cancelled")
	}
	if ctx.Err() != nil {
		return failure("queue_cancelled")
	}
	w.mu.Lock()
	closed := w.closed
	w.mu.Unlock()
	if closed {
		return failure("workspace_closed")
	}
	recovery, done := context.WithTimeout(ctx, 60*time.Second)
	err := w.ensureReady(recovery, progress)
	done()
	if ctx.Err() != nil {
		return failure("cursor_cancelled")
	}
	if err != nil {
		return failure(err.Error())
	}

	run, end := context.WithTimeout(ctx, 65*time.Second)
	defer end()
	data, _ := json.Marshal(in)
	var response *http.Response
	var e error
	for retry := 0; retry < 30; retry++ {
		response, e = w.request(run, "POST", "/research", bytes.NewReader(data))
		if e != nil || response.StatusCode != http.StatusTooManyRequests {
			break
		}
		_ = response.Body.Close()
		select {
		case <-run.Done():
			return failure("cursor_cancelled")
		case <-time.After(100 * time.Millisecond):
		}
	}
	if e != nil {
		return failure("worker_unavailable")
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		return failure("worker_unavailable")
	}
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 4096), 128000)
	for scanner.Scan() {
		var f research.Frame
		if json.Unmarshal(scanner.Bytes(), &f) != nil {
			return failure("worker_invalid_output")
		}
		if f.Progress != nil {
			progress(*f.Progress)
		}
		if f.Result != nil {
			return *f.Result
		}
	}
	if errors.Is(run.Err(), context.Canceled) {
		return failure("cursor_cancelled")
	}
	if run.Err() != nil {
		return failure("cursor_timeout")
	}
	return failure("worker_disconnected")
}

func reconcile(ctx context.Context, client *daytona.Client, deployment string) error {
	ctx, end := context.WithTimeout(ctx, 2*time.Minute)
	defer end()
	old := client.List(ctx, &daytona.ListSandboxesQuery{Labels: map[string]string{"stream-research-deployment": deployment}})
	for old.Next() {
		labels := old.Value().Labels
		if labels["stream-research-deployment"] != deployment || labels["stream-research-profile"] == "" || labels["stream-research-customer"] == "" || labels["stream-research-agent"] == "" {
			continue
		}
		if e := old.Value().DeleteAndWait(ctx, 30*time.Second); e != nil && !errors.Is(e, sdkerrors.ErrNotFound) {
			return errors.New("research: failed to reconcile owned workspace")
		}
	}
	if e := old.Err(); e != nil {
		return errors.New("research: failed to list owned workspaces")
	}
	return nil
}

// healthy checks the authenticated worker without exposing preview credentials in errors.
func (w *Workspace) healthy(ctx context.Context) bool {
	if w.link == nil {
		return false
	}
	probe, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	r, err := w.request(probe, "GET", "/health", nil)
	if err != nil {
		return false
	}
	defer r.Body.Close()
	return r.StatusCode == http.StatusNoContent
}

// ensureReady is called while holding exclusive, so recovery cannot interrupt research.
// A stopped sandbox is resumed in place: no clone, new revision, or replacement VM.
func (w *Workspace) ensureReady(ctx context.Context, progress func(research.Progress)) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return errors.New("workspace_closed")
	}
	if w.healthy(ctx) {
		return nil
	}
	if w.box == nil {
		return errors.New("worker_unavailable")
	}
	progress(research.Progress{Phase: "recovering_workspace"})
	if err := w.box.RefreshData(ctx); err != nil {
		return errors.New("workspace_lookup_failed")
	}
	if w.box.State != "started" {
		slog.Info("resuming research workspace", "profile", w.Profile.Name, "sandbox", w.box.ID, "state", w.box.State)
		if err := w.box.StartWithTimeout(ctx, 45*time.Second); err != nil {
			return errors.New("workspace_resume_failed")
		}
	}
	// Preview routes/tokens may have changed while the VM was stopped.
	link, err := w.box.GetPreviewLink(ctx, 8081)
	if err != nil {
		return errors.New("worker_endpoint_unavailable")
	}
	w.link = link
	if w.healthy(ctx) {
		return nil
	}
	progress(research.Progress{Phase: "starting_worker"})
	if err := w.box.Process.DeleteSession(ctx, "stream-research-worker"); err != nil && !errors.Is(err, sdkerrors.ErrNotFound) {
		return errors.New("worker_session_cleanup_failed")
	}
	if err := w.startWorker(ctx); err != nil {
		return err
	}
	for !w.healthy(ctx) {
		select {
		case <-ctx.Done():
			return errors.New("worker_readiness_timeout")
		case <-time.After(time.Second):
		}
	}
	slog.Info("research workspace recovered", "profile", w.Profile.Name, "sandbox", w.box.ID)
	return nil
}

func (w *Workspace) startWorker(ctx context.Context) error {
	if err := w.box.Process.CreateSession(ctx, "stream-research-worker"); err != nil {
		return errors.New("worker_session_create_failed")
	}
	// Lock the supervisor as well as its child; a stale process cannot spawn another
	// Cursor worker after its process session is replaced.
	command := "flock -n /opt/support/worker.lock sh -c 'while true; do /opt/support/research-worker; sleep 1; done'"
	if _, err := w.box.Process.ExecuteSessionCommand(ctx, "stream-research-worker", command, true, true); err != nil {
		return errors.New("worker_launch_failed")
	}
	return nil
}

func (w *Workspace) heartbeat(ctx context.Context) {
	select {
	case w.exclusive <- struct{}{}:
		defer func() { <-w.exclusive }()
	default:
		return // An active investigation owns the worker and is bounded to one minute.
	}
	if err := w.ensureReady(ctx, func(research.Progress) {}); err != nil {
		slog.Warn("research workspace recovery failed", "profile", w.Profile.Name, "reason", err.Error())
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.box != nil && !w.closed {
		if _, err := w.box.Process.ExecuteCommand(ctx, "true", options.WithExecuteTimeout(5*time.Second)); err != nil {
			slog.Warn("research workspace heartbeat failed", "profile", w.Profile.Name)
		}
	}
}
