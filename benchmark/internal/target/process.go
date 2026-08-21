package target

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Process is a child process with a readiness endpoint.
type Process struct {
	Command      string
	Args         []string
	Dir          string
	Env          []string
	DropEnv      []string
	ReadyURL     string
	ReadyTimeout time.Duration
}

func StartProcess(ctx context.Context, spec Process) (func(), error) {
	cmd := exec.CommandContext(ctx, spec.Command, spec.Args...)
	cmd.Dir = spec.Dir
	env := make([]string, 0, len(os.Environ())+len(spec.Env))
	for _, e := range os.Environ() {
		if dropsEnv(e, spec.DropEnv) {
			continue
		}
		env = append(env, e)
	}
	cmd.Env = append(env, spec.Env...)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	stop := func() {
		if cmd.Process == nil {
			return
		}
		_ = cmd.Process.Signal(os.Interrupt)
		done := make(chan struct{})
		go func() {
			_ = cmd.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			_ = cmd.Process.Kill()
		}
	}
	if spec.ReadyURL != "" {
		timeout := spec.ReadyTimeout
		if timeout == 0 {
			timeout = 120 * time.Second
		}
		if err := WaitHTTP(ctx, spec.ReadyURL, timeout); err != nil {
			stop()
			return nil, fmt.Errorf("did not become ready at %s: %w", spec.ReadyURL, err)
		}
	}
	return stop, nil
}

func dropsEnv(env string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(env, prefix) {
			return true
		}
	}
	return false
}

func WaitHTTP(ctx context.Context, readyURL string, timeout time.Duration) error {
	client := &http.Client{Timeout: time.Second}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, readyURL, nil)
		if err != nil {
			return err
		}
		resp, err := client.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
		}
		time.Sleep(250 * time.Millisecond)
	}
	return fmt.Errorf("timed out")
}
