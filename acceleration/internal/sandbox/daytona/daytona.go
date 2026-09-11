// Package daytona runs Python through Daytona's official Go SDK.
package daytona

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/daytona/clients/sdk-go/pkg/options"
	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

const apiKeyEnvVar = "DAYTONA_API_KEY"
const defaultTimeout = 60 * time.Second
const defaultRunTimeout = 30 * time.Second

type Options struct {
	APIKey, APIURL string
	// Deprecated: the official SDK obtains the toolbox address from Daytona.
	ProxyURL            string
	Timeout, RunTimeout time.Duration
	HTTPClient          *http.Client
	Logger              *slog.Logger
}

// A cancellable gate serializes creation, execution and deletion. Identity survives a
// failed delete so Close can be retried; a closed sandbox can never create another VM.
type Sandbox struct {
	client              *sdk.Client
	box                 *sdk.Sandbox
	gate                chan struct{}
	closed              atomic.Bool
	timeout, runTimeout time.Duration
}

func New(o Options) (*Sandbox, error) {
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APIKey == "" {
		return nil, errors.New("daytona: DAYTONA_API_KEY is required")
	}
	if o.Timeout <= 0 {
		o.Timeout = defaultTimeout
	}
	if o.RunTimeout <= 0 {
		o.RunTimeout = defaultRunTimeout
	}
	c, err := sdk.NewClientWithConfig(&types.DaytonaConfig{APIKey: o.APIKey, APIUrl: o.APIURL, HTTPClient: o.HTTPClient})
	if err != nil {
		return nil, err
	}
	return &Sandbox{client: c, gate: make(chan struct{}, 1), timeout: o.Timeout, runTimeout: o.RunTimeout}, nil
}
func Configured() bool { return os.Getenv(apiKeyEnvVar) != "" }
func (s *Sandbox) Run(ctx context.Context, code string) (sandbox.Result, error) {
	if strings.TrimSpace(code) == "" {
		return sandbox.Result{}, errors.New("daytona: no code to run")
	}
	select {
	case s.gate <- struct{}{}:
		defer func() { <-s.gate }()
	case <-ctx.Done():
		return sandbox.Result{}, ctx.Err()
	}
	if s.closed.Load() {
		return sandbox.Result{}, errors.New("daytona: sandbox is closed")
	}
	if s.box == nil {
		create, end := context.WithTimeout(ctx, s.timeout)
		box, err := s.client.Create(create, types.SnapshotParams{SandboxBaseParams: types.SandboxBaseParams{Language: types.CodeLanguagePython}}, options.WithWaitForStart(false))
		end()
		if err != nil {
			return sandbox.Result{}, err
		}
		s.box = box
	}
	ready, end := context.WithTimeout(ctx, s.timeout)
	err := s.box.WaitForStart(ready, s.timeout)
	end()
	if err != nil {
		return sandbox.Result{}, err
	}
	run, cancel := context.WithTimeout(ctx, s.runTimeout+5*time.Second)
	defer cancel()
	result, err := s.box.Process.CodeRun(run, code, options.WithCodeRunTimeout(s.runTimeout))
	if err != nil {
		return sandbox.Result{}, err
	}
	return sandbox.Result{Output: result.Result, ExitCode: result.ExitCode}, nil
}
func (s *Sandbox) Close() error {
	s.closed.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	select {
	case s.gate <- struct{}{}:
		defer func() { <-s.gate }()
	case <-ctx.Done():
		return ctx.Err()
	}

	if s.box == nil {
		return s.client.Close(ctx)
	}
	if err := s.box.Delete(ctx); err != nil {
		return err
	}
	s.box = nil
	return s.client.Close(ctx)
}
