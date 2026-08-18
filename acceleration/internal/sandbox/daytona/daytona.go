// Package daytona runs code in a Daytona sandbox.
//
// Daytona publishes a Go SDK, but this speaks its REST API directly for the same reason
// the other providers here do: two calls are wanted, and a dependency that pulls in a
// client, a model layer and a config loader to make them is a poor trade.
//
// One sandbox is created on first use and kept for the life of the agent. Creating one
// takes long enough to notice, and a conversation that delegates twice should not pay for
// it twice.
package daytona

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

const apiKeyEnvVar = "DAYTONA_API_KEY"

const (
	defaultAPIURL   = "https://app.daytona.io/api"
	defaultProxyURL = "https://proxy.app.daytona.io/toolbox"
	// defaultTimeout bounds one call. Creating a sandbox is the slow one, and it is still
	// meant to be quick: Daytona boots them in well under a second.
	defaultTimeout = 60 * time.Second
	// defaultRunTimeout is how long a piece of code may run before Daytona stops it.
	defaultRunTimeout = 30 * time.Second
	// language is what stateless code runs as. The sandbox is created for it, so it is
	// fixed here rather than asked of the model, which would only get it wrong.
	language = "python"
)

// errorBodyLimit caps how much of a failed response is read into an error message.
const errorBodyLimit = 2048

// Options configures a Sandbox. The key falls back to the environment, the way every
// other provider in this service is configured.
type Options struct {
	// APIKey defaults to DAYTONA_API_KEY.
	APIKey string
	// APIURL is where sandboxes are created. It defaults to the hosted platform.
	APIURL string
	// ProxyURL is where code is run. It defaults to the hosted platform.
	ProxyURL string
	// Timeout bounds one HTTP call.
	Timeout time.Duration
	// RunTimeout is how long a piece of code may run.
	RunTimeout time.Duration
	// HTTPClient replaces the one built from Timeout.
	HTTPClient *http.Client
	Logger     *slog.Logger
}

// Sandbox runs code in Daytona. It satisfies sandbox.Sandbox.
type Sandbox struct {
	apiKey     string
	apiURL     string
	proxyURL   string
	runTimeout time.Duration
	client     *http.Client
	logger     *slog.Logger

	mu sync.Mutex
	// id is the sandbox, created on first use and released by Close.
	id     string
	closed bool
}

// New validates the options and returns a Sandbox. It creates nothing: the sandbox is
// made the first time code is run, so an agent that never delegates never pays for one.
func New(options Options) (*Sandbox, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, errors.New("daytona: " + apiKeyEnvVar + " is required")
	}
	if options.APIURL == "" {
		options.APIURL = defaultAPIURL
	}
	if options.ProxyURL == "" {
		options.ProxyURL = defaultProxyURL
	}
	if options.Timeout <= 0 {
		options.Timeout = defaultTimeout
	}
	if options.RunTimeout <= 0 {
		options.RunTimeout = defaultRunTimeout
	}
	if options.HTTPClient == nil {
		options.HTTPClient = &http.Client{Timeout: options.Timeout}
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Sandbox{
		apiKey:     options.APIKey,
		apiURL:     strings.TrimSuffix(options.APIURL, "/"),
		proxyURL:   strings.TrimSuffix(options.ProxyURL, "/"),
		runTimeout: options.RunTimeout,
		client:     options.HTTPClient,
		logger:     options.Logger,
	}, nil
}

// Configured reports whether a Daytona key is available, so a caller can offer code
// execution when it can and stay quiet about it when it cannot.
func Configured() bool { return os.Getenv(apiKeyEnvVar) != "" }

// Run executes a piece of Python and returns what it printed.
func (s *Sandbox) Run(ctx context.Context, code string) (sandbox.Result, error) {
	if strings.TrimSpace(code) == "" {
		return sandbox.Result{}, errors.New("daytona: there is no code to run")
	}

	id, err := s.sandbox(ctx)
	if err != nil {
		return sandbox.Result{}, err
	}

	body := runRequest{
		Code:     code,
		Language: language,
		Timeout:  int(s.runTimeout.Seconds()),
	}

	var ran runResponse
	path := "/" + id + "/process/code-run"
	if err := s.call(ctx, http.MethodPost, s.proxyURL+path, body, &ran); err != nil {
		return sandbox.Result{}, err
	}
	return sandbox.Result{Output: ran.Result, ExitCode: ran.ExitCode}, nil
}

// Close releases the sandbox. Safe to call twice.
func (s *Sandbox) Close() error {
	s.mu.Lock()
	id := s.id
	s.id = ""
	s.closed = true
	s.mu.Unlock()

	if id == "" {
		return nil
	}

	// The context is the sandbox's own rather than a caller's: this runs during shutdown,
	// when whatever context the work had is already cancelled, and a sandbox left behind
	// goes on being billed.
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	if err := s.call(ctx, http.MethodDelete, s.apiURL+"/sandbox/"+id, nil, nil); err != nil {
		return fmt.Errorf("daytona: release sandbox %s: %w", id, err)
	}
	return nil
}

// sandbox returns the sandbox to run in, creating it the first time.
func (s *Sandbox) sandbox(ctx context.Context) (string, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return "", errors.New("daytona: the sandbox is closed")
	}
	if s.id != "" {
		id := s.id
		s.mu.Unlock()
		return id, nil
	}
	s.mu.Unlock()

	var created sandboxResponse
	if err := s.call(ctx, http.MethodPost, s.apiURL+"/sandbox",
		sandboxRequest{Language: language}, &created); err != nil {
		return "", err
	}
	if created.ID == "" {
		return "", errors.New("daytona: the sandbox that was created has no id")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// Two delegations racing here each created one. The loser's is released rather than
	// leaked, since only one can be remembered.
	if s.id != "" {
		go s.release(created.ID)
		return s.id, nil
	}
	if s.closed {
		go s.release(created.ID)
		return "", errors.New("daytona: the sandbox is closed")
	}
	s.id = created.ID
	s.logger.Debug("created a sandbox", "sandbox", created.ID)
	return s.id, nil
}

// release deletes a sandbox nothing is going to use.
func (s *Sandbox) release(id string) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	if err := s.call(ctx, http.MethodDelete, s.apiURL+"/sandbox/"+id, nil, nil); err != nil {
		s.logger.Error("could not release a sandbox", "sandbox", id, "error", err)
	}
}

func (s *Sandbox) call(ctx context.Context, method, url string, body, into any) error {
	var payload io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("daytona: encode %s: %w", url, err)
		}
		payload = bytes.NewReader(encoded)
	}

	request, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return fmt.Errorf("daytona: %s: %w", url, err)
	}
	request.Header.Set("Authorization", "Bearer "+s.apiKey)
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}

	response, err := s.client.Do(request)
	if err != nil {
		return fmt.Errorf("daytona: %s: %w", url, err)
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, errorBodyLimit))
		return fmt.Errorf("daytona: %s: %s: %s", url, response.Status, strings.TrimSpace(string(detail)))
	}

	if into == nil {
		return nil
	}
	if err := json.NewDecoder(response.Body).Decode(into); err != nil {
		return fmt.Errorf("daytona: decode %s: %w", url, err)
	}
	return nil
}

type sandboxRequest struct {
	Language string `json:"language"`
}

type sandboxResponse struct {
	ID string `json:"id"`
}

type runRequest struct {
	Code     string `json:"code"`
	Language string `json:"language"`
	Timeout  int    `json:"timeout"`
}

type runResponse struct {
	// Result is everything the code printed, which Daytona returns as one stream.
	Result   string `json:"result"`
	ExitCode int    `json:"exitCode"`
}
