package audioturn

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
	"golang.org/x/oauth2"
	"google.golang.org/api/idtoken"
)

const (
	DefaultModel             = "audioturn-stack16k-blend"
	Release                  = "c4497ce3ba47"
	SampleRate               = 16000
	MaxSamples               = 16 * SampleRate
	MinSamples               = 320
	eotMaxBody               = MaxSamples * 2
	eotClientLimit           = 5 * time.Second
	GateLimit                = 500 * time.Millisecond
	PrimaryLimit             = time.Second
	eotPrimaryRetryLimit     = 2
	eotPrimaryRetryWindow    = 250 * time.Millisecond
	eotPrimaryMinRetryWindow = 50 * time.Millisecond
)

type FailureClass string

const (
	FailureCanceled        FailureClass = "canceled"
	FailureAuthentication  FailureClass = "authentication"
	FailureInvalidRequest  FailureClass = "invalid_request"
	FailureTimeout         FailureClass = "timeout"
	FailureNetwork         FailureClass = "network"
	FailureTLS             FailureClass = "tls"
	FailurePermanentDNS    FailureClass = "permanent_dns"
	FailureTransientHTTP   FailureClass = "transient_http"
	FailurePermanentHTTP   FailureClass = "permanent_http"
	FailureInvalidResponse FailureClass = "invalid_response"
	FailureUnknown         FailureClass = "unknown"
)

// eotAttemptError retains only safe retry metadata. It deliberately discards transport,
// response-body, token, and URL details before an error reaches logging or fallback code.
type eotAttemptError struct {
	class         FailureClass
	retryAfter    time.Duration
	hasRetryAfter bool
}

func (e *eotAttemptError) Error() string {
	if e == nil {
		return "agent: EOT request failed"
	}
	switch e.class {
	case FailureAuthentication:
		return "agent: EOT authentication failed"
	case FailureInvalidRequest:
		return "agent: invalid EOT request"
	case FailureCanceled:
		return "agent: EOT request canceled"
	case FailureTimeout:
		return "agent: EOT request timed out"
	case FailureTransientHTTP, FailureNetwork:
		return "agent: EOT service temporarily unavailable"
	case FailureTLS, FailurePermanentDNS, FailurePermanentHTTP:
		return "agent: EOT service unavailable"
	default:
		return "agent: invalid EOT response"
	}
}

func (e *eotAttemptError) retryable() bool {
	return e != nil && (e.class == FailureTimeout || e.class == FailureNetwork || e.class == FailureTransientHTTP)
}

// IsTransientError reports whether a failed EOT request is safe to retry. It exposes
// only the client's bounded classification, never transport details or response bodies.
func IsTransientError(err error) bool {
	var failure *eotAttemptError
	return errors.As(err, &failure) && failure.retryable()
}

// Client scores an audio window and optionally transcribes it in the same request.
type Client struct {
	endpoint  *url.URL
	audience  string
	tokenFile string
	client    *http.Client
	anonymous bool

	tokenMu sync.Mutex
	tokens  oauth2.TokenSource
	cached  *oauth2.Token
	flight  *tokenFlight
}

type tokenResult struct {
	token string
	err   error
}

type tokenFlight struct {
	done   chan struct{}
	result tokenResult
}

// Score is the service's raw endpoint probability and the audio window it scored.
type Score struct {
	Probability float64
	Samples     int
}

// Transcript describes one audio window, not a delta from an earlier request.
// Word timestamps are milliseconds relative to the end of that window.
type Transcript struct {
	Text  string `json:"text"`
	Words []Word `json:"words"`
}

type Word struct {
	Text       string  `json:"text"`
	StartMS    int     `json:"start_ms"`
	EndMS      int     `json:"end_ms"`
	Confidence float64 `json:"p"`
}

type eotResponse struct {
	RequestID     string   `json:"request_id"`
	Model         string   `json:"model"`
	Release       string   `json:"release"`
	Probability   *float64 `json:"probability"`
	Wait          *float64 `json:"wait_probability"`
	SampleRate    int      `json:"sample_rate"`
	Samples       int      `json:"samples"`
	WindowSamples int      `json:"window_samples"`
}

type eotRequestBody struct {
	mu     sync.Mutex
	reader *bytes.Reader
	closed chan struct{}
	done   bool
}

func newEOTRequestBody(pcm []byte) *eotRequestBody {
	return &eotRequestBody{reader: bytes.NewReader(pcm), closed: make(chan struct{})}
}

func (b *eotRequestBody) Read(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.done {
		return 0, io.EOF
	}
	return b.reader.Read(p)
}

func (b *eotRequestBody) Close() error {
	b.mu.Lock()
	if !b.done {
		b.done = true
		close(b.closed)
	}
	b.mu.Unlock()
	return nil
}

// NewClient builds a reusable client. endpoint may be the service base URL or
// its /v1/eot URL. Plain HTTP is accepted only for loopback development servers.
func NewClient(endpoint, tokenFile string) (*Client, error) {
	if eotdefaults.IsHostedDemoOrigin(endpoint) {
		return nil, errors.New("agent: use NewHostedClient for the hosted demo endpoint")
	}
	return newEOTClient(endpoint, tokenFile)
}

func newEOTClient(endpoint, tokenFile string) (*Client, error) {
	parsed, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil || parsed == nil || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" {
		return nil, errors.New("agent: invalid EOT endpoint")
	}
	if parsed.Scheme != "https" && parsed.Scheme != "http" {
		return nil, errors.New("agent: EOT endpoint must use HTTPS")
	}
	if parsed.Scheme == "http" && !isLoopbackHost(parsed.Hostname()) {
		return nil, errors.New("agent: EOT HTTP endpoint must be loopback")
	}
	if parsed.Path == "" || parsed.Path == "/" {
		parsed.Path = "/v1/eot"
	} else if parsed.Path != "/v1/eot" {
		return nil, errors.New("agent: EOT endpoint path must be /v1/eot")
	}
	if parsed.RawPath != "" {
		return nil, errors.New("agent: invalid EOT endpoint path")
	}

	origin := &url.URL{Scheme: parsed.Scheme, Host: parsed.Host}
	client := &http.Client{
		Timeout: eotClientLimit,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	return &Client{
		endpoint:  parsed,
		audience:  origin.String(),
		tokenFile: strings.TrimSpace(tokenFile),
		client:    client,
	}, nil
}

// NewHostedClient builds the fixed hosted demo client. It sends no credentials
// and never probes ADC; callers that need an authenticated private scorer should use
// NewClient with that private endpoint and its existing credential configuration.
func NewHostedClient() (*Client, error) {
	client, err := newEOTClient(eotdefaults.HostedDemoEndpoint, "")
	if err != nil {
		return nil, err
	}
	client.anonymous = true
	return client, nil
}

// NewClientWithTokenSource builds a client that obtains bearer credentials from the
// supplied reusable source. Token sources are accepted only for HTTPS endpoints so a
// caller cannot accidentally send a bearer token over cleartext HTTP.
func NewClientWithTokenSource(endpoint string, source oauth2.TokenSource) (*Client, error) {
	if source == nil {
		return nil, errors.New("agent: EOT token source is required")
	}
	client, err := NewClient(endpoint, "")
	if err != nil {
		return nil, err
	}
	if client.endpoint.Scheme != "https" {
		return nil, errors.New("agent: EOT token sources require HTTPS")
	}
	client.tokens = source
	return client, nil
}

func isLoopbackHost(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Score sends an owned PCM16LE mono window. The caller must not mutate pcm until this
// method returns; it waits for net/http to close the request body before returning.
func (c *Client) Score(ctx context.Context, requestID string, pcm []byte) (Score, error) {
	score, _, err := c.score(ctx, requestID, pcm, false)
	return score, err
}

// Transcribe requests both the turn score and the words of the same window. It
// requires transcript support on the server; decision-only clients still use Score.
func (c *Client) Transcribe(ctx context.Context, requestID string, pcm []byte) (Score, *Transcript, error) {
	return c.score(ctx, requestID, pcm, true)
}

func (c *Client) score(ctx context.Context, requestID string, pcm []byte, transcribe bool) (Score, *Transcript, error) {
	if c == nil {
		return Score{}, nil, &eotAttemptError{class: FailureInvalidRequest}
	}
	if requestID == "" || len(pcm)%2 != 0 || len(pcm) < MinSamples*2 || len(pcm) > eotMaxBody {
		return Score{}, nil, &eotAttemptError{class: FailureInvalidRequest}
	}
	token, err := c.token(ctx)
	if err != nil {
		return Score{}, nil, &eotAttemptError{class: FailureAuthentication}
	}
	body := newEOTRequestBody(pcm)
	endpoint := *c.endpoint
	if transcribe {
		endpoint.RawQuery = "transcript=true&transcript_min_p=0"
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), body)
	if err != nil {
		_ = body.Close()
		return Score{}, nil, &eotAttemptError{class: FailureInvalidRequest}
	}
	request.ContentLength = int64(len(pcm))
	request.Header.Set("Content-Type", "audio/pcm;rate=16000;channels=1;format=s16le")
	request.Header.Set("X-Request-ID", requestID)
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := c.client.Do(request)
	if err != nil {
		// RoundTrippers may close request bodies asynchronously after returning an error.
		<-body.closed
		return Score{}, nil, &eotAttemptError{class: classifyEOTTransportError(err)}
	}
	// Close the response before joining the upload body: RoundTrippers may finish an
	// upload on another goroutine after the response arrives.
	defer func() { <-body.closed }()
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		class := FailurePermanentHTTP
		switch response.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			class = FailureAuthentication
		case http.StatusRequestTimeout, http.StatusTooManyRequests,
			http.StatusInternalServerError, http.StatusBadGateway,
			http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			class = FailureTransientHTTP
		}
		failure := &eotAttemptError{class: class}
		if response.StatusCode == http.StatusTooManyRequests || response.StatusCode == http.StatusServiceUnavailable {
			failure.retryAfter, failure.hasRetryAfter = parseEOTRetryAfter(response.Header.Get("Retry-After"), time.Now())
		}
		return Score{}, nil, failure
	}
	limit := int64(16 * 1024)
	if transcribe {
		limit = 64 * 1024
	}
	responseBytes, err := io.ReadAll(io.LimitReader(response.Body, limit+1))
	if err != nil {
		return Score{}, nil, &eotAttemptError{class: classifyEOTTransportError(err)}
	}
	if int64(len(responseBytes)) > limit {
		return Score{}, nil, &eotAttemptError{class: FailureInvalidResponse}
	}
	decoder := json.NewDecoder(bytes.NewReader(responseBytes))
	var decoded eotResponse
	if err := decoder.Decode(&decoded); err != nil {
		return Score{}, nil, &eotAttemptError{class: FailureInvalidResponse}
	}
	wantedSamples := len(pcm) / 2
	if decoded.RequestID != requestID || decoded.Model != DefaultModel || decoded.Release != Release ||
		decoded.Probability == nil || decoded.Wait == nil ||
		decoded.SampleRate != SampleRate || decoded.Samples != wantedSamples || decoded.WindowSamples != wantedSamples ||
		!validProbability(*decoded.Probability) || !validProbability(*decoded.Wait) ||
		math.Abs(*decoded.Wait-(1-*decoded.Probability)) > 1e-6 {
		return Score{}, nil, &eotAttemptError{class: FailureInvalidResponse}
	}
	var transcript *Transcript
	if transcribe {
		var result struct {
			Transcript *Transcript `json:"transcript"`
		}
		if err := decoder.Decode(&result); err != nil || result.Transcript == nil {
			return Score{}, nil, errors.New("agent: EOT transcript unavailable")
		}
		transcript = result.Transcript
		previous := -(wantedSamples*1000 + SampleRate - 1) / SampleRate
		for i := range transcript.Words {
			word := &transcript.Words[i]
			if strings.TrimSpace(word.Text) == "" || word.StartMS < previous || word.EndMS < word.StartMS ||
				word.StartMS > 0 || !validProbability(word.Confidence) {
				return Score{}, nil, &eotAttemptError{class: FailureInvalidResponse}
			}
			// Decoder durations can extend past a partial window's last audio frame.
			word.EndMS = min(word.EndMS, 0)
			previous = word.StartMS
		}
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return Score{}, nil, &eotAttemptError{class: FailureInvalidResponse}
	}
	return Score{Probability: *decoded.Probability, Samples: wantedSamples}, transcript, nil
}

func validProbability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func classifyEOTTransportError(err error) FailureClass {
	if err == nil {
		return FailureUnknown
	}
	if errors.Is(err, context.Canceled) {
		return FailureCanceled
	}
	var verifyErr *tls.CertificateVerificationError
	var unknownAuthority x509.UnknownAuthorityError
	var hostname x509.HostnameError
	var invalidCertificate x509.CertificateInvalidError
	var recordErr tls.RecordHeaderError
	if errors.As(err, &verifyErr) || errors.As(err, &unknownAuthority) ||
		errors.As(err, &hostname) || errors.As(err, &invalidCertificate) || errors.As(err, &recordErr) {
		return FailureTLS
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		if dnsErr.IsTimeout || dnsErr.IsTemporary {
			return FailureNetwork
		}
		return FailurePermanentDNS
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return FailureTimeout
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNABORTED) || errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.ETIMEDOUT) {
		return FailureNetwork
	}
	var networkErr net.Error
	if errors.As(err, &networkErr) && (networkErr.Timeout() || networkErr.Temporary()) {
		return FailureNetwork
	}
	return FailureUnknown
}

func parseEOTRetryAfter(value string, now time.Time) (time.Duration, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, false
	}
	digitsOnly := true
	for i := 0; i < len(value); i++ {
		if value[i] < '0' || value[i] > '9' {
			digitsOnly = false
			break
		}
	}
	if digitsOnly {
		seconds, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return 24 * time.Hour, true
		}
		return time.Duration(min(seconds, 86400)) * time.Second, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	return min(max(when.Sub(now), 0), 24*time.Hour), true
}

func (c *Client) token(ctx context.Context) (string, error) {
	if c.anonymous {
		return "", nil
	}
	if c.tokenFile != "" {
		file, err := os.Open(c.tokenFile)
		if err != nil {
			return "", errors.New("agent: EOT identity token is unavailable")
		}
		defer file.Close()
		raw, err := io.ReadAll(io.LimitReader(file, 16*1024+1))
		if err != nil || len(raw) == 0 || len(raw) > 16*1024 {
			return "", errors.New("agent: EOT identity token file is invalid")
		}
		token := strings.TrimSpace(string(raw))
		if token == "" || strings.ContainsAny(token, "\r\n \t") {
			return "", errors.New("agent: EOT identity token file is invalid")
		}
		return token, nil
	}
	if c.endpoint.Scheme == "http" {
		return "", nil
	}
	if err := ctx.Err(); err != nil {
		return "", errors.New("agent: EOT identity token request timed out")
	}
	c.tokenMu.Lock()
	if c.cached != nil && c.cached.Valid() {
		token := c.cached.AccessToken
		c.tokenMu.Unlock()
		return token, nil
	}
	if c.flight == nil {
		flight := &tokenFlight{done: make(chan struct{})}
		c.flight = flight
		source := c.tokens
		go func() {
			authClient := &http.Client{Timeout: eotClientLimit}
			sourceCtx := context.WithValue(context.Background(), oauth2.HTTPClient, authClient)
			var err error
			if source == nil {
				source, err = idtoken.NewTokenSource(sourceCtx, c.audience)
			}
			var token *oauth2.Token
			if err == nil {
				token, err = source.Token()
			}
			result := tokenResult{err: err}
			if err == nil && token != nil {
				result.token = token.AccessToken
			}
			if result.err == nil && result.token == "" {
				result.err = errors.New("empty token")
			}
			c.tokenMu.Lock()
			c.tokens = source
			if result.err == nil && token != nil {
				copy := *token
				c.cached = &copy
			}
			flight.result = result
			close(flight.done)
			c.flight = nil
			c.tokenMu.Unlock()
		}()
	}
	flight := c.flight
	c.tokenMu.Unlock()
	select {
	case <-flight.done:
		result := flight.result
		if result.err != nil {
			return "", errors.New("agent: EOT identity token refresh failed")
		}
		return result.token, nil
	case <-ctx.Done():
		return "", errors.New("agent: EOT identity token request timed out")
	}
}
