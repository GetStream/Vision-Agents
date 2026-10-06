package agent

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
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
	eotModel                 = "audioturn-stack16k-blend"
	eotRelease               = "c4497ce3ba47"
	eotSampleRate            = 16000
	eotMaxSamples            = 16 * eotSampleRate
	eotMinSamples            = 320
	eotMaxBody               = eotMaxSamples * 2
	eotClientLimit           = 5 * time.Second
	eotGateLimit             = 500 * time.Millisecond
	eotPrimaryLimit          = time.Second
	eotPrimaryRetryLimit     = 2
	eotPrimaryRetryWindow    = 250 * time.Millisecond
	eotPrimaryMinRetryWindow = 50 * time.Millisecond
)

type eotFailureClass string

const (
	eotFailureCanceled        eotFailureClass = "canceled"
	eotFailureAuthentication  eotFailureClass = "authentication"
	eotFailureInvalidRequest  eotFailureClass = "invalid_request"
	eotFailureTimeout         eotFailureClass = "timeout"
	eotFailureNetwork         eotFailureClass = "network"
	eotFailureTLS             eotFailureClass = "tls"
	eotFailurePermanentDNS    eotFailureClass = "permanent_dns"
	eotFailureTransientHTTP   eotFailureClass = "transient_http"
	eotFailurePermanentHTTP   eotFailureClass = "permanent_http"
	eotFailureInvalidResponse eotFailureClass = "invalid_response"
	eotFailureUnknown         eotFailureClass = "unknown"
)

// eotAttemptError retains only safe retry metadata. It deliberately discards transport,
// response-body, token, and URL details before an error reaches logging or fallback code.
type eotAttemptError struct {
	class         eotFailureClass
	retryAfter    time.Duration
	hasRetryAfter bool
}

func (e *eotAttemptError) Error() string {
	if e == nil {
		return "agent: EOT request failed"
	}
	switch e.class {
	case eotFailureAuthentication:
		return "agent: EOT authentication failed"
	case eotFailureInvalidRequest:
		return "agent: invalid EOT request"
	case eotFailureCanceled:
		return "agent: EOT request canceled"
	case eotFailureTimeout:
		return "agent: EOT request timed out"
	case eotFailureTransientHTTP, eotFailureNetwork:
		return "agent: EOT service temporarily unavailable"
	case eotFailureTLS, eotFailurePermanentDNS, eotFailurePermanentHTTP:
		return "agent: EOT service unavailable"
	default:
		return "agent: invalid EOT response"
	}
}

func (e *eotAttemptError) retryable() bool {
	return e != nil && (e.class == eotFailureTimeout || e.class == eotFailureNetwork || e.class == eotFailureTransientHTTP)
}

func eotErrorMetadata(err error) (eotFailureClass, time.Duration, bool) {
	var failure *eotAttemptError
	if errors.As(err, &failure) {
		return failure.class, failure.retryAfter, failure.hasRetryAfter
	}
	return eotFailureUnknown, 0, false
}

// IsTransientEOTError reports whether a failed EOT request is safe to retry. It exposes
// only the client's bounded classification, never transport details or response bodies.
func IsTransientEOTError(err error) bool {
	var failure *eotAttemptError
	return errors.As(err, &failure) && failure.retryable()
}

// EOTMode determines whether an acoustic score gates the semantic flow controller or
// answers eligible quiet-floor completion candidates directly.
type EOTMode string

const (
	EOTModeGate    EOTMode = "gate"
	EOTModePrimary EOTMode = "primary"
)

func (mode EOTMode) valid() bool { return mode == EOTModeGate || mode == EOTModePrimary }

// EOTClient asks the optional acoustic service whether a settled voice candidate has
// reached an endpoint. The configured EOT mode decides whether the score gates or resolves
// eligible quiet-floor candidates.
type EOTClient struct {
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

// EOTScore is the service's raw endpoint probability and the audio window it scored.
type EOTScore struct {
	Probability float64
	Samples     int
}

type eotResponse struct {
	RequestID     *string  `json:"request_id"`
	Model         *string  `json:"model"`
	Release       *string  `json:"release"`
	Probability   *float64 `json:"probability"`
	Wait          *float64 `json:"wait_probability"`
	SampleRate    *int     `json:"sample_rate"`
	Samples       *int     `json:"samples"`
	WindowSamples *int     `json:"window_samples"`
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

// NewEOTClient builds a reusable scalar client. endpoint may be the service base URL or
// its /v1/eot URL. Plain HTTP is accepted only for loopback development servers.
func NewEOTClient(endpoint, tokenFile string) (*EOTClient, error) {
	if eotdefaults.IsHostedDemoOrigin(endpoint) {
		return nil, errors.New("agent: use NewHostedDemoEOTClient for the hosted demo endpoint")
	}
	return newEOTClient(endpoint, tokenFile)
}

func newEOTClient(endpoint, tokenFile string) (*EOTClient, error) {
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
	return &EOTClient{
		endpoint:  parsed,
		audience:  origin.String(),
		tokenFile: strings.TrimSpace(tokenFile),
		client:    client,
	}, nil
}

// NewHostedDemoEOTClient builds the fixed hosted demo client. It sends no credentials
// and never probes ADC; callers that need an authenticated private scorer should use
// NewEOTClient with that private endpoint and its existing credential configuration.
func NewHostedDemoEOTClient() (*EOTClient, error) {
	client, err := newEOTClient(eotdefaults.HostedDemoEndpoint, "")
	if err != nil {
		return nil, err
	}
	client.anonymous = true
	return client, nil
}

// NewEOTClientWithTokenSource builds a client that obtains bearer credentials from the
// supplied reusable source. Token sources are accepted only for HTTPS endpoints so a
// caller cannot accidentally send a bearer token over cleartext HTTP.
func NewEOTClientWithTokenSource(endpoint string, source oauth2.TokenSource) (*EOTClient, error) {
	if source == nil {
		return nil, errors.New("agent: EOT token source is required")
	}
	client, err := NewEOTClient(endpoint, "")
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
func (c *EOTClient) Score(ctx context.Context, requestID string, pcm []byte) (EOTScore, error) {
	if c == nil {
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidRequest}
	}
	if requestID == "" || len(pcm)%2 != 0 || len(pcm) < eotMinSamples*2 || len(pcm) > eotMaxBody {
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidRequest}
	}
	token, err := c.token(ctx)
	if err != nil {
		return EOTScore{}, &eotAttemptError{class: eotFailureAuthentication}
	}
	body := newEOTRequestBody(pcm)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint.String(), body)
	if err != nil {
		_ = body.Close()
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidRequest}
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
		return EOTScore{}, &eotAttemptError{class: classifyEOTTransportError(err)}
	}
	// Close the response before joining the upload body: RoundTrippers may finish an
	// upload on another goroutine after the response arrives.
	defer func() { <-body.closed }()
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		class := eotFailurePermanentHTTP
		switch response.StatusCode {
		case http.StatusUnauthorized, http.StatusForbidden:
			class = eotFailureAuthentication
		case http.StatusRequestTimeout, http.StatusTooManyRequests,
			http.StatusInternalServerError, http.StatusBadGateway,
			http.StatusServiceUnavailable, http.StatusGatewayTimeout:
			class = eotFailureTransientHTTP
		}
		failure := &eotAttemptError{class: class}
		if response.StatusCode == http.StatusTooManyRequests || response.StatusCode == http.StatusServiceUnavailable {
			failure.retryAfter, failure.hasRetryAfter = parseEOTRetryAfter(response.Header.Get("Retry-After"), time.Now())
		}
		return EOTScore{}, failure
	}
	responseBytes, err := io.ReadAll(io.LimitReader(response.Body, 16*1024+1))
	if err != nil {
		return EOTScore{}, &eotAttemptError{class: classifyEOTTransportError(err)}
	}
	if len(responseBytes) > 16*1024 {
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidResponse}
	}
	var decoded eotResponse
	decoder := json.NewDecoder(strings.NewReader(string(responseBytes)))
	if err := decoder.Decode(&decoded); err != nil {
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidResponse}
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidResponse}
	}
	wantedSamples := len(pcm) / 2
	if decoded.RequestID == nil || *decoded.RequestID != requestID ||
		decoded.Model == nil || *decoded.Model != eotModel ||
		decoded.Release == nil || *decoded.Release != eotRelease ||
		decoded.Probability == nil || decoded.Wait == nil ||
		decoded.SampleRate == nil || *decoded.SampleRate != eotSampleRate ||
		decoded.Samples == nil || *decoded.Samples != wantedSamples ||
		decoded.WindowSamples == nil || *decoded.WindowSamples != wantedSamples ||
		!validProbability(*decoded.Probability) || !validProbability(*decoded.Wait) ||
		math.Abs(*decoded.Wait-(1-*decoded.Probability)) > 1e-6 {
		return EOTScore{}, &eotAttemptError{class: eotFailureInvalidResponse}
	}
	return EOTScore{Probability: *decoded.Probability, Samples: wantedSamples}, nil
}

func validProbability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func classifyEOTTransportError(err error) eotFailureClass {
	if err == nil {
		return eotFailureUnknown
	}
	if errors.Is(err, context.Canceled) {
		return eotFailureCanceled
	}
	var verifyErr *tls.CertificateVerificationError
	var unknownAuthority x509.UnknownAuthorityError
	var hostname x509.HostnameError
	var invalidCertificate x509.CertificateInvalidError
	var recordErr tls.RecordHeaderError
	if errors.As(err, &verifyErr) || errors.As(err, &unknownAuthority) ||
		errors.As(err, &hostname) || errors.As(err, &invalidCertificate) || errors.As(err, &recordErr) {
		return eotFailureTLS
	}
	var dnsErr *net.DNSError
	if errors.As(err, &dnsErr) {
		if dnsErr.IsTimeout || dnsErr.IsTemporary {
			return eotFailureNetwork
		}
		return eotFailurePermanentDNS
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return eotFailureTimeout
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNABORTED) || errors.Is(err, syscall.EPIPE) ||
		errors.Is(err, syscall.ETIMEDOUT) {
		return eotFailureNetwork
	}
	var networkErr net.Error
	if errors.As(err, &networkErr) && (networkErr.Timeout() || networkErr.Temporary()) {
		return eotFailureNetwork
	}
	return eotFailureUnknown
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
		if seconds < 0 {
			return 0, false
		}
		if seconds > int64((24*time.Hour)/time.Second) {
			return 24 * time.Hour, true
		}
		return time.Duration(seconds) * time.Second, true
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0, false
	}
	delay := when.Sub(now)
	if delay < 0 {
		delay = 0
	}
	if delay > 24*time.Hour {
		delay = 24 * time.Hour
	}
	return delay, true
}

func (c *EOTClient) token(ctx context.Context) (string, error) {
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
		go func() {
			authClient := &http.Client{Timeout: eotClientLimit}
			sourceCtx := context.WithValue(context.Background(), oauth2.HTTPClient, authClient)
			c.tokenMu.Lock()
			source := c.tokens
			c.tokenMu.Unlock()
			var err error
			if source == nil {
				source, err = idtoken.NewTokenSource(sourceCtx, c.audience)
				if err == nil {
					c.tokenMu.Lock()
					if c.tokens == nil {
						c.tokens = source
					} else {
						source = c.tokens
					}
					c.tokenMu.Unlock()
				}
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
			if result.err == nil && token != nil {
				copy := *token
				c.cached = &copy
			}
			flight.result = result
			close(flight.done)
			if c.flight == flight {
				c.flight = nil
			}
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

// pcm16leRing holds a participant's most recent bounded audio without sharing mutable
// storage with an in-flight request.
type pcm16leRing struct {
	mu     sync.Mutex
	sample []int16
	next   int
	full   bool
}

func newPCM16LERing() *pcm16leRing {
	return &pcm16leRing{sample: make([]int16, eotMaxSamples)}
}

func (r *pcm16leRing) append(samples []int16) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, sample := range samples {
		r.sample[r.next] = sample
		r.next++
		if r.next == len(r.sample) {
			r.next = 0
			r.full = true
		}
	}
}

func (r *pcm16leRing) snapshot() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	count := r.next
	start := 0
	if r.full {
		count = len(r.sample)
		start = r.next
	}
	if count < eotMinSamples {
		return nil
	}
	pcm := make([]byte, count*2)
	for i := 0; i < count; i++ {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(r.sample[(start+i)%len(r.sample)]))
	}
	return pcm
}

func (r *pcm16leRing) clear() {
	r.mu.Lock()
	clear(r.sample)
	r.next, r.full = 0, false
	r.mu.Unlock()
}
