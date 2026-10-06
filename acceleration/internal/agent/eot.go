package agent

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"google.golang.org/api/idtoken"
)

const (
	eotModel       = "audioturn-stack16k-blend"
	eotRelease     = "c4497ce3ba47"
	eotSampleRate  = 16000
	eotMaxSamples  = 16 * eotSampleRate
	eotMinSamples  = 320
	eotMaxBody     = eotMaxSamples * 2
	eotClientLimit = 5 * time.Second
	eotGateLimit   = 500 * time.Millisecond
)

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
		return EOTScore{}, errors.New("agent: EOT is disabled")
	}
	if requestID == "" || len(pcm)%2 != 0 || len(pcm) < eotMinSamples*2 || len(pcm) > eotMaxBody {
		return EOTScore{}, errors.New("agent: invalid EOT request window")
	}
	token, err := c.token(ctx)
	if err != nil {
		return EOTScore{}, err
	}
	body := newEOTRequestBody(pcm)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint.String(), body)
	if err != nil {
		_ = body.Close()
		return EOTScore{}, errors.New("agent: could not create EOT request")
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
		return EOTScore{}, errors.New("agent: EOT request failed")
	}
	// Close the response before joining the upload body: RoundTrippers may finish an
	// upload on another goroutine after the response arrives.
	defer func() { <-body.closed }()
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return EOTScore{}, fmt.Errorf("agent: EOT service returned status %d", response.StatusCode)
	}
	responseBytes, err := io.ReadAll(io.LimitReader(response.Body, 16*1024+1))
	if err != nil || len(responseBytes) > 16*1024 {
		return EOTScore{}, errors.New("agent: invalid EOT response body")
	}
	var decoded eotResponse
	decoder := json.NewDecoder(strings.NewReader(string(responseBytes)))
	if err := decoder.Decode(&decoded); err != nil {
		return EOTScore{}, errors.New("agent: invalid EOT response")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return EOTScore{}, errors.New("agent: trailing EOT response data")
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
		return EOTScore{}, errors.New("agent: EOT response did not match the request")
	}
	return EOTScore{Probability: *decoded.Probability, Samples: wantedSamples}, nil
}

func validProbability(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func (c *EOTClient) token(ctx context.Context) (string, error) {
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
