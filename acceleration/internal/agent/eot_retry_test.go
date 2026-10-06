package agent

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

func TestScoreEOTAttemptsRetriesTransientFailuresSequentially(t *testing.T) {
	for _, test := range []struct {
		name          string
		transientFail int
		probability   float64
		wantAttempts  int
	}{
		{name: "one transient then respond", transientFail: 1, probability: 0.9, wantAttempts: 2},
		{name: "two transient then low score", transientFail: 2, probability: 0.1, wantAttempts: 3},
	} {
		t.Run(test.name, func(t *testing.T) {
			pcm := make([]byte, eotMinSamples*2)
			for i := range pcm {
				pcm[i] = byte(i * 31)
			}
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				count := int(requests.Add(1))
				got, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read retry request body: %v", err)
					return
				}
				if !bytes.Equal(pcm, got) {
					t.Error("retry did not reuse the same immutable audio window")
					return
				}
				if r.Header.Get("X-Request-ID") == "" {
					t.Error("retry omitted the candidate request identifier")
					return
				}
				if count <= test.transientFail {
					http.Error(w, "temporary", http.StatusServiceUnavailable)
					return
				}
				writeEOTResponse(t, w, r.Header.Get("X-Request-ID"), len(got)/2, test.probability)
			}))
			t.Cleanup(server.Close)
			client, err := NewEOTClient(server.URL, "")
			require.NoError(t, err)
			ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
			defer cancel()

			score, scoreErr, attempts, class, elapsed, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", pcm, true)
			require.NoError(t, scoreErr)
			require.Equal(t, test.wantAttempts, attempts)
			require.Equal(t, int32(test.wantAttempts), requests.Load())
			require.Equal(t, eotFailureClass(""), class)
			require.Equal(t, test.probability, score.Probability)
			require.Less(t, elapsed, eotPrimaryLimit)
			require.False(t, budgetExhausted)
		})
	}
}

func TestScoreEOTAttemptsExhaustsOnlyThreeAttempts(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		http.Error(w, "temporary", http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)
	client, err := NewEOTClient(server.URL, "")
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
	defer cancel()

	_, scoreErr, attempts, class, elapsed, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
	require.Error(t, scoreErr)
	require.Equal(t, 3, attempts)
	require.Equal(t, int32(3), requests.Load())
	require.Equal(t, eotFailureTransientHTTP, class)
	require.False(t, budgetExhausted)
	require.Less(t, elapsed, eotPrimaryLimit+100*time.Millisecond)
}

func TestScoreEOTAttemptsHonorsRetryAfterAndTerminalHTTPResponse(t *testing.T) {
	for _, test := range []struct {
		name          string
		status        int
		retryAfter    string
		body          string
		wantAttempts  int
		wantClass     eotFailureClass
		wantElapsedMS time.Duration
	}{
		{name: "retry-after does not fit the remaining budget", status: http.StatusServiceUnavailable, retryAfter: "1", wantAttempts: 1, wantClass: eotFailureTransientHTTP, wantElapsedMS: 300 * time.Millisecond},
		{name: "permanent status is terminal", status: http.StatusBadRequest, wantAttempts: 1, wantClass: eotFailurePermanentHTTP, wantElapsedMS: 300 * time.Millisecond},
		{name: "unauthorized is terminal authentication failure", status: http.StatusUnauthorized, wantAttempts: 1, wantClass: eotFailureAuthentication, wantElapsedMS: 300 * time.Millisecond},
		{name: "forbidden is terminal authentication failure", status: http.StatusForbidden, wantAttempts: 1, wantClass: eotFailureAuthentication, wantElapsedMS: 300 * time.Millisecond},
		{name: "malformed successful response is terminal", status: http.StatusOK, body: "not json", wantAttempts: 1, wantClass: eotFailureInvalidResponse, wantElapsedMS: 300 * time.Millisecond},
	} {
		t.Run(test.name, func(t *testing.T) {
			var requests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				if test.retryAfter != "" {
					w.Header().Set("Retry-After", test.retryAfter)
				}
				w.WriteHeader(test.status)
				if test.body != "" {
					_, _ = io.WriteString(w, test.body)
				}
			}))
			t.Cleanup(server.Close)
			client, err := NewEOTClient(server.URL, "")
			require.NoError(t, err)
			ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
			defer cancel()

			_, scoreErr, attempts, class, elapsed, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
			require.Error(t, scoreErr)
			require.Equal(t, test.wantAttempts, attempts)
			require.Equal(t, int32(test.wantAttempts), requests.Load())
			require.Equal(t, test.wantClass, class)
			require.Less(t, elapsed, test.wantElapsedMS)
			require.False(t, budgetExhausted)
		})
	}
}

func TestScoreEOTAttemptsDistinguishesAttemptTimeoutFromTotalBudget(t *testing.T) {
	t.Run("attempt timeout retries", func(t *testing.T) {
		var requests atomic.Int32
		client, err := NewEOTClient("http://127.0.0.1:8000/v1/eot", "")
		require.NoError(t, err)
		client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
			count := requests.Add(1)
			pcm, readErr := io.ReadAll(request.Body)
			if readErr != nil {
				_ = request.Body.Close()
				return nil, readErr
			}
			_ = request.Body.Close()
			if count == 1 {
				<-request.Context().Done()
				return nil, request.Context().Err()
			}
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(eotJSON(request.Header.Get("X-Request-ID"), len(pcm)/2, 0.9))),
				Request:    request,
			}, nil
		})
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		score, scoreErr, attempts, class, elapsed, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
		require.NoError(t, scoreErr)
		require.Equal(t, 2, attempts)
		require.Equal(t, int32(2), requests.Load())
		require.Empty(t, class)
		require.False(t, budgetExhausted)
		require.GreaterOrEqual(t, elapsed, 450*time.Millisecond)
		require.Less(t, elapsed, 900*time.Millisecond)
		require.Equal(t, 0.9, score.Probability)
	})

	t.Run("primary total budget clips a longer parent deadline", func(t *testing.T) {
		var requests atomic.Int32
		client, err := NewEOTClient("http://127.0.0.1:8000/v1/eot", "")
		require.NoError(t, err)
		client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests.Add(1)
			_, readErr := io.Copy(io.Discard, request.Body)
			_ = request.Body.Close()
			if readErr != nil {
				return nil, readErr
			}
			<-request.Context().Done()
			return nil, request.Context().Err()
		})
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		_, scoreErr, attempts, class, elapsed, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
		require.Error(t, scoreErr)
		require.Equal(t, 3, attempts)
		require.Equal(t, int32(3), requests.Load())
		require.Equal(t, eotFailureTimeout, class)
		require.True(t, budgetExhausted)
		require.GreaterOrEqual(t, elapsed, 900*time.Millisecond)
		require.Less(t, elapsed, 1250*time.Millisecond)
	})
}

func TestWaitEOTRetryStopsWhenCanceledAfterEnteringBackoff(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	waiting := make(chan struct{})
	result := make(chan bool, 1)
	go func() {
		result <- waitEOTRetry(ctx, time.Hour, func() { close(waiting) })
	}()
	select {
	case <-waiting:
	case <-time.After(time.Second):
		t.Fatal("retry wait did not start")
	}
	cancel()
	select {
	case continued := <-result:
		require.False(t, continued)
	case <-time.After(time.Second):
		t.Fatal("retry wait did not stop after cancellation")
	}
}

func TestParseEOTRetryAfter(t *testing.T) {
	now := time.Date(2026, time.October, 6, 10, 0, 0, 0, time.UTC)
	date := now.Add(1200 * time.Millisecond).Format(http.TimeFormat)
	for _, test := range []struct {
		value string
		want  time.Duration
		valid bool
	}{
		{value: "2", want: 2 * time.Second, valid: true},
		{value: date, want: time.Second, valid: true},
		{value: "999999999999999999999999999999999999", want: 24 * time.Hour, valid: true},
		{value: "yesterday", valid: false},
		{value: "-1", valid: false},
	} {
		got, valid := parseEOTRetryAfter(test.value, now)
		require.Equal(t, test.valid, valid)
		if valid {
			require.Equal(t, test.want, got)
		}
	}
}

func TestScoreEOTAttemptsDoesNotRetryAuthenticationTLSOrPermanentDNS(t *testing.T) {
	t.Run("authentication failure", func(t *testing.T) {
		var requests atomic.Int32
		server := httptest.NewTLSServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
			requests.Add(1)
		}))
		t.Cleanup(server.Close)
		client, err := NewEOTClientWithTokenSource(server.URL, oauth2TokenSourceFunc(func() (*oauth2.Token, error) {
			return nil, errors.New("sensitive token source detail")
		}))
		require.NoError(t, err)
		ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
		defer cancel()
		_, scoreErr, attempts, class, _, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
		require.Error(t, scoreErr)
		require.Equal(t, 1, attempts)
		require.Equal(t, eotFailureAuthentication, class)
		require.False(t, budgetExhausted)
		require.Zero(t, requests.Load())
		require.NotContains(t, scoreErr.Error(), "sensitive")
	})

	t.Run("TLS certificate failure", func(t *testing.T) {
		var requests atomic.Int32
		server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requests.Add(1)
			pcm, _ := io.ReadAll(r.Body)
			writeEOTResponse(t, w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.9)
		}))
		t.Cleanup(server.Close)
		client, err := NewEOTClientWithTokenSource(server.URL, oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "opaque"}))
		require.NoError(t, err)
		ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
		defer cancel()
		_, scoreErr, attempts, class, _, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
		require.Error(t, scoreErr)
		require.Equal(t, 1, attempts)
		require.Equal(t, eotFailureTLS, class)
		require.False(t, budgetExhausted)
		require.Zero(t, requests.Load())
	})

	t.Run("permanent DNS failure", func(t *testing.T) {
		client, err := NewEOTClient("https://eot.invalid/v1/eot", "")
		require.NoError(t, err)
		client.tokens = oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "opaque"})
		var requests atomic.Int32
		client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests.Add(1)
			_ = request.Body.Close()
			return nil, &url.Error{Op: "Post", URL: "https://eot.invalid/v1/eot", Err: &net.DNSError{Err: "no such host", Name: "eot.invalid", IsNotFound: true}}
		})
		ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
		defer cancel()
		_, scoreErr, attempts, class, _, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
		require.Error(t, scoreErr)
		require.Equal(t, 1, attempts)
		require.Equal(t, eotFailurePermanentDNS, class)
		require.False(t, budgetExhausted)
		require.Equal(t, int32(1), requests.Load())
	})
}

func TestScoreEOTAttemptsCancellationStopsRequestAndBackoffRetries(t *testing.T) {
	t.Run("during upload", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		started := make(chan struct{})
		var starts sync.Once
		var requests atomic.Int32
		client, err := NewEOTClient("http://127.0.0.1:8000/v1/eot", "")
		require.NoError(t, err)
		client.client.Transport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
			requests.Add(1)
			buffer := make([]byte, 32)
			_, _ = r.Body.Read(buffer)
			starts.Do(func() { close(started) })
			<-r.Context().Done()
			_ = r.Body.Close()
			return nil, r.Context().Err()
		})
		done := make(chan struct {
			err      error
			attempts int
			class    eotFailureClass
		}, 1)
		go func() {
			_, scoreErr, attempts, class, _, _ := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
			done <- struct {
				err      error
				attempts int
				class    eotFailureClass
			}{scoreErr, attempts, class}
		}()
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("request upload did not start")
		}
		cancel()
		select {
		case result := <-done:
			require.Error(t, result.err)
			require.Equal(t, 1, result.attempts)
			require.Equal(t, eotFailureCanceled, result.class)
		case <-time.After(time.Second):
			t.Fatal("canceled upload did not drain its request body")
		}
		require.Equal(t, int32(1), requests.Load())
	})

	t.Run("between attempts", func(t *testing.T) {
		responseClosed := make(chan struct{})
		var requests atomic.Int32
		client, err := NewEOTClient("http://127.0.0.1:8000/v1/eot", "")
		require.NoError(t, err)
		client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
			requests.Add(1)
			_, _ = io.Copy(io.Discard, request.Body)
			_ = request.Body.Close()
			response := &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Header:     make(http.Header),
				Body:       &signalCloseReadCloser{ReadCloser: io.NopCloser(strings.NewReader("temporary")), closed: responseClosed},
				Request:    request,
			}
			return response, nil
		})
		ctx, cancel := context.WithCancel(context.Background())
		done := make(chan struct {
			err      error
			attempts int
			class    eotFailureClass
		}, 1)
		go func() {
			_, scoreErr, attempts, class, _, _ := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
			done <- struct {
				err      error
				attempts int
				class    eotFailureClass
			}{scoreErr, attempts, class}
		}()
		select {
		case <-responseClosed:
		case <-time.After(time.Second):
			t.Fatal("first failed attempt did not finish and close its response")
		}
		cancel()
		select {
		case result := <-done:
			require.Error(t, result.err)
			require.Equal(t, 1, result.attempts)
			require.Equal(t, eotFailureCanceled, result.class)
		case <-time.After(time.Second):
			t.Fatal("cancellation between attempts did not stop the retry")
		}
		require.Equal(t, int32(1), requests.Load())
	})
}

func TestScoreEOTAttemptsRetriesRecoverableNetworkReset(t *testing.T) {
	var requests atomic.Int32
	client, err := NewEOTClient("http://127.0.0.1:8000/v1/eot", "")
	require.NoError(t, err)
	client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		count := requests.Add(1)
		pcm, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			_ = request.Body.Close()
			return nil, readErr
		}
		_ = request.Body.Close()
		if count == 1 {
			return nil, &url.Error{Op: "Post", URL: request.URL.String(), Err: syscall.ECONNRESET}
		}
		response := &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(eotJSON(request.Header.Get("X-Request-ID"), len(pcm)/2, 0.9))),
			Request:    request,
		}
		return response, nil
	})
	ctx, cancel := context.WithTimeout(context.Background(), eotPrimaryLimit)
	defer cancel()

	score, scoreErr, attempts, class, _, budgetExhausted := scoreEOTAttempts(ctx, client, "candidate", make([]byte, eotMinSamples*2), true)
	require.NoError(t, scoreErr)
	require.Equal(t, 2, attempts)
	require.Equal(t, int32(2), requests.Load())
	require.Equal(t, eotFailureClass(""), class)
	require.False(t, budgetExhausted)
	require.Equal(t, 0.9, score.Probability)
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

type signalCloseReadCloser struct {
	io.ReadCloser
	closed chan struct{}
	once   sync.Once
}

func (body *signalCloseReadCloser) Close() error {
	err := body.ReadCloser.Close()
	body.once.Do(func() { close(body.closed) })
	return err
}
