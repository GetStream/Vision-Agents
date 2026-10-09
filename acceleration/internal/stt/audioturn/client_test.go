package audioturn

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

func TestEOTClientPostsTheOwnedPCMWindow(t *testing.T) {
	pcm := make([]byte, MinSamples*2)
	for i := 0; i < len(pcm)/2; i++ {
		pcm[2*i], pcm[2*i+1] = byte(i), byte(i>>8)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/v1/eot", r.URL.Path)
		require.Equal(t, "audio/pcm;rate=16000;channels=1;format=s16le", r.Header.Get("Content-Type"))
		require.Equal(t, "candidate-1", r.Header.Get("X-Request-ID"))
		require.Empty(t, r.Header.Get("X-EOT-Pauses"))
		require.Empty(t, r.Header.Get("X-EOT-Elapsed-Seconds"))
		got, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.Equal(t, pcm, got)
		writeEOTResponse(t, w, "candidate-1", len(pcm)/2, 0.83)
	}))
	t.Cleanup(server.Close)

	client, err := NewClient(server.URL, "")
	require.NoError(t, err)
	score, err := client.Score(context.Background(), "candidate-1", pcm)
	require.NoError(t, err)
	require.Equal(t, len(pcm)/2, score.Samples)
	require.Equal(t, 0.83, score.Probability)
	// The client never reuses the request slice; it remains caller-owned after scoring.
	require.Equal(t, byte(1), pcm[2])
}

func TestEOTClientUsesTheIdentityTokenFile(t *testing.T) {
	tokenPath := filepath.Join(t.TempDir(), "id-token")
	require.NoError(t, os.WriteFile(tokenPath, []byte("opaque.jwt.value\n"), 0o600))
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer opaque.jwt.value", r.Header.Get("Authorization"))
		writeEOTResponse(t, w, "candidate-2", MinSamples, 0.5)
	}))
	t.Cleanup(server.Close)

	client, err := NewClient(server.URL, tokenPath)
	require.NoError(t, err)
	_, err = client.Score(context.Background(), "candidate-2", make([]byte, MinSamples*2))
	require.NoError(t, err)
}

func TestEOTClientTokenSourceIsCachedAndIndependentOfCallerCancellation(t *testing.T) {
	var sourceCalls atomic.Int32
	started := make(chan struct{})
	release := make(chan struct{})
	source := oauth2.TokenSource(oauth2TokenSourceFunc(func() (*oauth2.Token, error) {
		if sourceCalls.Add(1) == 1 {
			close(started)
			<-release
		}
		return &oauth2.Token{AccessToken: "cached.identity.token", TokenType: "Bearer"}, nil
	}))
	var requests atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer cached.identity.token", r.Header.Get("Authorization"))
		pcm, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		requests.Add(1)
		writeEOTResponse(t, w, r.Header.Get("X-Request-ID"), len(pcm)/2, 0.5)
	}))
	t.Cleanup(server.Close)

	client, err := NewClientWithTokenSource(server.URL, source)
	require.NoError(t, err)
	client.client.Transport = server.Client().Transport
	pcm := make([]byte, MinSamples*2)

	firstContext, cancelFirst := context.WithCancel(context.Background())
	firstDone := make(chan error, 1)
	go func() {
		_, err := client.Score(firstContext, "cancelled-candidate", pcm)
		firstDone <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("token source did not start")
	}
	cancelFirst()
	select {
	case err := <-firstDone:
		require.Error(t, err)
	case <-time.After(time.Second):
		t.Fatal("caller cancellation did not release the EOT request")
	}

	secondDone := make(chan error, 1)
	go func() {
		_, err := client.Score(context.Background(), "live-candidate", pcm)
		secondDone <- err
	}()
	close(release)
	select {
	case err := <-secondDone:
		require.NoError(t, err)
	case <-time.After(time.Second):
		t.Fatal("a later caller did not reuse the completed token refresh")
	}

	_, err = client.Score(context.Background(), "warm-candidate", pcm)
	require.NoError(t, err)
	require.EqualValues(t, 1, sourceCalls.Load(), "warm requests must reuse the cached source token")
	require.EqualValues(t, 2, requests.Load(), "only uncancelled requests reach the service")
}

func TestEOTClientTokenSourceRequiresHTTPS(t *testing.T) {
	require.Error(t, func() error {
		_, err := NewClientWithTokenSource("http://127.0.0.1:8080", oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "secret"}))
		return err
	}())
	require.Error(t, func() error {
		_, err := NewClientWithTokenSource("https://example.com", nil)
		return err
	}())
}

type oauth2TokenSourceFunc func() (*oauth2.Token, error)

func (source oauth2TokenSourceFunc) Token() (*oauth2.Token, error) { return source() }

func TestEOTClientRejectsUnsafeAndMalformedResponses(t *testing.T) {
	require.Error(t, func() error {
		_, err := NewClient("http://example.com", "")
		return err
	}())
	require.Error(t, func() error {
		_, err := NewClient("https://example.com/?token=secret", "")
		return err
	}())

	tests := []struct {
		name string
		body string
	}{
		{"missing probability", `{"request_id":"c","model":"audioturn-stack16k-blend","release":"c4497ce3ba47","wait_probability":0.5,"sample_rate":16000,"samples":320,"window_samples":320}`},
		{"wrong request id", eotJSON("other", 320, 0.5)},
		{"wrong model", strings.Replace(eotJSON("c", 320, 0.5), DefaultModel, "other", 1)},
		{"noncomplementary wait", strings.Replace(eotJSON("c", 320, 0.5), `"wait_probability":0.5`, `"wait_probability":0.8`, 1)},
		{"nonfinite score", strings.Replace(eotJSON("c", 320, 0.5), `"probability":0.5`, `"probability":NaN`, 1)},
		{"trailing data", eotJSON("c", 320, 0.5) + ` {}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			client, err := NewClient(server.URL, "")
			require.NoError(t, err)
			_, err = client.Score(context.Background(), "c", make([]byte, 640))
			require.Error(t, err)
		})
	}

	client, err := NewClient("http://127.0.0.1:1", "")
	require.NoError(t, err)
	_, err = client.Score(context.Background(), "c", make([]byte, 639))
	require.Error(t, err)
	_, err = client.Score(context.Background(), "c", make([]byte, eotMaxBody+2))
	require.Error(t, err)
}

func eotJSON(id string, samples int, probability float64) string {
	return fmt.Sprintf(`{"request_id":%q,"model":%q,"release":%q,"probability":%v,"wait_probability":%v,"sample_rate":16000,"samples":%d,"window_samples":%d}`,
		id, DefaultModel, Release, probability, 1-probability, samples, samples)
}

func writeEOTResponse(t *testing.T, w http.ResponseWriter, id string, samples int, probability float64) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if _, err := io.WriteString(w, eotJSON(id, samples, probability)); err != nil {
		t.Error(err)
	}
}
