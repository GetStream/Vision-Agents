package agent

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEOTClientPostsTheOwnedPCMWindow(t *testing.T) {
	pcm := make([]byte, eotMinSamples*2)
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

	client, err := NewEOTClient(server.URL, "")
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
		writeEOTResponse(t, w, "candidate-2", eotMinSamples, 0.5)
	}))
	t.Cleanup(server.Close)

	client, err := NewEOTClient(server.URL, tokenPath)
	require.NoError(t, err)
	_, err = client.Score(context.Background(), "candidate-2", make([]byte, eotMinSamples*2))
	require.NoError(t, err)
}

func TestEOTClientRejectsUnsafeAndMalformedResponses(t *testing.T) {
	require.Error(t, func() error {
		_, err := NewEOTClient("http://example.com", "")
		return err
	}())
	require.Error(t, func() error {
		_, err := NewEOTClient("https://example.com/?token=secret", "")
		return err
	}())

	tests := []struct {
		name string
		body string
	}{
		{"missing probability", `{"request_id":"c","model":"audioturn-stack16k-blend","release":"c4497ce3ba47","wait_probability":0.5,"sample_rate":16000,"samples":320,"window_samples":320}`},
		{"wrong request id", eotJSON("other", 320, 0.5)},
		{"wrong model", strings.Replace(eotJSON("c", 320, 0.5), eotModel, "other", 1)},
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
			client, err := NewEOTClient(server.URL, "")
			require.NoError(t, err)
			_, err = client.Score(context.Background(), "c", make([]byte, 640))
			require.Error(t, err)
		})
	}

	client, err := NewEOTClient("http://127.0.0.1:1", "")
	require.NoError(t, err)
	_, err = client.Score(context.Background(), "c", make([]byte, 639))
	require.Error(t, err)
	_, err = client.Score(context.Background(), "c", make([]byte, eotMaxBody+2))
	require.Error(t, err)
}

func TestPCM16LERingWrapsChronologicallyAndCopiesSnapshots(t *testing.T) {
	ring := &pcm16leRing{sample: make([]int16, eotMinSamples)}
	first := make([]int16, eotMinSamples)
	for i := range first {
		first[i] = int16(i)
	}
	ring.append(first)
	second := []int16{1000, 1001}
	ring.append(second)
	snapshot := ring.snapshot()
	require.Len(t, snapshot, eotMinSamples*2)
	require.Equal(t, int16(2), int16(uint16(snapshot[0])|uint16(snapshot[1])<<8))
	require.Equal(t, int16(1000), int16(uint16(snapshot[len(snapshot)-4])|uint16(snapshot[len(snapshot)-3])<<8))
	require.Equal(t, int16(1001), int16(uint16(snapshot[len(snapshot)-2])|uint16(snapshot[len(snapshot)-1])<<8))
	before := append([]byte(nil), snapshot...)
	ring.append([]int16{2000})
	require.Equal(t, before, snapshot)
	ring.clear()
	require.Nil(t, ring.snapshot())
}

func eotJSON(id string, samples int, probability float64) string {
	return fmt.Sprintf(`{"request_id":%q,"model":%q,"release":%q,"probability":%v,"wait_probability":%v,"sample_rate":16000,"samples":%d,"window_samples":%d}`,
		id, eotModel, eotRelease, probability, 1-probability, samples, samples)
}

func writeEOTResponse(t *testing.T, w http.ResponseWriter, id string, samples int, probability float64) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if _, err := io.WriteString(w, eotJSON(id, samples, probability)); err != nil {
		t.Error(err)
	}
}
