package audioturn

import (
	"context"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eotdefaults"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

const hostedEOTChildVar = "VISION_AGENTS_HOSTED_EOT_CHILD"

func TestEOTConstructorsRejectCredentialsForHostedOrigin(t *testing.T) {
	for _, endpoint := range []string{
		eotdefaults.HostedDemoEndpoint,
		"https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app",
		"https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app:443/v1/eot",
		"https://audioturn-demo-eu-5gdhza7snq-ez.a.run.app./v1/eot",
	} {
		_, err := NewClient(endpoint, "")
		require.ErrorContains(t, err, "NewHostedClient", endpoint)
		_, err = NewClient(endpoint, "/run/secrets/eot-token")
		require.ErrorContains(t, err, "NewHostedClient", endpoint)
		_, err = NewClientWithTokenSource(endpoint, oauth2TokenSourceFunc(func() (*oauth2.Token, error) {
			return &oauth2.Token{AccessToken: "must-not-be-used"}, nil
		}))
		require.ErrorContains(t, err, "NewHostedClient", endpoint)
	}
}

func TestHostedDemoEOTClientWorksInFreshProcessWithoutCredentials(t *testing.T) {
	if os.Getenv(hostedEOTChildVar) == "1" {
		t.Run("child", testHostedDemoEOTAnonymousScore)
		return
	}

	home := t.TempDir()
	command := exec.Command(os.Args[0], "-test.run=^TestHostedDemoEOTClientWorksInFreshProcessWithoutCredentials$")
	command.Env = cleanHostedEOTEnvironment(os.Environ(), hostedEOTChildVar+"=1", "HOME="+home, "PATH=")
	output, err := command.CombinedOutput()
	require.NoErrorf(t, err, "isolated client process failed: %s", output)
}

func cleanHostedEOTEnvironment(environment []string, replacements ...string) []string {
	clean := make([]string, 0, len(environment)+len(replacements))
	for _, entry := range environment {
		key, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(key, "ROUTER_EOT_") || strings.HasPrefix(key, "GOOGLE_") ||
			strings.HasPrefix(key, "CLOUDSDK_") || strings.HasPrefix(key, "GCLOUD_") ||
			strings.HasPrefix(key, "GCE_METADATA_") || key == "HOME" ||
			key == "PATH" || key == hostedEOTChildVar {
			continue
		}
		clean = append(clean, entry)
	}
	return append(clean, replacements...)
}

func testHostedDemoEOTAnonymousScore(t *testing.T) {
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if strings.HasPrefix(key, "ROUTER_EOT_") || strings.HasPrefix(key, "GOOGLE_") ||
			strings.HasPrefix(key, "CLOUDSDK_") || strings.HasPrefix(key, "GCLOUD_") ||
			strings.HasPrefix(key, "GCE_METADATA_") {
			t.Fatalf("fresh process unexpectedly contains credential/config environment key %q", key)
		}
	}
	if path := os.Getenv("PATH"); path != "" {
		t.Fatalf("gcloud should be unavailable in the fresh process, PATH=%q", path)
	}

	client, err := NewHostedClient()
	require.NoError(t, err)
	var requests int
	client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests++
		if request.URL.String() != eotdefaults.HostedDemoEndpoint {
			t.Errorf("request endpoint = %q, want fixed hosted endpoint", request.URL)
		}
		if authorization := request.Header.Get("Authorization"); authorization != "" {
			t.Errorf("hosted request unexpectedly carried Authorization")
		}
		pcm, readErr := io.ReadAll(request.Body)
		_ = request.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(eotJSON(request.Header.Get("X-Request-ID"), len(pcm)/2, 0.75))),
			Request:    request,
		}, nil
	})

	score, err := client.Score(context.Background(), "anonymous-candidate", make([]byte, MinSamples*2))
	require.NoError(t, err)
	require.Equal(t, 0.75, score.Probability)
	require.Equal(t, 1, requests)
}

func TestHostedDemoEOTClientDoesNotFollowRedirects(t *testing.T) {
	client, err := NewHostedClient()
	require.NoError(t, err)
	var requests int
	client.client.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		requests++
		if got := request.Header.Get("Authorization"); got != "" {
			t.Errorf("hosted request unexpectedly carried Authorization")
		}
		_ = request.Body.Close()
		return &http.Response{
			StatusCode: http.StatusFound,
			Header:     http.Header{"Location": []string{"https://attacker.example/collect"}},
			Body:       io.NopCloser(strings.NewReader("")),
			Request:    request,
		}, nil
	})
	_, err = client.Score(context.Background(), "redirect-candidate", make([]byte, MinSamples*2))
	require.Error(t, err)
	require.Equal(t, 1, requests, "the hosted client must not forward requests through redirects")
}
