package keypool

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool/keypooltest"
)

// The vendor answers below are the examples each vendor documents, not responses recorded
// from a live account.

// openAIHeaders are the rate-limit headers of OpenAI's example response:
// https://developers.openai.com/api/docs/guides/rate-limits (rate limits in headers).
var openAIHeaders = http.Header{
	"X-Ratelimit-Limit-Requests":           {"60"},
	"X-Ratelimit-Limit-Tokens":             {"150000"},
	"X-Ratelimit-Remaining-Requests":       {"59"},
	"X-Ratelimit-Remaining-Tokens":         {"149984"},
	"X-Ratelimit-Reset-Requests":           {"1s"},
	"X-Ratelimit-Reset-Tokens":             {"6m0s"},
	"X-Ratelimit-Remaining-Project-Tokens": {"149000"},
}

// anthropicHeaders are the headers Anthropic documents on its responses, with values of their
// documented formats (counts, and RFC 3339 resets):
// https://platform.claude.com/docs/en/api/rate-limits#response-headers
var anthropicHeaders = http.Header{
	"Anthropic-Ratelimit-Requests-Limit":          {"4000"},
	"Anthropic-Ratelimit-Requests-Remaining":      {"3999"},
	"Anthropic-Ratelimit-Requests-Reset":          {"2026-11-05T12:00:01Z"},
	"Anthropic-Ratelimit-Tokens-Remaining":        {"2390000"},
	"Anthropic-Ratelimit-Input-Tokens-Remaining":  {"1999000"},
	"Anthropic-Ratelimit-Output-Tokens-Remaining": {"391000"},
}

// anthropicSpendCapBody is Anthropic's 429 for an organization past its monthly spend cap, as
// documented, its request id replaced: https://platform.claude.com/docs/en/api/rate-limits#reaching-your-spend-cap
const anthropicSpendCapBody = `{"type":"error","error":{"type":"rate_limit_error","message":"You have reached your API usage limits: your organization has crossed its monthly API usage threshold, set based on your organization's API tier. You will regain access on 2026-09-01 at 00:00 UTC.","details":{"error_code":"enforced_spend_limit_reached"}},"request_id":"req_example"}`

// anthropicRateLimitBody is Anthropic's 429 for a rate limit: the error type it documents for
// one, without the spend cap's error_code: https://platform.claude.com/docs/en/api/errors
const anthropicRateLimitBody = `{"type":"error","error":{"type":"rate_limit_error","message":"This request would exceed your organization's rate limit."}}`

// VendorRateLimitSuite proves the pool counts a vendor's 429 and keeps the headroom its
// headers report, per variable and key, only on keys read from a variable, and that doing so
// leaves what the caller gets as it was. Every test has keys and variables of its own, new on
// every run: pools and the metrics are the process's.
type VendorRateLimitSuite struct {
	suite.Suite

	// server answers each key in answers with its status, headers and body, and every other
	// key with 200, a websocket handshake on it accepted.
	server *httptest.Server

	mu      sync.Mutex
	answers map[string]answer
	seen    []string
}

type answer struct {
	status int
	header http.Header
	body   string
}

func TestVendorRateLimitSuite(t *testing.T) {
	suite.Run(t, new(VendorRateLimitSuite))
}

func (s *VendorRateLimitSuite) SetupSuite() {
	upgrader := websocket.Upgrader{}
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		s.mu.Lock()
		s.seen = append(s.seen, key)
		given, ok := s.answers[key]
		s.mu.Unlock()

		if ok && given.status == 0 && websocket.IsWebSocketUpgrade(r) {
			if conn, err := upgrader.Upgrade(w, r, given.header); err == nil {
				_ = conn.Close()
			}
			return
		}
		for name, values := range given.header {
			w.Header()[name] = values
		}
		if !ok {
			if websocket.IsWebSocketUpgrade(r) {
				if conn, err := upgrader.Upgrade(w, r, nil); err == nil {
					_ = conn.Close()
				}
				return
			}
			_, _ = io.WriteString(w, "answered")
			return
		}
		w.WriteHeader(given.status)
		_, _ = io.WriteString(w, given.body)
	}))
}

func (s *VendorRateLimitSuite) TearDownSuite() { s.server.Close() }

func (s *VendorRateLimitSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.answers = map[string]answer{}
	s.seen = nil
}

func (s *VendorRateLimitSuite) answer(key string, given answer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.answers[key] = given
}

func (s *VendorRateLimitSuite) keysSeen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.seen...)
}

// pool is the pool FromEnv builds for two keys read from a variable named for the test.
func (s *VendorRateLimitSuite) pool(name string) pooled {
	id := rand.Text()
	keyVar := fmt.Sprintf("VR_%s_%s_API_KEY", strings.ToUpper(name), id)
	keys := []string{fmt.Sprintf("vr-%s-%s-1", name, id), fmt.Sprintf("vr-%s-%s-2", name, id)}
	s.T().Setenv(keyVar, strings.Join(keys, ","))
	return pooled{Pool: FromEnv("", keyVar), keyVars: []string{keyVar}, keys: keys}
}

// post sends a request through pool and returns the status and body it ended with.
func (s *VendorRateLimitSuite) post(pool *Pool) (int, string) {
	request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("body"))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer placeholder")
	response, err := pool.WrapClient(nil, Bearer).Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(body)
}

func (s *VendorRateLimitSuite) dial(pool *Pool) error {
	endpoint := "ws" + strings.TrimPrefix(s.server.URL, "http")
	conn, _, _, err := pool.Dial(context.Background(), websocket.DefaultDialer, func(key string) (string, http.Header) {
		return endpoint, http.Header{"Authorization": {"Bearer " + key}}
	})
	if conn != nil {
		_ = conn.Close()
	}
	return err
}

func (s *VendorRateLimitSuite) limited(keyVar string) map[string]float64 {
	return keypooltest.Series(s.T(), "router_vendor_rate_limited_total", keyVar)
}

func (s *VendorRateLimitSuite) remaining(keyVar string) map[string]float64 {
	return keypooltest.Series(s.T(), "router_vendor_ratelimit_remaining", keyVar)
}

func (s *VendorRateLimitSuite) TestA429IsCountedUnderTheKeyItRefusedAndTheNextKeyAnswers() {
	p := s.pool("passed")
	s.answer(p.keys[0], answer{status: http.StatusTooManyRequests, body: anthropicRateLimitBody})

	status, _ := s.post(p.Pool)

	s.Equal(http.StatusOK, status)
	s.Equal(p.keys, s.keysSeen())
	s.Equal(map[string]float64{"0": 1, "1": 0}, s.limited(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestEveryKeyA429RefusesIsCountedAndTheLastIsHandedBackWhole() {
	p := s.pool("exhausted")
	for _, key := range p.keys {
		s.answer(key, answer{status: http.StatusTooManyRequests, body: anthropicRateLimitBody})
	}

	status, body := s.post(p.Pool)

	s.Equal(http.StatusTooManyRequests, status)
	s.Equal(anthropicRateLimitBody, body)
	s.Equal(map[string]float64{"0": 1, "1": 1}, s.limited(p.keyVars[0]))
}

// The spend cap is a 429 too, and the pool still rests the key and moves on as it did; it is
// only not a rate limit.
func (s *VendorRateLimitSuite) TestAnthropicsSpendCapIsNotARateLimit() {
	p := s.pool("capped")
	for _, key := range p.keys {
		s.answer(key, answer{status: http.StatusTooManyRequests, body: anthropicSpendCapBody})
	}

	status, body := s.post(p.Pool)

	s.Equal(http.StatusTooManyRequests, status)
	s.Equal(anthropicSpendCapBody, body)
	s.Equal(p.keys, s.keysSeen())
	s.Equal(map[string]float64{"0": 0, "1": 0}, s.limited(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAnotherRefusalOrFailureIsNotARateLimit() {
	for _, status := range []int{http.StatusOK, http.StatusUnauthorized, http.StatusPaymentRequired,
		http.StatusForbidden, http.StatusInternalServerError, http.StatusServiceUnavailable, 529} {
		p := s.pool(fmt.Sprintf("status%d", status))
		s.answer(p.keys[0], answer{status: status, body: `{"error":"rate limit"}`})

		s.post(p.Pool)

		s.Equal(map[string]float64{"0": 0, "1": 0}, s.limited(p.keyVars[0]), "status %d", status)
	}
}

// openai.go and the speech providers pass a customer's own key to FromEnv beside the router's
// variable: the pool takes the variable's name, but its keys were never read from it.
func (s *VendorRateLimitSuite) TestACustomersOwnKeyIsNeitherCountedNorItsHeadroomKept() {
	id := rand.Text()
	keyVar := fmt.Sprintf("VR_BYOK_%s_API_KEY", id)
	s.T().Setenv(keyVar, "vr-router-"+id)
	customer := "vr-customer-" + id
	pool := FromEnv(customer, keyVar)
	s.answer(customer, answer{status: http.StatusTooManyRequests, header: openAIHeaders})

	s.post(pool)

	s.Empty(s.limited(keyVar))
	s.Empty(s.remaining(keyVar))
	s.Empty(s.limited(""), "nor under no variable")
	s.Empty(s.remaining(""), "nor under no variable")
	s.Equal(1.0, keypooltest.Refused(s.T(), keyVar, "429"), "the refusal counter keeps naming it as on the base")
}

// anthropic.go and the OpenAI-compatible LLMs read the keys with Getenv and pass them in, with
// no variable.
func (s *VendorRateLimitSuite) TestKeysPassedInAreCountedUnderTheVariableTheyWereReadFrom() {
	id := rand.Text()
	keyVar := fmt.Sprintf("VR_PASSED_%s_API_KEY", id)
	keys := []string{"vr-passed-" + id + "-1", "vr-passed-" + id + "-2"}
	s.T().Setenv(keyVar, strings.Join(keys, ","))
	pool := FromEnv(Getenv(keyVar))
	s.answer(keys[1], answer{status: http.StatusTooManyRequests})
	pool.Next()

	s.post(pool)

	s.Equal(map[string]float64{"0": 0, "1": 1}, s.limited(keyVar))
	s.Equal(0.0, keypooltest.Refused(s.T(), keyVar, "429"), "the refusal counter names it unknown as on the base")
}

func (s *VendorRateLimitSuite) TestAPoolOfNewIsNotCounted() {
	id := rand.Text()
	keyVar := fmt.Sprintf("VR_NEW_%s_API_KEY", id)
	s.T().Setenv(keyVar, "vr-new-"+id)
	Getenv(keyVar)
	pool := New("vr-new-" + id)
	s.answer("vr-new-"+id, answer{status: http.StatusTooManyRequests, header: openAIHeaders})

	s.post(pool)

	s.Empty(s.limited(keyVar))
	s.Empty(s.remaining(keyVar))
}

func (s *VendorRateLimitSuite) TestOpenAIsHeadroomIsKeptPerKey() {
	p := s.pool("openai")
	s.answer(p.keys[0], answer{status: http.StatusOK, header: openAIHeaders})
	s.answer(p.keys[1], answer{status: http.StatusOK, header: http.Header{"X-Ratelimit-Remaining-Requests": {"12"}}})

	s.post(p.Pool)
	s.post(p.Pool)

	s.Equal(map[string]float64{
		"0/x-ratelimit-remaining-requests":       59,
		"0/x-ratelimit-remaining-tokens":         149984,
		"0/x-ratelimit-remaining-project-tokens": 149000,
		"1/x-ratelimit-remaining-requests":       12,
	}, s.remaining(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestTheLastValueReportedIsKept() {
	p := s.pool("last")
	s.answer(p.keys[0], answer{status: http.StatusOK, header: http.Header{"X-Ratelimit-Remaining-Requests": {"59"}}})
	s.post(p.Pool)
	s.answer(p.keys[0], answer{status: http.StatusTooManyRequests, header: http.Header{"X-Ratelimit-Remaining-Requests": {"0"}}})
	s.answer(p.keys[1], answer{status: http.StatusTooManyRequests})

	s.post(p.Pool)

	s.Equal(map[string]float64{"0/x-ratelimit-remaining-requests": 0}, s.remaining(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAnthropicsHeadroomIsKept() {
	p := s.pool("anthropic")
	s.answer(p.keys[0], answer{status: http.StatusOK, header: anthropicHeaders})

	s.post(p.Pool)

	s.Equal(map[string]float64{
		"0/anthropic-ratelimit-requests-remaining":      3999,
		"0/anthropic-ratelimit-tokens-remaining":        2390000,
		"0/anthropic-ratelimit-input-tokens-remaining":  1999000,
		"0/anthropic-ratelimit-output-tokens-remaining": 391000,
	}, s.remaining(p.keyVars[0]))
}

// ElevenLabs sends how many requests run and how many may: https://elevenlabs.io/docs/overview/models#concurrency-and-priority
func (s *VendorRateLimitSuite) TestElevenLabsConcurrencyHeadroomIsTheMaximumLessTheCurrent() {
	p := s.pool("elevenlabs")
	s.answer(p.keys[0], answer{status: http.StatusOK, header: http.Header{
		"Current-Concurrent-Requests": {"3"}, "Maximum-Concurrent-Requests": {"10"}}})
	s.answer(p.keys[1], answer{status: http.StatusOK, header: http.Header{"Current-Concurrent-Requests": {"3"}}})
	s.post(p.Pool)
	s.post(p.Pool)

	s.Equal(map[string]float64{"0/concurrent-requests": 7}, s.remaining(p.keyVars[0]), "one of the two says nothing of what is left")

	q := s.pool("elevenlabsmax")
	s.answer(q.keys[0], answer{status: http.StatusOK, header: http.Header{"Maximum-Concurrent-Requests": {"10"}}})
	s.post(q.Pool)
	s.Empty(s.remaining(q.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAVendorThatReportsNoHeadroomHasNoSeries() {
	p := s.pool("silent")
	s.answer(p.keys[0], answer{status: http.StatusOK, header: http.Header{
		"X-Ratelimit-Limit-Requests": {"60"}, "X-Ratelimit-Reset-Requests": {"1s"}, "Retry-After": {"3"}}})

	s.post(p.Pool)

	s.Empty(s.remaining(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestOnlyACountIsKept() {
	cases := []struct {
		name   string
		values []string
		want   float64
		kept   bool
	}{
		{"integer", []string{"59"}, 59, true},
		{"zero", []string{"0"}, 0, true},
		{"fraction", []string{"1.5"}, 1.5, true},
		{"exponent", []string{"1e3"}, 1000, true},
		{"first of two values", []string{"5", "6"}, 5, true},
		{"empty", []string{""}, 0, false},
		{"text", []string{"abc"}, 0, false},
		{"a duration", []string{"6m0s"}, 0, false},
		{"negative", []string{"-1"}, 0, false},
		{"not a number", []string{"NaN"}, 0, false},
		{"infinite", []string{"Inf"}, 0, false},
		{"negative infinite", []string{"-Inf"}, 0, false},
		{"a list", []string{"12, 13"}, 0, false},
	}
	for _, c := range cases {
		p := s.pool("count")
		s.answer(p.keys[0], answer{status: http.StatusOK, header: http.Header{"X-Ratelimit-Remaining-Requests": c.values}})

		s.post(p.Pool)

		if c.kept {
			s.Equal(map[string]float64{"0/x-ratelimit-remaining-requests": c.want}, s.remaining(p.keyVars[0]), c.name)
		} else {
			s.Empty(s.remaining(p.keyVars[0]), c.name)
		}
	}
}

func (s *VendorRateLimitSuite) TestARefusedHandshakeIsCountedAndTheNextKeyDialled() {
	p := s.pool("handshake")
	s.answer(p.keys[0], answer{status: http.StatusTooManyRequests, header: http.Header{"Retry-After": {"10"}}})

	s.Require().NoError(s.dial(p.Pool))

	s.Equal(p.keys, s.keysSeen())
	s.Equal(map[string]float64{"0": 1, "1": 0}, s.limited(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAHandshakeThatGetsNoAnswerFailsAsItDid() {
	p := s.pool("unanswered")
	closed := httptest.NewServer(http.NotFoundHandler())
	endpoint := "ws" + strings.TrimPrefix(closed.URL, "http")
	closed.Close()

	_, response, key, err := p.Dial(context.Background(), websocket.DefaultDialer, func(key string) (string, http.Header) {
		return endpoint, http.Header{"Authorization": {"Bearer " + key}}
	})

	s.Error(err)
	s.Nil(response)
	s.Equal(p.keys[0], key)
	s.Equal(map[string]float64{"0": 0, "1": 0}, s.limited(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAnAcceptedHandshakeKeepsItsHeadroom() {
	p := s.pool("accepted")
	s.answer(p.keys[0], answer{header: http.Header{"Current-Concurrent-Requests": {"1"}, "Maximum-Concurrent-Requests": {"5"}}})

	s.Require().NoError(s.dial(p.Pool))

	s.Equal(map[string]float64{"0": 0, "1": 0}, s.limited(p.keyVars[0]))
	s.Equal(map[string]float64{"0/concurrent-requests": 4}, s.remaining(p.keyVars[0]))
}

// A handshake is counted under the key it was dialled with, not the one now next in turn:
// a count under the wrong index would point an operator at a key that was fine.
func (s *VendorRateLimitSuite) TestAHandshakeOnTheSecondKeyIsCountedUnderTheSecondKey() {
	p := s.pool("handshakesecond")
	s.answer(p.keys[1], answer{status: http.StatusTooManyRequests, header: http.Header{"Retry-After": {"10"}}})
	p.Next()

	s.Require().NoError(s.dial(p.Pool))

	s.Equal([]string{p.keys[1], p.keys[0]}, s.keysSeen())
	s.Equal(map[string]float64{"0": 0, "1": 1}, s.limited(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAnAcceptedHandshakeOnTheSecondKeyKeepsItsHeadroomUnderTheSecondKey() {
	p := s.pool("acceptedsecond")
	s.answer(p.keys[1], answer{header: http.Header{"Current-Concurrent-Requests": {"1"}, "Maximum-Concurrent-Requests": {"5"}}})
	p.Next()

	s.Require().NoError(s.dial(p.Pool))

	s.Equal(map[string]float64{"1/concurrent-requests": 4}, s.remaining(p.keyVars[0]))
}

func (s *VendorRateLimitSuite) TestAFrameReportsHeadroomOnlyOnTheRoutersKeysAndOnlyAsACount() {
	p := s.pool("frame")

	p.Remaining(p.keys[1], "tokens", 49950)
	p.Remaining(p.keys[0], "requests", math.NaN())
	p.Remaining(p.keys[0], "requests", -1)
	p.Remaining("vr-not-in-the-pool", "requests", 5)

	s.Equal(map[string]float64{"1/tokens": 49950}, s.remaining(p.keyVars[0]))

	id := rand.Text()
	keyVar := fmt.Sprintf("VR_FRAMEBYOK_%s_API_KEY", id)
	s.T().Setenv(keyVar, "vr-router-"+id)
	customer := FromEnv("vr-customer-"+id, keyVar)
	customer.Remaining("vr-customer-"+id, "tokens", 5)
	s.Empty(s.remaining(keyVar))
	s.Empty(s.remaining(""), "nor under no variable")
}

func (s *VendorRateLimitSuite) TestNoLabelHoldsAKey() {
	p := s.pool("labelled")
	for _, key := range p.keys {
		s.answer(key, answer{status: http.StatusTooManyRequests, header: openAIHeaders})
	}

	s.post(p.Pool)

	for _, series := range []map[string]float64{s.limited(p.keyVars[0]), s.remaining(p.keyVars[0])} {
		s.NotEmpty(series)
		for labels := range series {
			for _, key := range p.keys {
				s.NotContains(labels, key)
			}
		}
	}
}

// The pool reads the start of a 429 to tell a spend cap: what it read is put back.
func (s *VendorRateLimitSuite) TestA429LongerThanWhatIsReadOfItIsHandedBackWhole() {
	p := s.pool("long")
	long := strings.Repeat("x", 3*drainLimit)
	for _, key := range p.keys {
		s.answer(key, answer{status: http.StatusTooManyRequests, body: long})
	}

	status, body := s.post(p.Pool)

	s.Equal(http.StatusTooManyRequests, status)
	s.Equal(long, body)
	s.Equal(p.keys, s.keysSeen())
}

// GOOGLE_API_KEY and GEMINI_API_KEY may hold the same key: its series keeps the first name it
// was read under, not the last.
func (s *VendorRateLimitSuite) TestKeysReadFromTwoVariablesKeepTheFirst() {
	p := s.pool("first")
	second := fmt.Sprintf("VR_SECOND_%s_API_KEY", rand.Text())
	s.T().Setenv(second, strings.Join(p.keys, ","))
	FromEnv("", second)
	s.answer(p.keys[0], answer{status: http.StatusTooManyRequests})

	s.post(p.Pool)

	s.Equal(map[string]float64{"0": 1, "1": 0}, s.limited(p.keyVars[0]))
	s.Empty(s.limited(second))
}

// Providers are built, and send, on many goroutines over one pool: run under -race.
func (s *VendorRateLimitSuite) TestRequestsSentWhileTheProviderIsBuiltAgainAreEachCounted() {
	p := s.pool("concurrent")
	for _, key := range p.keys {
		s.answer(key, answer{status: http.StatusTooManyRequests, header: http.Header{"X-Ratelimit-Remaining-Requests": {"0"}}})
	}
	const senders = 8
	client := p.WrapClient(nil, Bearer)
	errs := make(chan error, senders)
	var wg sync.WaitGroup
	for range senders {
		wg.Go(func() {
			FromEnv("", p.keyVars[0])
			request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("body"))
			if err == nil {
				request.Header.Set("Authorization", "Bearer placeholder")
				var response *http.Response
				if response, err = client.Do(request); err == nil {
					_ = response.Body.Close()
				}
			}
			errs <- err
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		s.Require().NoError(err)
	}

	limited := s.limited(p.keyVars[0])
	s.Equal(float64(senders*len(p.keys)), limited["0"]+limited["1"])
	s.Equal(map[string]float64{"0/x-ratelimit-remaining-requests": 0, "1/x-ratelimit-remaining-requests": 0}, s.remaining(p.keyVars[0]))
}
