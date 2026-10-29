package keypool

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
)

// RefusedAttemptsSuite proves each attempt a provider refuses on a pooled key is counted
// once, under the pool's provider, and that counting leaves the turn of the keys as it was.
// Every test has keys and variables of its own, new on every run: pools and the counter are
// shared by the whole process, so a rerun (-count) must not meet what the last one left.
type RefusedAttemptsSuite struct {
	suite.Suite

	// server answers each key in statuses with its status: 429 with a Retry-After, 401 as a
	// wrong key, 402 as a spent quota, any other as is, each with the key's body in bodies
	// instead when it has one. Every other key is answered, and a websocket handshake on it
	// accepted.
	server *httptest.Server

	mu       sync.Mutex
	statuses map[string]int
	bodies   map[string]string
	seen     []string
}

// outOfCreditBody is how Gemini refused the staging router's key once its prepaid credits ran
// out (router log 2026-10-27 21:27Z): a 402 that does not say quota_exceeded.
const outOfCreditBody = `[{"error":{"code":402,"message":"Your prepayment credits are depleted. Please go to AI Studio at https://ai.studio/projects to manage your project and billing. Learn more at https://ai.google.dev/gemini-api/docs/billing#prepay. ","status":"RESOURCE_EXHAUSTED"}}]`

func TestRefusedAttemptsSuite(t *testing.T) {
	suite.Run(t, new(RefusedAttemptsSuite))
}

func (s *RefusedAttemptsSuite) SetupSuite() {
	upgrader := websocket.Upgrader{}
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		s.mu.Lock()
		s.seen = append(s.seen, key)
		status := s.statuses[key]
		body, hasBody := s.bodies[key]
		s.mu.Unlock()

		switch {
		case hasBody:
			w.WriteHeader(status)
			_, _ = io.WriteString(w, body)
		case status == 0:
			if websocket.IsWebSocketUpgrade(r) {
				if conn, err := upgrader.Upgrade(w, r, nil); err == nil {
					_ = conn.Close()
				}
				return
			}
			_, _ = io.WriteString(w, "answered")
		case status == http.StatusTooManyRequests:
			w.Header().Set("Retry-After", "10")
			w.WriteHeader(status)
		case status == http.StatusUnauthorized:
			w.WriteHeader(status)
			_, _ = io.WriteString(w, invalidKeyBody)
		case status == http.StatusPaymentRequired:
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"detail":{"code":"quota_exceeded"}}`)
		default:
			w.WriteHeader(status)
		}
	}))
}

func (s *RefusedAttemptsSuite) TearDownSuite() { s.server.Close() }

func (s *RefusedAttemptsSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.statuses = map[string]int{}
	s.bodies = map[string]string{}
	s.seen = nil
}

func (s *RefusedAttemptsSuite) answer(status int, keys ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.statuses[key] = status
	}
}

func (s *RefusedAttemptsSuite) answerWith(status int, body string, keys ...string) {
	s.answer(status, keys...)
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.bodies[key] = body
	}
}

func (s *RefusedAttemptsSuite) keysSeen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.seen...)
}

// pooled is a pool FromEnv built, with the variables and keys it was built from.
type pooled struct {
	*Pool
	keyVars []string
	keys    []string
}

// pool is the pool FromEnv builds for two keys read from a variable named for the test.
// Their names are new on every run.
func (s *RefusedAttemptsSuite) pool(name string) pooled {
	id := rand.Text()
	keyVar := fmt.Sprintf("RA_%s_%s_API_KEY", strings.ToUpper(name), id)
	keys := []string{fmt.Sprintf("ra-%s-%s-1", name, id), fmt.Sprintf("ra-%s-%s-2", name, id)}
	s.T().Setenv(keyVar, strings.Join(keys, ","))
	return pooled{Pool: FromEnv("", keyVar), keyVars: []string{keyVar}, keys: keys}
}

// post sends a request through pool and returns the status it ended with.
func (s *RefusedAttemptsSuite) post(pool *Pool) int {
	status, _ := s.postAs(pool, "Bearer placeholder")
	return status
}

// postAs sends a request with the Authorization header authorization, none when it is "",
// through pool and returns the status and the body it ended with.
func (s *RefusedAttemptsSuite) postAs(pool *Pool, authorization string) (int, string) {
	request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("body"))
	s.Require().NoError(err)
	if authorization != "" {
		request.Header.Set("Authorization", authorization)
	}
	response, err := pool.WrapClient(nil, Bearer).Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(body)
}

func (s *RefusedAttemptsSuite) dial(pool *Pool) (*http.Response, string, error) {
	endpoint := "ws" + strings.TrimPrefix(s.server.URL, "http")
	conn, response, key, err := pool.Dial(context.Background(), websocket.DefaultDialer, func(key string) (string, http.Header) {
		return endpoint, http.Header{"Authorization": {"Bearer " + key}}
	})
	if conn != nil {
		_ = conn.Close()
	}
	return response, key, err
}

// refusals is every series of the counter, as key_var → status → count.
func refusals(t *testing.T) map[string]map[string]float64 {
	registry := prometheus.NewPedanticRegistry()
	registry.MustRegister(refusedAttempts)
	families, err := registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	counted := map[string]map[string]float64{}
	for _, family := range families {
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if counted[labels["key_var"]] == nil {
				counted[labels["key_var"]] = map[string]float64{}
			}
			counted[labels["key_var"]][labels["status"]] = metric.GetCounter().GetValue()
		}
	}
	return counted
}

func (s *RefusedAttemptsSuite) counted(keyVar string) map[string]float64 {
	return refusals(s.T())[keyVar]
}

// The keys seen and the next key in these tests are what the base before the counter
// answered to the same requests.

func (s *RefusedAttemptsSuite) TestARefusalAnotherKeyGetsPastIsCountedOnceAndTheCallSucceeds() {
	p := s.pool("passed")
	s.answer(http.StatusTooManyRequests, p.keys[0])

	s.Equal(http.StatusOK, s.post(p.Pool))

	s.Equal(p.keys, s.keysSeen())
	s.Equal(p.keys[1], p.Next())
	s.Equal(map[string]float64{"429": 1}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestEveryRefusedAttemptIsCountedWhenNoKeyIsLeft() {
	p := s.pool("exhausted")
	s.answer(http.StatusTooManyRequests, p.keys...)

	s.Equal(http.StatusTooManyRequests, s.post(p.Pool))

	s.Equal(p.keys, s.keysSeen())
	s.Equal(p.keys[0], p.Next())
	s.Equal(map[string]float64{"429": 2}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAnAnsweredRequestCountsNoRefusal() {
	p := s.pool("answered")

	s.Equal(http.StatusOK, s.post(p.Pool))

	s.Equal(p.keys[:1], s.keysSeen())
	s.Equal(p.keys[1], p.Next())
	s.Empty(s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAServerErrorCountsNoRefusal() {
	p := s.pool("failing")
	s.answer(http.StatusInternalServerError, p.keys[0])

	s.Equal(http.StatusInternalServerError, s.post(p.Pool))

	s.Equal(p.keys[:1], s.keysSeen())
	s.Equal(p.keys[1], p.Next())
	s.Empty(s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestASpentQuotaIsCountedUnderItsStatus() {
	p := s.pool("spent")
	s.answer(http.StatusPaymentRequired, p.keys[0])

	s.Equal(http.StatusOK, s.post(p.Pool))

	s.Equal(p.keys, s.keysSeen())
	s.Equal([]string{p.keys[1], p.keys[1]}, []string{p.Next(), p.Next()}, "the spent key rests")
	s.Equal(map[string]float64{"402": 1}, s.counted(p.keyVars[0]))
}

// A wrong key is not rested or moved past, as on the base; it is now counted.
func (s *RefusedAttemptsSuite) TestAWrongKeyIsCountedAndHandedBack() {
	p := s.pool("wrong")
	s.answer(http.StatusUnauthorized, p.keys[0])

	status, body := s.postAs(p.Pool, "Bearer placeholder")

	s.Equal(http.StatusUnauthorized, status)
	s.Equal(invalidKeyBody, body)
	s.Equal(p.keys[:1], s.keysSeen())
	s.Equal([]string{p.keys[1], p.keys[0]}, []string{p.Next(), p.Next()}, "the wrong key does not rest")
	s.Equal(map[string]float64{"401": 1}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAnOutOfCreditKeyIsCountedAndItsRefusalHandedBackWhole() {
	p := s.pool("credit")
	s.answerWith(http.StatusPaymentRequired, outOfCreditBody, p.keys[0])

	status, body := s.postAs(p.Pool, "Bearer placeholder")

	s.Equal(http.StatusPaymentRequired, status)
	s.Equal(outOfCreditBody, body)
	s.Equal(p.keys[:1], s.keysSeen())
	s.Equal([]string{p.keys[1], p.keys[0]}, []string{p.Next(), p.Next()}, "the key does not rest")
	s.Equal(map[string]float64{"402": 1}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAForbiddenKeyCountsNoRefusal() {
	p := s.pool("forbidden")
	s.answer(http.StatusForbidden, p.keys[0])

	s.Equal(http.StatusForbidden, s.post(p.Pool))

	s.Equal(p.keys[:1], s.keysSeen())
	s.Empty(s.counted(p.keyVars[0]))
}

// A request that carries no key is sent as it is, never on a pooled key, so its 401 says
// nothing of the pool's keys.
func (s *RefusedAttemptsSuite) TestAWrongKeyOnARequestWithoutAPooledKeyCountsNothing() {
	p := s.pool("keyless")
	s.answer(http.StatusUnauthorized, "")

	status, _ := s.postAs(p.Pool, "")

	s.Equal(http.StatusUnauthorized, status)
	s.Equal([]string{""}, s.keysSeen())
	s.Empty(s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestARefusedHandshakeIsCountedOnceAndDialledOnTheNextKey() {
	p := s.pool("dialled")
	s.answer(http.StatusTooManyRequests, p.keys[0])

	response, key, err := s.dial(p.Pool)

	s.Require().NoError(err)
	s.Equal(http.StatusSwitchingProtocols, response.StatusCode)
	s.Equal(p.keys[1], key)
	s.Equal(p.keys, s.keysSeen())
	s.Equal(p.keys[1], p.Next())
	s.Equal(map[string]float64{"429": 1}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestEveryRefusedHandshakeIsCountedWhenNoKeyIsLeft() {
	p := s.pool("undialled")
	s.answer(http.StatusTooManyRequests, p.keys...)

	response, key, err := s.dial(p.Pool)

	s.Require().Error(err)
	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal(p.keys[1], key)
	s.Equal(p.keys, s.keysSeen())
	s.Equal(p.keys[0], p.Next())
	s.Equal(map[string]float64{"429": 2}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAHandshakeOnAWrongOrOutOfCreditKeyIsCountedAndNotDialledAgain() {
	for _, tc := range []struct {
		name   string
		status int
		body   string
	}{
		{"wrong key", http.StatusUnauthorized, invalidKeyBody},
		{"out of credit", http.StatusPaymentRequired, outOfCreditBody},
	} {
		s.Run(tc.name, func() {
			s.SetupTest()
			p := s.pool("plaindial")
			s.answerWith(tc.status, tc.body, p.keys[0])

			response, key, err := s.dial(p.Pool)

			s.Require().Error(err)
			s.Equal(tc.status, response.StatusCode)
			s.Equal(p.keys[0], key)
			s.Equal(p.keys[:1], s.keysSeen())
			s.Equal([]string{p.keys[1], p.keys[0]}, []string{p.Next(), p.Next()}, "the key does not rest")
			s.Equal(map[string]float64{strconv.Itoa(tc.status): 1}, s.counted(p.keyVars[0]))
		})
	}
}

func (s *RefusedAttemptsSuite) TestAHandshakeOnASpentKeyIsCountedOnceAndDialledOnTheNextKey() {
	p := s.pool("spentdial")
	s.answer(http.StatusPaymentRequired, p.keys[0])

	response, key, err := s.dial(p.Pool)

	s.Require().NoError(err)
	s.Equal(http.StatusSwitchingProtocols, response.StatusCode)
	s.Equal(p.keys[1], key)
	s.Equal([]string{p.keys[1], p.keys[1]}, []string{p.Next(), p.Next()}, "the spent key rests")
	s.Equal(map[string]float64{"402": 1}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAHandshakeThatFailsWithoutARefusalCountsNothing() {
	p := s.pool("broken")
	s.answer(http.StatusInternalServerError, p.keys[0])

	response, key, err := s.dial(p.Pool)

	s.Require().Error(err)
	s.Equal(http.StatusInternalServerError, response.StatusCode)
	s.Equal(p.keys[0], key)
	s.Equal(p.keys[:1], s.keysSeen())
	s.Equal(p.keys[1], p.Next())
	s.Empty(s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestARefusalIsCountedUnderItsVariableAndNeverItsKey() {
	p := s.pool("labelled")
	s.answer(http.StatusTooManyRequests, p.keys...)

	s.post(p.Pool)

	counted := refusals(s.T())
	s.Equal(map[string]float64{"429": 2}, counted[p.keyVars[0]])
	for keyVar, statuses := range counted {
		for _, key := range p.keys {
			s.NotContains(keyVar, key)
			for status := range statuses {
				s.NotContains(status, key)
			}
		}
	}
}

func (s *RefusedAttemptsSuite) TestAPoolNoVariableNamesIsCountedAsUnknown() {
	id := rand.Text()
	pool := New(fmt.Sprintf("ra-unnamed-%s-1,ra-unnamed-%s-2", id, id))
	s.answer(http.StatusTooManyRequests, fmt.Sprintf("ra-unnamed-%s-1", id))
	before := s.counted("unknown")["429"]

	s.Equal(http.StatusOK, s.post(pool))

	s.Equal(before+1, s.counted("unknown")["429"])
}

func (s *RefusedAttemptsSuite) TestAPoolOfKeysPassedInTakesTheVariableTheSameKeysAreReadFrom() {
	id := rand.Text()
	keys := []string{fmt.Sprintf("ra-adopted-%s-1", id), fmt.Sprintf("ra-adopted-%s-2", id)}
	keyVar := fmt.Sprintf("RA_ADOPTED_%s_API_KEY", id)
	passed := FromEnv(strings.Join(keys, ","))
	s.T().Setenv(keyVar, strings.Join(keys, ","))
	FromEnv("", keyVar)
	s.answer(http.StatusTooManyRequests, keys[0])

	s.Equal(http.StatusOK, s.post(passed))

	s.Equal(map[string]float64{"429": 1}, s.counted(keyVar))
}

func (s *RefusedAttemptsSuite) TestAPoolKeepsTheFirstVariableItIsNamedFor() {
	first := s.pool("first")
	second := fmt.Sprintf("RA_SECOND_%s_API_KEY", rand.Text())
	s.T().Setenv(second, strings.Join(first.keys, ","))
	FromEnv("", second)
	s.answer(http.StatusTooManyRequests, first.keys[0])

	s.Equal(http.StatusOK, s.post(first.Pool))

	s.Equal(map[string]float64{"429": 1}, s.counted(first.keyVars[0]))
	s.Empty(s.counted(second))
}

func (s *RefusedAttemptsSuite) TestThePoolIsNamedForTheFirstOfSeveralVariables() {
	id := rand.Text()
	first, second := fmt.Sprintf("RA_PRIMARY_%s_API_KEY", id), fmt.Sprintf("RA_FALLBACK_%s_API_KEY", id)
	keys := []string{fmt.Sprintf("ra-several-%s-1", id), fmt.Sprintf("ra-several-%s-2", id)}
	s.T().Setenv(second, strings.Join(keys, ","))
	pool := FromEnv("", first, second)
	s.answer(http.StatusTooManyRequests, keys[0])

	s.Equal(http.StatusOK, s.post(pool))

	s.Equal(map[string]float64{"429": 1}, s.counted(first))
	s.Empty(s.counted(second))
}

func (s *RefusedAttemptsSuite) TestAQuotaFrameIsCountedAndRestsTheKey() {
	p := s.pool("frame")

	p.QuotaExceeded(p.keys[0])

	s.Equal(map[string]float64{"quota_exceeded": 1}, s.counted(p.keyVars[0]))
	s.Equal([]string{p.keys[1], p.keys[1]}, []string{p.Next(), p.Next()})
}

// A key refused after the handshake is counted but, like a 401 at the handshake, not rested.
func (s *RefusedAttemptsSuite) TestAKeyRefusedAfterTheHandshakeIsCountedAndStaysInTurn() {
	p := s.pool("invalid")

	p.InvalidKey()

	s.Equal(map[string]float64{"invalid_key": 1}, s.counted(p.keyVars[0]))
	s.Equal([]string{p.keys[0], p.keys[1], p.keys[0]}, []string{p.Next(), p.Next(), p.Next()})
}

// A frame for want of funds is counted under 402, as the handshake counts it, and not rested.
func (s *RefusedAttemptsSuite) TestAFrameForWantOfFundsIsCountedAs402AndStaysInTurn() {
	p := s.pool("funds")

	p.OutOfFunds()

	s.Equal(map[string]float64{"402": 1}, s.counted(p.keyVars[0]))
	s.Equal([]string{p.keys[0], p.keys[1], p.keys[0]}, []string{p.Next(), p.Next(), p.Next()})
}

// refusedAlert is the key_refused condition, fresh, after a first check a limit before start
// that takes in what earlier tests counted, so its next checks hold only what the test did.
func (s *RefusedAttemptsSuite) refusedAlert(start time.Time) alerts.ItemCheck {
	conditions := Alerts(time.Hour)
	s.Require().Len(conditions, 1)
	s.Equal("key_refused", conditions[0].Name)
	s.Equal(alerts.Warn, conditions[0].Level, "a refused key does not page")
	conditions[0].Items(start.Add(-time.Hour))
	return conditions[0].Items
}

func (s *RefusedAttemptsSuite) TestARefusedKeyAlertsOncePerVariableAndStatusUntilTheLimitPasses() {
	start := time.Now()
	refused := s.refusedAlert(start)
	p := s.pool("alerted")
	item := p.keyVars[0] + "/429"
	s.answer(http.StatusTooManyRequests, p.keys...)

	s.post(p.Pool)
	held := refused(start.Add(time.Second))
	s.Equal(map[string]any{"value": 2.0, "increase": 2.0}, held[item])
	for name := range held {
		for _, key := range p.keys {
			s.NotContains(name, key)
		}
	}

	s.post(p.Pool)
	s.NotContains(refused(start.Add(time.Hour)), item, "held back by the limit")

	s.post(p.Pool)
	s.Contains(refused(start.Add(time.Hour+2*time.Second)), item, "sent again once the limit passed")
}

func (s *RefusedAttemptsSuite) TestASpentQuotaFrameAlertsUnderItsVariable() {
	start := time.Now()
	refused := s.refusedAlert(start)
	p := s.pool("framed")

	p.QuotaExceeded(p.keys[0])

	s.Contains(refused(start.Add(time.Second)), p.keyVars[0]+"/quota_exceeded")
}

// The statuses a key is not refused on are the ones the base counted nothing for.
func (s *RefusedAttemptsSuite) TestWhatIsNotARefusalAlertsNothing() {
	for _, tc := range []struct {
		name   string
		status int
	}{
		{name: "answered"},
		{name: "forbidden", status: http.StatusForbidden},
		{name: "server error", status: http.StatusInternalServerError},
	} {
		s.Run(tc.name, func() {
			start := time.Now()
			refused := s.refusedAlert(start)
			p := s.pool("quiet")
			s.answer(tc.status, p.keys[0])

			s.post(p.Pool)

			for item := range refused(start.Add(time.Second)) {
				s.False(strings.HasPrefix(item, p.keyVars[0]+"/"), item)
			}
		})
	}
}

// badKeyAnswers are how each provider that does not answer a made-up key with a 401 answered
// one on 2026-10-27, verbatim (a bogus-key curl of each endpoint).
var badKeyAnswers = []struct {
	name   string
	status int
	body   string
}{
	{"gemini openai-compatible", http.StatusBadRequest, "[{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"Please pass a valid API key\",\n    \"status\": \"INVALID_ARGUMENT\"\n  }\n}\n]"},
	{"gemini native", http.StatusBadRequest, "{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"API key not valid. Please pass a valid API key.\",\n    \"status\": \"INVALID_ARGUMENT\",\n    \"details\": [\n      {\n        \"@type\": \"type.googleapis.com/google.rpc.ErrorInfo\",\n        \"reason\": \"API_KEY_INVALID\",\n        \"domain\": \"googleapis.com\",\n        \"metadata\": {\n          \"service\": \"generativelanguage.googleapis.com\"\n        }\n      },\n      {\n        \"@type\": \"type.googleapis.com/google.rpc.LocalizedMessage\",\n        \"locale\": \"en-US\",\n        \"message\": \"API key not valid. Please pass a valid API key.\"\n      }\n    ]\n  }\n}\n"},
	{"google reason alone", http.StatusBadRequest, `{"error":{"code":400,"message":"x","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}`},
	{"xai", http.StatusBadRequest, `{"code":"invalid-argument","error":"Incorrect API key provided. You can obtain an API key from https://console.x.ai."}`},
	{"xai handshake", http.StatusBadRequest, `{"code":"Client specified an invalid argument","error":"Incorrect API key provided. You can obtain an API key from https://console.x.ai."}`},
	{"baseten", http.StatusForbidden, "{\"error\":\"please check the api-key you provided\"}\n"},
	{"baseten dedicated deployment", http.StatusForbidden, "{\n\t\"error\": \"Authentication failed\"\n}\n"},
	{"inworld key not in base64 id:secret", http.StatusForbidden, `{"code":7,"message":"Invalid authorization credentials","details":[]}`},
	{"inworld key in the real shape", http.StatusForbidden, `{"code":7,"message":"API key does not exist or was deleted: \"Ym9n***\"","details":[]}`},
}

// sender sends a request carrying a key through a pool, the way a provider does.
type sender struct {
	name string
	do   func(pool *Pool, request *http.Request) (*http.Response, error)
}

var senders = []sender{
	{"WrapClient", func(pool *Pool, request *http.Request) (*http.Response, error) {
		return pool.WrapClient(nil, Bearer).Do(request)
	}},
	{"WrapDoer", func(pool *Pool, request *http.Request) (*http.Response, error) {
		return pool.WrapDoer(nil, Bearer).Do(request)
	}},
	{"Send", func(pool *Pool, request *http.Request) (*http.Response, error) {
		return pool.Send(request.Context(), func(key string) (*http.Response, error) {
			attempt := request.Clone(request.Context())
			attempt.Header.Set("Authorization", "Bearer "+key)
			return http.DefaultClient.Do(attempt)
		})
	}},
}

// sendVia sends a request through pool with via and returns the status and body it ended with.
func (s *RefusedAttemptsSuite) sendVia(via sender, pool *Pool) (int, string) {
	request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("body"))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer placeholder")
	response, err := via.do(pool, request)
	s.Require().NoError(err)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, string(body)
}

// A provider's bad-key 400 or 403 is counted as a wrong key's 401 is: not rested or moved
// past, handed back whole.
func (s *RefusedAttemptsSuite) TestABadKeyAnswerThatIsNotA401IsCountedAndHandedBackWhole() {
	for _, via := range senders {
		for _, tc := range badKeyAnswers {
			s.Run(via.name+"/"+tc.name, func() {
				s.SetupTest()
				p := s.pool("badkey")
				s.answerWith(tc.status, tc.body, p.keys[0])

				status, body := s.sendVia(via, p.Pool)

				s.Equal(tc.status, status)
				s.Equal(tc.body, body)
				s.Equal(p.keys[:1], s.keysSeen())
				s.Equal([]string{p.keys[1], p.keys[0]}, []string{p.Next(), p.Next()}, "the key does not rest")
				s.Equal(map[string]float64{strconv.Itoa(tc.status): 1}, s.counted(p.keyVars[0]))
			})
		}
	}
}

func (s *RefusedAttemptsSuite) TestAHandshakeOnABadKeyThatIsNotA401IsCountedAndNotDialledAgain() {
	for _, tc := range badKeyAnswers {
		s.Run(tc.name, func() {
			s.SetupTest()
			p := s.pool("badkeydial")
			s.answerWith(tc.status, tc.body, p.keys[0])

			response, key, err := s.dial(p.Pool)

			s.Require().Error(err)
			s.Equal(tc.status, response.StatusCode)
			body, readErr := io.ReadAll(response.Body)
			s.Require().NoError(readErr)
			// The dialer keeps at most 1 KiB of a refused handshake's body.
			s.Equal(tc.body[:min(len(tc.body), 1024)], string(body))
			s.Equal(p.keys[0], key)
			s.Equal(p.keys[:1], s.keysSeen())
			s.Equal([]string{p.keys[1], p.keys[0]}, []string{p.Next(), p.Next()}, "the key does not rest")
			s.Equal(map[string]float64{strconv.Itoa(tc.status): 1}, s.counted(p.keyVars[0]))
		})
	}
}

// A 400 or 403 is counted on what its body says, not on its status. The body is read only
// as far as the pool reads any refusal (drainLimit) and is still handed back whole.
func (s *RefusedAttemptsSuite) TestA400Or403IsCountedOnlyWhenItsBodySaysTheKeyIsNotValid() {
	marker := "Incorrect API key provided"
	for _, tc := range []struct {
		name    string
		status  int
		body    string
		counted bool
	}{
		{"bad request", http.StatusBadRequest, `{"error":{"code":400,"message":"Invalid JSON payload received. Unknown name \"foo\": Cannot find field.","status":"INVALID_ARGUMENT"}}`, false},
		{"other google reason", http.StatusBadRequest, `{"error":{"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_SERVICE_BLOCKED"}]}}`, false},
		{"model not allowed", http.StatusForbidden, `{"error":"you do not have access to this model"}`, false},
		{"baseten text on a 400", http.StatusBadRequest, `{"error":"Authentication failed"}`, false},
		{"xai text on a 403", http.StatusForbidden, marker, false},
		{"google reason on a 403", http.StatusForbidden, `"API_KEY_INVALID"`, false},
		{"empty 400", http.StatusBadRequest, "", false},
		{"empty 403", http.StatusForbidden, "", false},
		{"not json 400", http.StatusBadRequest, "<html>Bad Request</html>", false},
		{"not json with the marker", http.StatusBadRequest, "error: " + marker, true},
		{"marker in another case", http.StatusBadRequest, strings.ToLower(marker), false},
		{"marker ending at the limit", http.StatusBadRequest, strings.Repeat(" ", drainLimit-len(marker)) + marker, true},
		{"marker across the limit", http.StatusBadRequest, strings.Repeat(" ", drainLimit-len(marker)+1) + marker, false},
		{"marker past the limit", http.StatusForbidden, strings.Repeat(" ", drainLimit) + marker, false},
		{"marker on a server error", http.StatusInternalServerError, marker, false},
		{"marker on a 404", http.StatusNotFound, marker, false},
	} {
		s.Run(tc.name, func() {
			s.SetupTest()
			p := s.pool("edge")
			s.answerWith(tc.status, tc.body, p.keys[0])

			status, body := s.postAs(p.Pool, "Bearer placeholder")

			s.Equal(tc.status, status)
			s.Equal(tc.body, body)
			s.Equal(p.keys[:1], s.keysSeen())
			s.Equal(p.keys[1], p.Next())
			if tc.counted {
				s.Equal(map[string]float64{strconv.Itoa(tc.status): 1}, s.counted(p.keyVars[0]))
			} else {
				s.Empty(s.counted(p.keyVars[0]))
			}
		})
	}
}

// A response a Send callback built without a body has nothing to tell a bad key by: a 400 or
// 403 of it is not counted, and a 401 still is.
func (s *RefusedAttemptsSuite) TestARefusalWithoutABodyIsCountedOnItsStatusAlone() {
	for _, tc := range []struct {
		status  int
		counted bool
	}{
		{http.StatusBadRequest, false},
		{http.StatusForbidden, false},
		{http.StatusUnauthorized, true},
	} {
		s.Run(strconv.Itoa(tc.status), func() {
			p := s.pool("bodiless")

			response, err := p.Send(context.Background(), func(string) (*http.Response, error) {
				return &http.Response{StatusCode: tc.status, Header: http.Header{}}, nil
			})

			s.Require().NoError(err)
			s.Equal(tc.status, response.StatusCode)
			s.Nil(response.Body)
			if tc.counted {
				s.Equal(map[string]float64{strconv.Itoa(tc.status): 1}, s.counted(p.keyVars[0]))
			} else {
				s.Empty(s.counted(p.keyVars[0]))
			}
		})
	}
}

// A handshake answered with a 400 or 403 that is not about the key is not counted, and is
// handed back as it came.
func (s *RefusedAttemptsSuite) TestAHandshakeRefusedForAnotherReasonThan400Or403IsNotCounted() {
	for _, tc := range []struct {
		name   string
		status int
		body   string
	}{
		{"bad request", http.StatusBadRequest, `{"error":{"code":400,"message":"Invalid JSON payload received.","status":"INVALID_ARGUMENT"}}`},
		{"model not allowed", http.StatusForbidden, `{"error":"you do not have access to this model"}`},
		{"empty 400", http.StatusBadRequest, ""},
		{"empty 403", http.StatusForbidden, ""},
	} {
		s.Run(tc.name, func() {
			s.SetupTest()
			p := s.pool("dialother")
			s.answerWith(tc.status, tc.body, p.keys[0])

			response, key, err := s.dial(p.Pool)

			s.Require().Error(err)
			s.Equal(tc.status, response.StatusCode)
			body, readErr := io.ReadAll(response.Body)
			s.Require().NoError(readErr)
			s.Equal(tc.body, string(body))
			s.Equal(p.keys[0], key)
			s.Empty(s.counted(p.keyVars[0]))
		})
	}
}

// closeSpy is a body that records that it was closed.
type closeSpy struct {
	io.Reader
	closed bool
}

func (c *closeSpy) Close() error {
	c.closed = true
	return nil
}

// Closing the body a counted 400 or 403 is handed back with closes the provider's own body.
func (s *RefusedAttemptsSuite) TestClosingACountedAnswersBodyClosesTheProvidersBody() {
	p := s.pool("closes")
	provider := &closeSpy{Reader: strings.NewReader(`{"error":"Incorrect API key provided"}`)}

	response, err := p.Send(context.Background(), func(string) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusBadRequest, Header: http.Header{}, Body: provider}, nil
	})

	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.True(provider.closed)
	s.Equal(map[string]float64{"400": 1}, s.counted(p.keyVars[0]))
}
