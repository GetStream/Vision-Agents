package keypool

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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
	// wrong key, 402 as a spent quota, any other as is. Every other key is answered, and a
	// websocket handshake on it accepted.
	server *httptest.Server

	mu       sync.Mutex
	statuses map[string]int
	seen     []string
}

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
		s.mu.Unlock()

		switch status {
		case 0:
			if websocket.IsWebSocketUpgrade(r) {
				if conn, err := upgrader.Upgrade(w, r, nil); err == nil {
					_ = conn.Close()
				}
				return
			}
			_, _ = io.WriteString(w, "answered")
		case http.StatusTooManyRequests:
			w.Header().Set("Retry-After", "10")
			w.WriteHeader(status)
		case http.StatusUnauthorized:
			w.WriteHeader(status)
			_, _ = io.WriteString(w, invalidKeyBody)
		case http.StatusPaymentRequired:
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
	s.seen = nil
}

func (s *RefusedAttemptsSuite) answer(status int, keys ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.statuses[key] = status
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
	request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("body"))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer placeholder")
	response, err := pool.WrapClient(nil, Bearer).Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
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
	s.Equal(p.keys[1], p.Next())
	s.Equal(map[string]float64{"402": 1}, s.counted(p.keyVars[0]))
}

func (s *RefusedAttemptsSuite) TestAWrongKeyCountsNoRefusal() {
	p := s.pool("wrong")
	s.answer(http.StatusUnauthorized, p.keys[0])

	s.Equal(http.StatusUnauthorized, s.post(p.Pool))

	s.Equal(p.keys[:1], s.keysSeen())
	s.Equal(p.keys[1], p.Next())
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
	s.NotContains(refused(start.Add(30*time.Minute)), item, "held back by the limit")

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
		{name: "wrong key", status: http.StatusUnauthorized},
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
