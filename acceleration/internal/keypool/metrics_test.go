package keypool

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/gorilla/websocket"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"
)

// RefusedAttemptsSuite proves each attempt a provider refuses on a pooled key is counted
// once, under the pool's provider, and that counting leaves the turn of the keys as it was.
// Every test has keys and a variable of its own: pools and the counter are shared by the
// whole process.
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

// pool is the pool FromEnv builds for keys read from the variable provider.
func (s *RefusedAttemptsSuite) pool(provider, keys string) *Pool {
	s.T().Setenv(provider, keys)
	return FromEnv("", provider)
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

// refusals is every series of the counter, as provider → status → count.
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
			if counted[labels["provider"]] == nil {
				counted[labels["provider"]] = map[string]float64{}
			}
			counted[labels["provider"]][labels["status"]] = metric.GetCounter().GetValue()
		}
	}
	return counted
}

func (s *RefusedAttemptsSuite) counted(provider string) map[string]float64 {
	return refusals(s.T())[provider]
}

// The keys seen and the next key in these tests are what the base before the counter
// answered to the same requests.

func (s *RefusedAttemptsSuite) TestARefusalAnotherKeyGetsPastIsCountedOnceAndTheCallSucceeds() {
	pool := s.pool("RA_PASSED_API_KEY", "ra-passed-1,ra-passed-2")
	s.answer(http.StatusTooManyRequests, "ra-passed-1")

	s.Equal(http.StatusOK, s.post(pool))

	s.Equal([]string{"ra-passed-1", "ra-passed-2"}, s.keysSeen())
	s.Equal("ra-passed-2", pool.Next())
	s.Equal(map[string]float64{"429": 1}, s.counted("RA_PASSED_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestEveryRefusedAttemptIsCountedWhenNoKeyIsLeft() {
	pool := s.pool("RA_EXHAUSTED_API_KEY", "ra-exhausted-1,ra-exhausted-2")
	s.answer(http.StatusTooManyRequests, "ra-exhausted-1", "ra-exhausted-2")

	s.Equal(http.StatusTooManyRequests, s.post(pool))

	s.Equal([]string{"ra-exhausted-1", "ra-exhausted-2"}, s.keysSeen())
	s.Equal("ra-exhausted-1", pool.Next())
	s.Equal(map[string]float64{"429": 2}, s.counted("RA_EXHAUSTED_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestAnAnsweredRequestCountsNoRefusal() {
	pool := s.pool("RA_ANSWERED_API_KEY", "ra-answered-1,ra-answered-2")

	s.Equal(http.StatusOK, s.post(pool))

	s.Equal([]string{"ra-answered-1"}, s.keysSeen())
	s.Equal("ra-answered-2", pool.Next())
	s.Empty(s.counted("RA_ANSWERED_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestAServerErrorCountsNoRefusal() {
	pool := s.pool("RA_FAILING_API_KEY", "ra-failing-1,ra-failing-2")
	s.answer(http.StatusInternalServerError, "ra-failing-1")

	s.Equal(http.StatusInternalServerError, s.post(pool))

	s.Equal([]string{"ra-failing-1"}, s.keysSeen())
	s.Equal("ra-failing-2", pool.Next())
	s.Empty(s.counted("RA_FAILING_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestASpentQuotaIsCountedUnderItsStatus() {
	pool := s.pool("RA_SPENT_API_KEY", "ra-spent-1,ra-spent-2")
	s.answer(http.StatusPaymentRequired, "ra-spent-1")

	s.Equal(http.StatusOK, s.post(pool))

	s.Equal([]string{"ra-spent-1", "ra-spent-2"}, s.keysSeen())
	s.Equal("ra-spent-2", pool.Next())
	s.Equal(map[string]float64{"402": 1}, s.counted("RA_SPENT_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestAWrongKeyCountsNoRefusal() {
	pool := s.pool("RA_WRONG_API_KEY", "ra-wrong-1,ra-wrong-2")
	s.answer(http.StatusUnauthorized, "ra-wrong-1")

	s.Equal(http.StatusUnauthorized, s.post(pool))

	s.Equal([]string{"ra-wrong-1"}, s.keysSeen())
	s.Equal("ra-wrong-2", pool.Next())
	s.Empty(s.counted("RA_WRONG_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestARefusedHandshakeIsCountedOnceAndDialledOnTheNextKey() {
	pool := s.pool("RA_DIALLED_API_KEY", "ra-dialled-1,ra-dialled-2")
	s.answer(http.StatusTooManyRequests, "ra-dialled-1")

	response, key, err := s.dial(pool)

	s.Require().NoError(err)
	s.Equal(http.StatusSwitchingProtocols, response.StatusCode)
	s.Equal("ra-dialled-2", key)
	s.Equal([]string{"ra-dialled-1", "ra-dialled-2"}, s.keysSeen())
	s.Equal("ra-dialled-2", pool.Next())
	s.Equal(map[string]float64{"429": 1}, s.counted("RA_DIALLED_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestEveryRefusedHandshakeIsCountedWhenNoKeyIsLeft() {
	pool := s.pool("RA_UNDIALLED_API_KEY", "ra-undialled-1,ra-undialled-2")
	s.answer(http.StatusTooManyRequests, "ra-undialled-1", "ra-undialled-2")

	response, key, err := s.dial(pool)

	s.Require().Error(err)
	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal("ra-undialled-2", key)
	s.Equal([]string{"ra-undialled-1", "ra-undialled-2"}, s.keysSeen())
	s.Equal("ra-undialled-1", pool.Next())
	s.Equal(map[string]float64{"429": 2}, s.counted("RA_UNDIALLED_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestARefusalIsCountedUnderItsVariableAndNeverItsKey() {
	pool := s.pool("RA_LABELLED_API_KEY", "ra-secret-1,ra-secret-2")
	s.answer(http.StatusTooManyRequests, "ra-secret-1", "ra-secret-2")

	s.post(pool)

	counted := refusals(s.T())
	s.Equal(map[string]float64{"429": 2}, counted["RA_LABELLED_API_KEY"])
	for provider, statuses := range counted {
		s.NotContains(provider, "ra-secret")
		for status := range statuses {
			s.NotContains(status, "ra-secret")
		}
	}
}

func (s *RefusedAttemptsSuite) TestAPoolNoVariableNamesIsCountedAsUnknown() {
	pool := New("ra-unnamed-1,ra-unnamed-2")
	s.answer(http.StatusTooManyRequests, "ra-unnamed-1")
	before := s.counted("unknown")["429"]

	s.Equal(http.StatusOK, s.post(pool))

	s.Equal(before+1, s.counted("unknown")["429"])
}

func (s *RefusedAttemptsSuite) TestAPoolOfKeysPassedInTakesTheVariableTheSameKeysAreReadFrom() {
	passed := FromEnv("ra-adopted-1,ra-adopted-2")
	s.pool("RA_ADOPTED_API_KEY", "ra-adopted-1,ra-adopted-2")
	s.answer(http.StatusTooManyRequests, "ra-adopted-1")

	s.Equal(http.StatusOK, s.post(passed))

	s.Equal(map[string]float64{"429": 1}, s.counted("RA_ADOPTED_API_KEY"))
}

func (s *RefusedAttemptsSuite) TestAPoolKeepsTheFirstVariableItIsNamedFor() {
	first := s.pool("RA_FIRST_API_KEY", "ra-first-1,ra-first-2")
	s.pool("RA_SECOND_API_KEY", "ra-first-1,ra-first-2")
	s.answer(http.StatusTooManyRequests, "ra-first-1")

	s.Equal(http.StatusOK, s.post(first))

	s.Equal(map[string]float64{"429": 1}, s.counted("RA_FIRST_API_KEY"))
	s.Empty(s.counted("RA_SECOND_API_KEY"))
}
