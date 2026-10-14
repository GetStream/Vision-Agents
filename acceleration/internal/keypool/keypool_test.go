package keypool

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/suite"
)

type KeyPoolSuite struct {
	suite.Suite

	// server answers 429 to the keys in refused, the status in spent with a body saying the
	// quota is exceeded, 401 to the keys in invalid, and echoes the body to every other.
	server *httptest.Server

	mu      sync.Mutex
	refused map[string]bool
	spent   map[string]int
	invalid map[string]bool
	// seen is the key on every request, in order, and bodies what each carried.
	seen   []string
	bodies []string
}

func TestKeyPoolSuite(t *testing.T) {
	suite.Run(t, new(KeyPoolSuite))
}

func (s *KeyPoolSuite) SetupSuite() {
	upgrader := websocket.Upgrader{}
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		body, _ := io.ReadAll(r.Body)

		s.mu.Lock()
		s.seen = append(s.seen, key)
		s.bodies = append(s.bodies, string(body))
		refused, spent, invalid := s.refused[key], s.spent[key], s.invalid[key]
		s.mu.Unlock()

		if spent != 0 {
			w.WriteHeader(spent)
			_, _ = io.WriteString(w, `{"detail":{"code":"quota_exceeded","message":"This request exceeds your API key quota."}}`)
			return
		}
		if invalid {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, invalidKeyBody)
			return
		}
		if refused {
			w.Header().Set("Retry-After", "10")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		if websocket.IsWebSocketUpgrade(r) {
			conn, err := upgrader.Upgrade(w, r, nil)
			if err == nil {
				_ = conn.Close()
			}
			return
		}
		_, _ = w.Write(body)
	}))
}

func (s *KeyPoolSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.refused = map[string]bool{}
	s.spent = map[string]int{}
	s.invalid = map[string]bool{}
	s.seen = nil
	s.bodies = nil
}

func (s *KeyPoolSuite) refuse(keys ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.refused[key] = true
	}
}

// invalidKeyBody is how a key that is wrong, rather than spent, is refused.
const invalidKeyBody = `{"detail":{"status":"invalid_api_key","message":"Invalid API key"}}`

func (s *KeyPoolSuite) spend(status int, keys ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.spent[key] = status
	}
}

func (s *KeyPoolSuite) keysSeen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.seen...)
}

func (s *KeyPoolSuite) post(client *http.Client, body string) *http.Response {
	request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer placeholder")
	response, err := client.Do(request)
	s.Require().NoError(err)
	return response
}

// clock is a pool whose time stands still until moved.
func clock(pool *Pool) *time.Time {
	now := time.Unix(1_700_000_000, 0)
	pool.now = func() time.Time { return now }
	return &now
}

func (s *KeyPoolSuite) TestParseSplitsOnCommasAndDropsBlanksAndRepeats() {
	s.Equal([]string{"a", "b", "c"}, Parse(" a , b,,a", "c, b "))
	s.Empty(Parse("", " , "))
}

func (s *KeyPoolSuite) TestEnvReadsTheVariableThenItsNumberedSiblingsInOrder() {
	s.T().Setenv("KEYPOOL_TEST_KEY", "k1,k2")
	s.T().Setenv("KEYPOOL_TEST_KEY_10", "k5")
	s.T().Setenv("KEYPOOL_TEST_KEY_2", "k4")
	s.T().Setenv("KEYPOOL_TEST_KEY_1", "k3")
	s.T().Setenv("KEYPOOL_TEST_KEY_X", "ignored")

	s.Equal([]string{"k1", "k2", "k3", "k4", "k5"}, Env("KEYPOOL_TEST_KEY"))
	s.Equal("k1,k2,k3,k4,k5", Getenv("KEYPOOL_TEST_KEY"))
}

func (s *KeyPoolSuite) TestEnvReadsNumberedKeysWithoutTheBareVariable() {
	s.T().Setenv("KEYPOOL_TEST_ONLY_1", "k1")
	s.T().Setenv("KEYPOOL_TEST_ONLY_2", "k2")

	s.Equal([]string{"k1", "k2"}, Env("KEYPOOL_TEST_ONLY"))
}

func (s *KeyPoolSuite) TestEnvFallsBackToTheNextNameOnlyWhenTheFirstHoldsNothing() {
	s.T().Setenv("KEYPOOL_TEST_SECOND", "k2")

	s.Equal([]string{"k2"}, Env("KEYPOOL_TEST_FIRST", "KEYPOOL_TEST_SECOND"))
	s.T().Setenv("KEYPOOL_TEST_FIRST", "k1")
	s.Equal([]string{"k1"}, Env("KEYPOOL_TEST_FIRST", "KEYPOOL_TEST_SECOND"))
}

func (s *KeyPoolSuite) TestKeysAreTakenInTurn() {
	pool := New("k1,k2,k3")

	s.Equal([]string{"k1", "k2", "k3", "k1"}, []string{pool.Next(), pool.Next(), pool.Next(), pool.Next()})
}

func (s *KeyPoolSuite) TestARestingKeyIsSkippedUntilItsCooldownEnds() {
	pool := New("k1,k2,k3")
	now := clock(pool)

	pool.Cooldown("k2", time.Minute)
	s.Equal([]string{"k1", "k3", "k1", "k3"}, []string{pool.Next(), pool.Next(), pool.Next(), pool.Next()})

	*now = now.Add(time.Minute)
	s.Equal([]string{"k1", "k2", "k3"}, []string{pool.Next(), pool.Next(), pool.Next()})
}

func (s *KeyPoolSuite) TestWhenEveryKeyRestsTheOneThatRecoversFirstIsTaken() {
	pool := New("k1,k2,k3")
	clock(pool)

	pool.Cooldown("k1", 3*time.Minute)
	pool.Cooldown("k2", time.Minute)
	pool.Cooldown("k3", 2*time.Minute)

	s.Equal("k2", pool.Next())
}

func (s *KeyPoolSuite) TestPinsTakeTheirKeysInTurnAndKeepToThem() {
	pool := New("k1,k2")
	first, second := pool.Pin(), pool.Pin()

	s.Equal([]string{"k1", "k1"}, []string{first.Next(), first.Next()})
	s.Equal([]string{"k2", "k2"}, []string{second.Next(), second.Next()})
}

func (s *KeyPoolSuite) TestAPinMovesOnOnlyOnceItsKeyRests() {
	pool := New("k1,k2,k3")
	clock(pool)
	pinned := pool.Pin()
	s.Equal("k1", pinned.Next())

	pinned.Cooldown("k1", time.Minute)

	s.Equal([]string{"k2", "k2"}, []string{pinned.Next(), pinned.Next()})
	s.Equal("k3", pool.Next(), "the pin shares the pool's turn")
}

func (s *KeyPoolSuite) TestSendGivesTheNextKeyToARequestRefusedWith429() {
	s.refuse("k1")
	pool := New("k1,k2")

	response, err := pool.Send(context.Background(), func(key string) (*http.Response, error) {
		request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("body-for-"+key))
		s.Require().NoError(err)
		request.Header.Set("Authorization", "Bearer "+key)
		return http.DefaultClient.Do(request)
	})
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"body-for-k1", "body-for-k2"}, s.bodies)
}

func (s *KeyPoolSuite) TestACooldownIsCappedAndNeverShortened() {
	pool := New("k1,k2")
	now := clock(pool)

	pool.Cooldown("k1", 24*time.Hour)
	s.Equal(now.Add(MaxCooldown), pool.until[0])
	pool.Cooldown("k1", time.Second)
	s.Equal(now.Add(MaxCooldown), pool.until[0])
}

func (s *KeyPoolSuite) TestRetryAfterSaysHowLongAKeyRests() {
	pool := New("k1")
	now := clock(pool)

	seconds := &http.Response{Header: http.Header{"Retry-After": {"7"}}}
	s.Equal(7*time.Second, pool.retryAfter(seconds))

	millis := &http.Response{Header: http.Header{"Retry-After-Ms": {"1500"}}}
	s.Equal(1500*time.Millisecond, pool.retryAfter(millis))

	date := &http.Response{Header: http.Header{"Retry-After": {now.Add(time.Minute).UTC().Format(http.TimeFormat)}}}
	s.Equal(time.Minute, pool.retryAfter(date))

	s.Equal(DefaultCooldown, pool.retryAfter(&http.Response{Header: http.Header{}}))
}

func (s *KeyPoolSuite) TestARefusedRequestIsSentAgainOnTheNextKeyWithItsBody() {
	s.refuse("k1")
	pool := New("k1,k2")
	client := pool.WrapClient(nil, Bearer)

	response := s.post(client, "hello")
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	s.Require().NoError(err)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("hello", string(body))
	s.Equal([]string{"k1", "k2"}, s.keysSeen())
	s.Equal([]string{"hello", "hello"}, s.bodies)
}

func (s *KeyPoolSuite) TestARefusedKeyIsNotTriedAgainWhileItRests() {
	s.refuse("k1")
	pool := New("k1,k2")
	client := pool.WrapClient(nil, Bearer)

	s.post(client, "one").Body.Close()
	s.post(client, "two").Body.Close()
	s.post(client, "three").Body.Close()

	s.Equal([]string{"k1", "k2", "k2", "k2"}, s.keysSeen())
}

func (s *KeyPoolSuite) TestAKeyWhoseQuotaIsSpentHandsTheRequestToTheNextKey() {
	for _, status := range []int{http.StatusUnauthorized, http.StatusPaymentRequired} {
		s.SetupTest()
		s.spend(status, "k1")
		pool := New("k1,k2")
		now := clock(pool)
		client := pool.WrapClient(nil, Bearer)

		response := s.post(client, "hello")
		body, err := io.ReadAll(response.Body)
		response.Body.Close()
		s.Require().NoError(err)

		s.Equal(http.StatusOK, response.StatusCode, "status %d", status)
		s.Equal("hello", string(body))
		s.Equal([]string{"k1", "k2"}, s.keysSeen())

		*now = now.Add(MaxCooldown - time.Second)
		s.Equal("k2", pool.Next(), "a spent key rests for as long as a key can")
		s.Equal("k2", pool.Next())
		*now = now.Add(time.Second)
		s.Equal([]string{"k1", "k2"}, []string{pool.Next(), pool.Next()})
	}
}

func (s *KeyPoolSuite) TestAWrongKeyIsReportedAsItCameAndNotRested() {
	s.mu.Lock()
	s.invalid["k1"] = true
	s.mu.Unlock()
	pool := New("k1,k2")
	client := pool.WrapClient(nil, Bearer)

	response := s.post(client, "hello")
	body, err := io.ReadAll(response.Body)
	response.Body.Close()
	s.Require().NoError(err)

	s.Equal(http.StatusUnauthorized, response.StatusCode)
	s.Equal(invalidKeyBody, string(body), "the refusal reaches the caller whole")
	s.Equal([]string{"k1"}, s.keysSeen())
	s.Equal("k2", pool.Next())
	s.Equal("k1", pool.Next(), "a wrong key is the operator's to fix, not the pool's to rest")
}

func (s *KeyPoolSuite) TestAHandshakeOnASpentKeyIsDialledAgainOnTheNextKey() {
	s.spend(http.StatusUnauthorized, "k1")
	pool := New("k1,k2")
	endpoint := "ws://" + strings.TrimPrefix(s.server.URL, "http://")

	conn, _, key, err := pool.Dial(s.T().Context(), websocket.DefaultDialer, func(key string) (string, http.Header) {
		return endpoint, http.Header{"Authorization": {"Bearer " + key}}
	})
	s.Require().NoError(err)
	conn.Close()

	s.Equal("k2", key)
	s.Equal([]string{"k1", "k2"}, s.keysSeen())
}

func (s *KeyPoolSuite) TestWhenEveryKeyIsRefusedThe429ReachesTheCallerWithoutAWait() {
	s.refuse("k1", "k2")
	pool := New("k1,k2")
	client := pool.WrapClient(nil, Bearer)

	response := s.post(client, "hello")
	defer response.Body.Close()

	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Empty(response.Header.Get("Retry-After"))
	s.Equal([]string{"k1", "k2"}, s.keysSeen())
}

func (s *KeyPoolSuite) TestASingleKeyRefusalKeepsItsRetryAfter() {
	s.refuse("only")
	client := New("only").WrapClient(nil, Bearer)

	response := s.post(client, "hello")
	defer response.Body.Close()

	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal("10", response.Header.Get("Retry-After"))
}

func (s *KeyPoolSuite) TestAWrappedDoerRotatesLikeAWrappedClient() {
	s.refuse("k1")
	doer := New("k1,k2").WrapDoer(nil, Bearer)

	request, err := http.NewRequest(http.MethodPost, s.server.URL, strings.NewReader("hello"))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer placeholder")
	response, err := doer.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{"k1", "k2"}, s.keysSeen())
}

func (s *KeyPoolSuite) TestAnEmptyPoolSendsTheRequestAsItCame() {
	client := New().WrapClient(nil, Bearer)

	request, err := http.NewRequest(http.MethodGet, s.server.URL, nil)
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer mine")
	response, err := client.Do(request)
	s.Require().NoError(err)
	response.Body.Close()

	s.Equal([]string{"mine"}, s.keysSeen())
}

func (s *KeyPoolSuite) TestARequestThatCarriesNoKeyIsSentWithoutOne() {
	s.refuse("k1")
	client := New("k1,k2").WrapClient(nil, Bearer)

	request, err := http.NewRequest(http.MethodGet, s.server.URL, nil)
	s.Require().NoError(err)
	response, err := client.Do(request)
	s.Require().NoError(err)
	response.Body.Close()

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal([]string{""}, s.keysSeen())
}

func (s *KeyPoolSuite) TestAQueryKeyGoesOnTheQueryString() {
	request, err := http.NewRequest(http.MethodGet, "https://example.com/path?a=1", nil)
	s.Require().NoError(err)

	Query("key").set(request, "k1")

	s.Equal("1", request.URL.Query().Get("a"))
	s.Equal("k1", request.URL.Query().Get("key"))
}

func (s *KeyPoolSuite) TestProvidersBuiltFromTheSameKeysShareAPool() {
	s.T().Setenv("KEYPOOL_TEST_SHARED", "shared-1,shared-2")

	fromEnv := FromEnv("", "KEYPOOL_TEST_SHARED")
	explicit := FromEnv("shared-1, shared-2")

	s.Same(fromEnv, explicit)
	s.NotSame(fromEnv, FromEnv("shared-1"))
}

func (s *KeyPoolSuite) TestARefusedHandshakeIsDialledAgainOnTheNextKey() {
	s.refuse("k1")
	pool := New("k1,k2")
	endpoint := "ws" + strings.TrimPrefix(s.server.URL, "http")

	conn, _, key, err := pool.Dial(context.Background(), websocket.DefaultDialer, func(key string) (string, http.Header) {
		return endpoint, http.Header{"Authorization": {"Bearer " + key}}
	})
	s.Require().NoError(err)
	defer conn.Close()

	s.Equal("k2", key)
	s.Equal([]string{"k1", "k2"}, s.keysSeen())
	s.Equal("k2", pool.Next())
}

func (s *KeyPoolSuite) TestAHandshakeEveryKeyRefusesFailsWithThe429() {
	s.refuse("k1", "k2")
	pool := New("k1,k2")
	endpoint := "ws" + strings.TrimPrefix(s.server.URL, "http")

	_, response, _, err := pool.Dial(context.Background(), websocket.DefaultDialer, func(key string) (string, http.Header) {
		return endpoint, http.Header{"Authorization": {"Bearer " + key}}
	})

	s.Require().Error(err)
	s.Equal(http.StatusTooManyRequests, response.StatusCode)
	s.Equal([]string{"k1", "k2"}, s.keysSeen())
}
