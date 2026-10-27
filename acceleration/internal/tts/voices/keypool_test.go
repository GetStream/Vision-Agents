package voices

import (
	"context"
	"crypto/rand"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"
)

// KeyPoolSuite proves the voice clients send on the key pool the speech providers share, so a
// key a vendor refuses on a voice request is counted as it is on a call. Every test has keys
// of its own, new on every run: pools and the counter are shared by the whole process.
type KeyPoolSuite struct {
	suite.Suite
	ctx    context.Context
	server *httptest.Server

	mu sync.Mutex
	// statuses is what a key is answered with other than a 200.
	statuses map[string]int
	// keys is the key each request carried, in order, from whichever header the vendor reads.
	keys []string
	// owners is the key each voice was cloned with on the ElevenLabs paths: only that key
	// finds and deletes it, as only its account holds it.
	owners map[string]string
}

func TestKeyPoolSuite(t *testing.T) {
	suite.Run(t, new(KeyPoolSuite))
}

func (s *KeyPoolSuite) SetupTest() {
	s.ctx = context.Background()
	s.statuses = map[string]int{}
	s.keys = nil
	s.owners = map[string]string{}
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Each path reads the key only where its vendor does, so a key put anywhere else is
		// not recognised as one.
		key := r.Header.Get("Authorization")
		switch {
		case strings.HasPrefix(r.URL.Path, "/v1/voices"):
			key = r.Header.Get("xi-api-key")
		case r.URL.Path == "/voices", r.URL.Path == "/model":
			key = strings.TrimPrefix(key, "Bearer ")
		case r.URL.Path == "/voices/v1/voices":
			key = strings.TrimPrefix(key, "Basic ")
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.keys = append(s.keys, key)

		if status := s.statuses[key]; status != 0 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"detail":"invalid api key"}`))
			return
		}
		switch r.URL.Path {
		case "/v1/voices/add":
			id := fmt.Sprintf("voice%015d", len(s.owners)+1)
			s.owners[id] = key
			_, _ = w.Write([]byte(`{"voice_id":"` + id + `"}`))
			return
		}
		if id, ok := strings.CutPrefix(r.URL.Path, "/v1/voices/"); ok && r.Method == http.MethodDelete {
			if s.owners[id] != key {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			delete(s.owners, id)
			_, _ = w.Write([]byte(`{}`))
			return
		}
		switch r.URL.Path {
		case "/v1/voices":
			_, _ = w.Write([]byte(`{"voices":[{"voice_id":"abc","name":"Rachel","category":"premade"}]}`))
		case "/voices":
			_, _ = w.Write([]byte(`{"data":[{"id":"1","name":"Aadhya"}],"has_more":false}`))
		case "/voices/v1/voices":
			_, _ = w.Write([]byte(`{"voices":[{"voiceId":"Ashley","displayName":"Ashley"}]}`))
		case "/model":
			_, _ = w.Write([]byte(`{"_id":"fish-voice"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	s.T().Cleanup(s.server.Close)
}

// newKeys returns n keys no other test or run has used.
func (s *KeyPoolSuite) newKeys(n int) []string {
	keys := make([]string, n)
	for i := range keys {
		keys[i] = rand.Text()
	}
	return keys
}

func (s *KeyPoolSuite) refuse(status int, keys ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		s.statuses[key] = status
	}
}

func (s *KeyPoolSuite) keysSeen() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.keys...)
}

// refusals is the counter's series under keyVar, as status → count.
func (s *KeyPoolSuite) refusals(keyVar string) map[string]float64 {
	families, err := prometheus.DefaultGatherer.Gather()
	s.Require().NoError(err)
	counted := map[string]float64{}
	for _, family := range families {
		if family.GetName() != "router_keypool_refused_attempts_total" {
			continue
		}
		for _, metric := range family.GetMetric() {
			labels := map[string]string{}
			for _, pair := range metric.GetLabel() {
				labels[pair.GetName()] = pair.GetValue()
			}
			if labels["key_var"] == keyVar {
				counted[labels["status"]] = metric.GetCounter().GetValue()
			}
		}
	}
	return counted
}

// added is how much each status under keyVar rose from before, leaving out the ones that did
// not.
func added(before, after map[string]float64) map[string]float64 {
	rose := map[string]float64{}
	for status, count := range after {
		if count != before[status] {
			rose[status] = count - before[status]
		}
	}
	return rose
}

func (s *KeyPoolSuite) TestAnElevenLabsListRefusedWithA401IsCountedOnce() {
	key := s.newKeys(1)[0]
	s.refuse(http.StatusUnauthorized, key)
	lister, err := NewElevenLabs(ElevenLabsOptions{APIKey: key, BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := s.refusals("ELEVENLABS_API_KEY")

	_, err = lister.List(s.ctx)

	s.Require().Error(err)
	s.Contains(err.Error(), "the elevenlabs key cannot list voices (http 401)")
	s.Equal(map[string]float64{"401": 1}, added(before, s.refusals("ELEVENLABS_API_KEY")))
	s.Equal([]string{key}, s.keysSeen())
}

func (s *KeyPoolSuite) TestACartesiaListRefusedWithA401IsCountedOnce() {
	key := s.newKeys(1)[0]
	s.refuse(http.StatusUnauthorized, key)
	lister, err := NewCartesia(CartesiaOptions{APIKey: key, BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := s.refusals("CARTESIA_API_KEY")

	_, err = lister.List(s.ctx)

	s.Require().Error(err)
	s.Contains(err.Error(), "the cartesia key cannot list voices (http 401)")
	s.Equal(map[string]float64{"401": 1}, added(before, s.refusals("CARTESIA_API_KEY")))
}

func (s *KeyPoolSuite) TestAnInworldListRefusedWithA401IsCountedOnce() {
	key := s.newKeys(1)[0]
	s.refuse(http.StatusUnauthorized, key)
	lister, err := NewInworld(InworldOptions{APIKey: key, BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := s.refusals("INWORLD_API_KEY")

	_, err = lister.List(s.ctx)

	s.Require().Error(err)
	s.Contains(err.Error(), "the inworld key cannot list voices (http 401)")
	s.Equal(map[string]float64{"401": 1}, added(before, s.refusals("INWORLD_API_KEY")))
}

func (s *KeyPoolSuite) TestAFishCloneRefusedWithA401IsCountedOnce() {
	key := s.newKeys(1)[0]
	s.refuse(http.StatusUnauthorized, key)
	cloner, err := NewFish(FishOptions{APIKey: key, BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := s.refusals("FISH_API_KEY")

	_, err = cloner.Prepare(s.ctx, Request{Name: "Mine", Samples: []Sample{{Name: "a.wav", Content: []byte("RIFF")}}})

	s.Require().Error(err)
	s.Contains(err.Error(), "http 401")
	s.Equal(map[string]float64{"401": 1}, added(before, s.refusals("FISH_API_KEY")))
}

func (s *KeyPoolSuite) TestListsAnsweredWithA200AreNotCountedAndReadAsBefore() {
	keys := s.newKeys(3)
	elevenLabs, err := NewElevenLabs(ElevenLabsOptions{APIKey: keys[0], BaseURL: s.server.URL})
	s.Require().NoError(err)
	cartesia, err := NewCartesia(CartesiaOptions{APIKey: keys[1], BaseURL: s.server.URL})
	s.Require().NoError(err)
	inworld, err := NewInworld(InworldOptions{APIKey: keys[2], BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := map[string]map[string]float64{}
	for _, keyVar := range []string{"ELEVENLABS_API_KEY", "CARTESIA_API_KEY", "INWORLD_API_KEY"} {
		before[keyVar] = s.refusals(keyVar)
	}

	fromElevenLabs, err := elevenLabs.List(s.ctx)
	s.Require().NoError(err)
	fromCartesia, err := cartesia.List(s.ctx)
	s.Require().NoError(err)
	fromInworld, err := inworld.List(s.ctx)
	s.Require().NoError(err)

	s.Equal([]Library{{ID: "abc", Name: "Rachel"}}, fromElevenLabs)
	s.Equal([]Library{{ID: "1", Name: "Aadhya"}}, fromCartesia)
	s.Equal([]Library{{ID: "Ashley", Name: "Ashley", Preview: true}}, fromInworld)
	for keyVar, counted := range before {
		s.Empty(added(counted, s.refusals(keyVar)), keyVar)
	}
	s.Equal(keys, s.keysSeen(), "each list goes out with the key it was given, in the header its vendor reads")
}

func (s *KeyPoolSuite) TestEachClientReadsItsKeysFromTheVariableItsProviderReads() {
	keys := s.newKeys(4)
	s.T().Setenv("ELEVENLABS_API_KEY", keys[0])
	s.T().Setenv("CARTESIA_API_KEY", keys[1])
	s.T().Setenv("INWORLD_API_KEY", keys[2])
	s.T().Setenv("FISH_API_KEY", keys[3])
	elevenLabs, err := NewElevenLabs(ElevenLabsOptions{BaseURL: s.server.URL})
	s.Require().NoError(err)
	cartesia, err := NewCartesia(CartesiaOptions{BaseURL: s.server.URL})
	s.Require().NoError(err)
	inworld, err := NewInworld(InworldOptions{BaseURL: s.server.URL})
	s.Require().NoError(err)
	fish, err := NewFish(FishOptions{BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = elevenLabs.List(s.ctx)
	s.Require().NoError(err)
	_, err = cartesia.List(s.ctx)
	s.Require().NoError(err)
	_, err = inworld.List(s.ctx)
	s.Require().NoError(err)
	_, err = fish.Prepare(s.ctx, Request{Name: "Mine", Samples: []Sample{{Name: "a.wav", Content: []byte("RIFF")}}})
	s.Require().NoError(err)

	s.Equal(keys, s.keysSeen())
}

// A 429 on the first key fails the list rather than moving to another account's key, and
// rests that key for the calls that share the pool.
func (s *KeyPoolSuite) TestAListRateLimitedOnTheFirstKeyFailsAndRestsItForCalls() {
	keys := s.newKeys(2)
	s.refuse(http.StatusTooManyRequests, keys[0])
	joined := keys[0] + "," + keys[1]
	lister, err := NewElevenLabs(ElevenLabsOptions{APIKey: joined, BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := s.refusals("ELEVENLABS_API_KEY")

	_, err = lister.List(s.ctx)
	s.Require().Error(err)
	_, err = lister.List(s.ctx)
	s.Require().Error(err)

	s.Contains(err.Error(), "http 429")
	s.Equal([]string{keys[0], keys[0]}, s.keysSeen(), "each list is sent once, on the first key")
	s.Equal(map[string]float64{"429": 2}, added(before, s.refusals("ELEVENLABS_API_KEY")))
	s.Equal(keys[1], keypool.FromEnv(joined).Next(), "a call skips the key the list rested")
}

// A voice is cloned into the account of the key that made it, and only that account can
// find or delete it, so every voice request goes on the first key: after a call took a turn
// on the shared pool too, and when the next key is one the vendor refuses.
func (s *KeyPoolSuite) TestEveryVoiceRequestGoesOnTheFirstKey() {
	keys := s.newKeys(2)
	s.refuse(http.StatusUnauthorized, keys[1])
	joined := keys[0] + "," + keys[1]
	keypool.FromEnv(joined).Next()
	lister, err := NewElevenLabs(ElevenLabsOptions{APIKey: joined, BaseURL: s.server.URL})
	s.Require().NoError(err)
	before := s.refusals("ELEVENLABS_API_KEY")

	for range 3 {
		_, err := lister.List(s.ctx)
		s.Require().NoError(err)
	}

	s.Equal([]string{keys[0], keys[0], keys[0]}, s.keysSeen())
	s.Empty(added(before, s.refusals("ELEVENLABS_API_KEY")))
}

// The router builds the library and the cloner as two clients (cmd/router/main.go); a voice
// one lists is the voice the other cloned only when both send on the same key.
func (s *KeyPoolSuite) TestTheListerAndTheClonerSendOnTheSameKey() {
	keys := s.newKeys(2)
	s.T().Setenv("ELEVENLABS_API_KEY", keys[0])
	s.T().Setenv("ELEVENLABS_API_KEY_1", keys[1])
	lister, err := NewElevenLabs(ElevenLabsOptions{BaseURL: s.server.URL})
	s.Require().NoError(err)
	cloner, err := NewElevenLabs(ElevenLabsOptions{BaseURL: s.server.URL})
	s.Require().NoError(err)

	_, err = lister.List(s.ctx)
	s.Require().NoError(err)
	_, err = cloner.Prepare(s.ctx, Request{Name: "Mine", Samples: []Sample{{Name: "a.wav", Content: []byte("RIFF")}}})
	s.Require().NoError(err)

	s.Equal([]string{keys[0], keys[0]}, s.keysSeen())
}

// A call rate limited on the first key rests it for the pool; a cloned voice is still
// deleted on that key, where it lives, rather than reported gone by another account.
func (s *KeyPoolSuite) TestAVoiceIsDeletedOnItsKeyWhileACallRestsIt() {
	keys := s.newKeys(2)
	joined := keys[0] + "," + keys[1]
	cloner, err := NewElevenLabs(ElevenLabsOptions{APIKey: joined, BaseURL: s.server.URL})
	s.Require().NoError(err)
	prepared, err := cloner.Prepare(s.ctx, Request{Name: "Mine", Samples: []Sample{{Name: "a.wav", Content: []byte("RIFF")}}})
	s.Require().NoError(err)
	keypool.FromEnv(joined).Cooldown(keys[0], 30*time.Second)

	s.Require().NoError(cloner.Delete(s.ctx, prepared.ExternalID))

	s.mu.Lock()
	defer s.mu.Unlock()
	s.Empty(s.owners, "the voice is gone from the account that held it")
	s.Equal([]string{keys[0], keys[0]}, s.keys)
}
