package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/stretchr/testify/suite"
)

type AdaptiveSuite struct{ suite.Suite }

func TestAdaptiveSuite(t *testing.T) { suite.Run(t, new(AdaptiveSuite)) }

func (s *AdaptiveSuite) TestABackchannelCanBecomeAnInterruptionWithoutRestartingTheWindow() {
	state := &adaptiveSpeaker{}
	now := time.Now()
	first, serial := state.append("reply", true, make([]int16, 3200), now)
	s.Len(first, 6400)
	state.inflight = false // completed prediction below threshold
	next, nextSerial := state.append("reply", true, make([]int16, 1600), now.Add(100*time.Millisecond))
	s.Equal(serial, nextSerial)
	s.Len(next, 9600)
}

func (s *AdaptiveSuite) TestAQuietGapRetiresAnInflightPrediction() {
	state := &adaptiveSpeaker{}
	now := time.Now()
	_, serial := state.append("reply", true, make([]int16, 3200), now)
	body, fresh := state.append("reply", true, make([]int16, 3200), now.Add(time.Second))
	s.NotEqual(serial, fresh)
	s.NotEmpty(body)
	_, newer := state.append("next-reply", true, make([]int16, 3200), now.Add(1100*time.Millisecond))
	s.NotEqual(fresh, newer)
}

func (s *AdaptiveSuite) TestAnUnavailableEndpointAndWrongResponseIdentityFallBack() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"probability": .99, "request_id": "another-utterance"})
	}))
	client, err := NewInterruptionClient(server.URL, "", .5, time.Second)
	s.Require().NoError(err)
	stop, err := client.score(context.Background(), "current", make([]byte, 6400))
	s.Error(err)
	s.False(stop)
	server.Close()
	stop, err = client.score(context.Background(), "current", make([]byte, 6400))
	s.Error(err)
	s.False(stop)
}

func (s *AdaptiveSuite) TestASlowEndpointTimesOutWithoutTakingTheFloor() {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(time.Second):
		}
	}))
	defer server.Close()
	client, err := NewInterruptionClient(server.URL, "", .5, 20*time.Millisecond)
	s.Require().NoError(err)
	stop, err := client.score(context.Background(), "current", make([]byte, 6400))
	s.Error(err)
	s.False(stop)
}

func (s *AgentSuite) adaptiveReply(handler http.Handler) (stt.Participant, string) {
	server := httptest.NewServer(handler)
	s.T().Cleanup(server.Close)
	client, err := NewInterruptionClient(server.URL, "", .5, time.Second)
	s.Require().NoError(err)
	s.adaptiveClient = client
	zero := time.Duration(0)
	s.replySilence = &zero
	s.join(true)
	setStubVoiceSilent(s.voice)
	participant := stt.Participant{ID: "caller", UserID: "caller", Name: "Caller"}
	s.speak(participant)
	s.says(participant, "Tell me about dinner")
	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "reply did not start")
	turn := s.model.requests()[0].ID
	s.synthesises(turn)
	s.eventually(func() bool { return len(s.edge.heard()) > 0 }, "audio did not reach playback")
	return participant, turn
}

func (s *AgentSuite) overlappingAudio(participant stt.Participant) {
	pcm := make([]int16, 3200)
	for i := range pcm {
		pcm[i] = 8000
	}
	s.edge.inbound <- InboundAudio{Participant: participant, Audio: audio.PcmData{Samples: pcm, SampleRate: 16000, Channels: 1}}
}

func (s *AgentSuite) TestAdaptiveAudioStopsPlaybackBeforeATranscriptArrives() {
	p, turn := s.adaptiveReply(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"probability": .9, "request_id": r.Header.Get("X-Request-ID")})
	}))
	s.overlappingAudio(p)
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) > 0 }, "audio did not stop playback")
	s.False(s.agent.speaking(turn))
	s.Empty(s.edge.heard())
}

func (s *AgentSuite) TestAdaptiveBackchannelLeavesPlaybackRunningAndLaterSpeechCanInterrupt() {
	var requests atomic.Int32
	p, turn := s.adaptiveReply(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probability := .1
		if requests.Add(1) > 1 {
			probability = .9
		}
		json.NewEncoder(w).Encode(map[string]any{"probability": probability, "request_id": r.Header.Get("X-Request-ID")})
	}))
	s.overlappingAudio(p)
	s.eventually(func() bool {
		s.agent.adaptive.mu.Lock()
		defer s.agent.adaptive.mu.Unlock()
		state := s.agent.adaptive.speakers[p.ID]
		return state != nil && state.scored > 0 && !state.inflight
	}, "first prediction did not settle")
	s.True(s.agent.speaking(turn))
	s.NotEmpty(s.edge.heard())
	s.Zero(countOf[Interrupted](s.reported()))
	s.overlappingAudio(p)
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) > 0 }, "later speech could not interrupt")
	s.False(s.agent.speaking(turn))
}

func (s *AgentSuite) TestALateAdaptiveResultCannotInterruptANewerReply() {
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	s.T().Cleanup(func() { once.Do(func() { close(release) }) })
	p, turn := s.adaptiveReply(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(entered)
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"probability": .9, "request_id": r.Header.Get("X-Request-ID")})
	}))
	s.overlappingAudio(p)
	select {
	case <-entered:
	case <-time.After(settleFor):
		s.FailNow("prediction did not start")
	}
	s.agent.Interrupt()
	s.eventually(func() bool { return countOf[Interrupted](s.reported()) > 0 }, "first reply did not stop")
	s.says(p, "Tell me about lunch instead")
	s.eventually(func() bool { return len(s.model.requests()) > 1 }, "new reply did not start")
	newTurn := s.model.requests()[len(s.model.requests())-1].ID
	s.NotEqual(turn, newTurn)
	s.eventually(func() bool { return s.agent.speaking(newTurn) }, "new reply is not active")
	once.Do(func() { close(release) })
	s.eventually(func() bool {
		s.agent.adaptive.mu.Lock()
		defer s.agent.adaptive.mu.Unlock()
		return !s.agent.adaptive.speakers[p.ID].inflight
	}, "old prediction did not settle")
	s.True(s.agent.speaking(newTurn))
	s.Equal(1, countOf[Interrupted](s.reported()))
}
