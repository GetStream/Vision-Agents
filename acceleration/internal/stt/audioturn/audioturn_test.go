package audioturn

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/stretchr/testify/suite"
)

type transcriptionReply struct {
	probability float64
	words       []Word
	unavailable bool
}

type AudioTurnSuite struct {
	suite.Suite
	provider *STT
	client   *Client
	replies  chan transcriptionReply
	requests chan []byte
}

func TestAudioTurnSuite(t *testing.T) { suite.Run(t, new(AudioTurnSuite)) }

func (s *AudioTurnSuite) SetupTest() {
	s.replies = make(chan transcriptionReply, 8)
	s.requests = make(chan []byte, 8)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pcm, err := io.ReadAll(r.Body)
		if err != nil {
			return
		}
		if r.URL.RawQuery != "transcript=true&transcript_min_p=0" {
			http.Error(w, "transcript query required", http.StatusBadRequest)
			return
		}
		var reply transcriptionReply
		if r.Header.Get("X-Request-ID") != "transcription-preflight" {
			s.requests <- pcm
			select {
			case reply = <-s.replies:
			case <-r.Context().Done():
				return
			}
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, eotJSON(r.Header.Get("X-Request-ID"), len(pcm)/2, reply.probability)+"\n")
		if reply.unavailable {
			return
		}
		text := make([]string, len(reply.words))
		for i, word := range reply.words {
			text[i] = word.Text
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"transcript": Transcript{Text: strings.Join(text, " "), Words: reply.words}})
	}))
	s.T().Cleanup(server.Close)
	var err error
	s.client, err = NewClient(server.URL, "")
	s.Require().NoError(err)
	s.provider, err = New(Options{Client: s.client})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = s.provider.Close() })
}

func (s *AudioTurnSuite) start() { s.Require().NoError(s.provider.Start(s.T().Context())) }

func (s *AudioTurnSuite) feed(seconds int) {
	pcm := stt.PcmData{Samples: make([]int16, seconds*SampleRate), SampleRate: SampleRate, Channels: 1}
	for i := range pcm.Samples {
		pcm.Samples[i] = int16(i)
	}
	s.Require().NoError(s.provider.ProcessAudio(pcm, stt.Participant{ID: "caller"}))
}

func (s *AudioTurnSuite) transcript() stt.Transcript {
	deadline := time.NewTimer(2 * time.Second)
	defer deadline.Stop()
	for {
		select {
		case event := <-s.provider.Events():
			if transcript, ok := event.(stt.Transcript); ok {
				return transcript
			}
			if failure, ok := event.(stt.Error); ok {
				s.FailNow("transcription failed", "%v", failure.Err)
			}
		case <-deadline.C:
			s.FailNow("no transcript arrived")
		}
	}
}

func (s *AudioTurnSuite) TestStableWordsSettleIndependentlyOfTheTurnScore() {
	s.start()
	s.replies <- transcriptionReply{probability: 0.1, words: []Word{{Text: "seven", StartMS: -500, EndMS: -100, Confidence: 0.95}}}
	s.feed(1)
	partial := s.transcript()
	s.Equal("seven", partial.Text)
	s.False(partial.Final())
	s.Equal(0.1, *partial.TurnProbability)

	s.replies <- transcriptionReply{probability: 0.9, words: []Word{
		{Text: "seven", StartMS: -1500, EndMS: -1100, Confidence: 0.95},
		{Text: "thirty", StartMS: -800, EndMS: -500, Confidence: 0.97},
	}}
	s.feed(1)
	revision := s.transcript()
	s.Equal("seven thirty", revision.Text)
	s.False(revision.Final(), "a high turn score must not commit changing words")
	s.Equal(0.9, *revision.TurnProbability)

	s.replies <- transcriptionReply{probability: 0.1, words: []Word{
		{Text: "seven", StartMS: -2500, EndMS: -2100, Confidence: 0.95},
		{Text: "thirty", StartMS: -1800, EndMS: -1500, Confidence: 0.97},
	}}
	s.feed(1)
	final := s.transcript()
	s.Equal("seven thirty", final.Text)
	s.True(final.Final(), "a low turn score must not strand settled words")
	s.Equal(partial.Utterance, final.Utterance)
	s.Equal(0.1, *final.TurnProbability)
	s.Equal(float64(3000), final.AudioDurationMs)

	s.replies <- transcriptionReply{probability: 0.95, words: []Word{
		{Text: "seven", StartMS: -3500, EndMS: -3100, Confidence: 0.95},
		{Text: "thirty", StartMS: -2800, EndMS: -2500, Confidence: 0.97},
		{Text: "seven", StartMS: -500, EndMS: -100, Confidence: 0.99},
	}}
	s.feed(1)
	next := s.transcript()
	s.Equal("seven", next.Text, "repeated words in a new turn are not a duplicate")
	s.Equal(final.Utterance+1, next.Utterance)
	s.Equal(float64(1000), next.AudioDurationMs)
	<-s.requests
	<-s.requests
	<-s.requests
	s.Len(<-s.requests, 4*SampleRate*2, "keep the audio context after settling")
}

func (s *AudioTurnSuite) TestLongSpeechKeepsItsPrefixAndRevisesTheOverlappingTail() {
	s.start()
	s.replies <- transcriptionReply{probability: 0.1, words: []Word{
		{Text: "first", StartMS: -11000, EndMS: -10800, Confidence: 0.9},
		{Text: "word", StartMS: -2000, EndMS: -1800, Confidence: 0.9},
		{Text: "wrong", StartMS: -800, EndMS: -300, Confidence: 0.7},
	}}
	s.feed(12)
	s.Equal("first word wrong", s.transcript().Text)
	s.replies <- transcriptionReply{probability: 0.9, words: []Word{
		{Text: "word", StartMS: -9980, EndMS: -9800, Confidence: 0.9},
		{Text: "corrected", StartMS: -8800, EndMS: -8300, Confidence: 0.98},
		{Text: "ending", StartMS: -1000, EndMS: -500, Confidence: 0.9},
	}}
	s.feed(8)
	revision := s.transcript()
	s.Equal("first word corrected ending", revision.Text)
	s.False(revision.Final())
	s.Equal(float64(20000), revision.AudioDurationMs)
	<-s.requests
	s.Len(<-s.requests, MaxSamples*2)
}

func (s *AudioTurnSuite) TestAudioArrivingDuringInferenceKeepsTheWordsInFlight() {
	s.start()
	s.feed(1)
	select {
	case <-s.requests:
	case <-time.After(time.Second):
		s.FailNow("no audio request arrived")
	}
	s.feed(1)
	s.replies <- transcriptionReply{probability: 0.9, words: []Word{{Text: "hello", StartMS: -500, EndMS: -100}}}
	s.Equal("hello", s.transcript().Text)
	s.replies <- transcriptionReply{probability: 0.9, words: []Word{
		{Text: "hello", StartMS: -1500, EndMS: -1100},
		{Text: "again", StartMS: -500, EndMS: -100},
	}}
	revision := s.transcript()
	s.Equal("hello again", revision.Text)
	s.False(revision.Final())
	s.Len(<-s.requests, 2*SampleRate*2, "include both the snapshot and audio received during inference")
}

func (s *AudioTurnSuite) TestCloseCancelsAnOutstandingRequest() {
	s.start()
	s.feed(1)
	select {
	case <-s.requests:
	case <-time.After(time.Second):
		s.FailNow("no audio request arrived")
	}
	done := make(chan error, 1)
	go func() { done <- s.provider.Close() }()
	select {
	case err := <-done:
		s.NoError(err)
	case <-time.After(time.Second):
		s.FailNow("Close did not cancel inference")
	}
}

func (s *AudioTurnSuite) TestAnUnsupportedDeploymentDoesNotPretendSilenceWasTranscribed() {
	s.replies <- transcriptionReply{unavailable: true}
	_, transcript, err := s.client.Transcribe(s.T().Context(), "unsupported", make([]byte, MinSamples*2))
	s.ErrorContains(err, "transcript unavailable")
	s.Nil(transcript)
}

func (s *AudioTurnSuite) TestUntranscribedAudioCannotBeSilentlyOverwritten() {
	s.start()
	err := s.provider.ProcessAudio(stt.PcmData{Samples: make([]int16, MaxSamples+1), SampleRate: SampleRate, Channels: 1}, stt.Participant{ID: "caller"})
	s.ErrorContains(err, "fell behind")
}

func (s *AudioTurnSuite) TestWordTimestampsCanRoundPastTheWindowEdges() {
	s.start()
	s.replies <- transcriptionReply{probability: 0.9, words: []Word{{Text: "gold.", StartMS: -1001, EndMS: 40, Confidence: 0.868}}}
	s.Require().NoError(s.provider.ProcessAudio(stt.PcmData{
		Samples: make([]int16, SampleRate+1), SampleRate: SampleRate, Channels: 1,
	}, stt.Participant{ID: "caller"}))
	partial := s.transcript()
	s.False(partial.Final())
	s.Equal("gold.", partial.Text)
	s.replies <- transcriptionReply{probability: 0.9, words: []Word{{Text: "gold.", StartMS: -600, EndMS: 40, Confidence: 0.868}}}
	s.feed(1)
	s.False(s.transcript().Final(), "unchanged words at the audio edge are still provisional")
}
