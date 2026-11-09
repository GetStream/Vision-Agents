package session

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Every case router_silent_sessions_total documents. Written out rather than read from
// agent.Silences, so a case added, dropped or renamed there fails here.
var documentedSilences = []string{"no_inbound_audio", "silent_inbound_audio", "no_words", "no_audible_outbound"}

// silences is every case and kind of router_silent_sessions_total, by "case/kind".
func (s *SessionSuite) silences() map[string]float64 {
	counts := map[string]float64{}
	for _, silence := range documentedSilences {
		for _, kind := range []string{"phone", "web"} {
			value, ok := s.series("router_silent_sessions_total", map[string]string{"case": silence, "kind": kind})
			s.Require().True(ok, "no series for %s/%s", silence, kind)
			counts[silence+"/"+kind] = value
		}
	}
	return counts
}

// silentOnce checks that, since before, the one series counted one more and no other moved.
func (s *SessionSuite) silentOnce(before map[string]float64, series string) {
	after := s.silences()
	want := map[string]float64{}
	for key, count := range before {
		want[key] = count
	}
	want[series]++
	s.Equal(want, after)
}

// sendsTo pushes one frame at level into the session's call, and waits for its agent to have
// counted it.
func (s *SessionSuite) sendsTo(created *Session, from stt.Participant, level int16) {
	frames := created.current().CallAudio().Frames
	samples := make([]int16, 320)
	for i := range samples {
		samples[i] = level
	}
	s.call().inbound <- agent.InboundAudio{Participant: from, Audio: audio.PcmData{Samples: samples, SampleRate: 16_000, Channels: 1}}
	s.eventually(func() bool { return created.current().CallAudio().Frames == frames+1 }, "the agent never counted the frame")
}

// closes closes a session the way its caller does, twice, so a second close counting again
// is seen.
func (s *SessionSuite) closes(created *Session) {
	closed, err := s.manager.Close(created.ID(), backend("acme"))
	s.Require().NoError(err)
	s.Require().True(closed)
	s.Require().NoError(created.Close())
}

func (s *SessionSuite) TestAWebSessionACallerJoinedWithoutAudioIsCountedOnceAsNoInboundAudio() {
	// Catches the count missing, taken as a phone session, or counted again by a second close.
	s.present = []stt.Participant{visitor}
	s.manages()
	before := s.silences()
	created := s.joins(Spec{})
	s.settles(created)

	s.closes(created)

	s.silentOnce(before, "no_inbound_audio/web")
}

func (s *SessionSuite) TestAPhoneSessionWhoseCallerSentOnlySilenceIsCountedAsSilentInboundAudio() {
	// Catches a phone session counted as a web one.
	s.manages()
	before := s.silences()
	created := s.rings(inbound(), caller)
	s.settles(created)
	s.sendsTo(created, caller, 0)

	s.closes(created)

	s.silentOnce(before, "silent_inbound_audio/phone")
}

func (s *SessionSuite) TestAWebSessionWithTheCallerHeardIsNotCountedSilent() {
	// Catches a normal call counted as silent.
	s.present = []stt.Participant{visitor}
	s.manages()
	before := s.silences()
	created := s.joins(Spec{})
	s.settles(created)
	s.sendsTo(created, visitor, audio.AudibleSample)
	s.ears.emitter.Send(stt.Transcript{Participant: visitor, Mode: stt.ModeFinal, Text: "hello", Language: "en", Confidence: 1})
	s.eventually(func() bool { return created.current().CallAudio().Words == 1 }, "the words were never heard")

	s.closes(created)

	s.Equal(before, s.silences())
}

func (s *SessionSuite) TestEveryVoiceRunOfASessionIsJudgedTogether() {
	// Catches a session judged on its last run alone: voice stopped and started again after
	// the caller was heard.
	s.present = []stt.Participant{visitor}
	s.manages()
	before := s.silences()
	created := s.writes(Spec{})
	s.Require().NoError(created.StartVoice(s.ctx, Call{}))
	s.settles(created)
	s.sendsTo(created, visitor, audio.AudibleSample)
	s.ears.emitter.Send(stt.Transcript{Participant: visitor, Mode: stt.ModeFinal, Text: "hello", Language: "en", Confidence: 1})
	s.eventually(func() bool { return created.current().CallAudio().Words == 1 }, "the words were never heard")
	s.Require().NoError(created.StopVoice(s.ctx))
	s.Require().NoError(created.StartVoice(s.ctx, Call{}))
	s.settles(created)

	s.closes(created)

	s.Equal(before, s.silences())
}
