package openaicompat

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

type RealtimeAudioSuite struct{ suite.Suite }

func TestRealtimeAudioSuite(t *testing.T) { suite.Run(t, new(RealtimeAudioSuite)) }

func (s *RealtimeAudioSuite) TestResamplingIsIndependentOfOddChunkBoundaries() {
	input := make([]int16, 10_003)
	for i := range input {
		input[i] = int16(i*31%30_001 - 15_000)
	}

	var whole resampler
	want := whole.Flush(whole.Append(nil, input))
	var chunked resampler
	var got []byte
	for offset, size := 0, 1; offset < len(input); size = size%337 + 1 {
		end := min(offset+size, len(input))
		got = chunked.Append(got, input[offset:end])
		offset = end
	}
	got = chunked.Flush(got)
	s.Equal(want, got)
	s.Len(got, 2*((3*len(input)+1)/2))
	s.Equal(input[0], int16(binary.LittleEndian.Uint16(got[:2])))
}

func (s *RealtimeAudioSuite) TestSilenceDoesNotCreateATurn() {
	var endpoint endpointer
	var commits int
	endpoint.Push(make([]int16, stt.SampleRate*2), stt.Participant{ID: "quiet"}, func(frame endpointFrame) {
		if frame.commit {
			commits++
		}
	})
	endpoint.Close(func(frame endpointFrame) {
		if frame.commit {
			commits++
		}
	})
	s.Zero(commits)
}

func (s *RealtimeAudioSuite) TestSpeechLongerThanASixtySecondBackendLimitStaysOneTurn() {
	var endpoint endpointer
	var commits int
	speech := make([]int16, stt.SampleRate*61)
	for i := range speech {
		speech[i] = 2000
	}
	endpoint.Push(speech, stt.Participant{ID: "speaker"}, func(frame endpointFrame) {
		if frame.commit {
			commits++
		}
	})
	s.Zero(commits, "local endpointing must not split a long utterance at a provider limit")
	endpoint.Close(func(frame endpointFrame) {
		if frame.commit {
			commits++
		}
	})
	s.Equal(1, commits)
}

func (s *RealtimeAudioSuite) TestPreRollNeverCarriesAudioAcrossParticipants() {
	var endpoint endpointer
	var frames []endpointFrame
	emit := func(frame endpointFrame) { frames = append(frames, frame) }
	endpoint.Push(make([]int16, realtimeFrameSamples*realtimePreRollFrames), stt.Participant{ID: "old"}, emit)

	speech := make([]int16, realtimeFrameSamples*10)
	for i := range speech {
		speech[i] = 2000
	}
	participant := stt.Participant{ID: "new"}
	endpoint.Push(speech, participant, emit)
	endpoint.Close(emit)

	var samples int
	for _, frame := range frames {
		if frame.include {
			samples += len(frame.samples)
			s.Equal(participant, frame.participant)
		}
	}
	s.Equal(len(speech), samples, "the old speaker's preroll is discarded when ownership changes")
}

func BenchmarkRealtimeResampler(b *testing.B) {
	frame := make([]int16, realtimeFrameSamples)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		var converter resampler
		buffer := make([]byte, 0, realtimeAppendBytes*2)
		for range 50 {
			buffer = converter.Append(buffer, frame)
			for len(buffer) >= realtimeAppendBytes {
				_ = append([]byte(nil), buffer[:realtimeAppendBytes]...)
				copy(buffer, buffer[realtimeAppendBytes:])
				buffer = buffer[:len(buffer)-realtimeAppendBytes]
			}
		}
		buffer = converter.Flush(buffer)
		if len(buffer) > 0 {
			_ = append([]byte(nil), buffer...)
		}
	}
}
