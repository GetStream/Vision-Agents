package streamedge

import (
	"math"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// BenchmarkSpeakerWrite is a second of a 24 kHz voice going through the speaker, which is
// what every reply costs the router per second it is spoken.
func BenchmarkSpeakerWrite(b *testing.B) {
	const rate = 24_000
	samples := make([]int16, rate/5)
	for i := range samples {
		samples[i] = int16(0.2 * math.MaxInt16 * math.Sin(2*math.Pi*220*float64(i)/rate))
	}
	chunk := audio.PcmData{Samples: samples, SampleRate: rate, Channels: 1}
	talker := newSpeaker(nil)
	b.ReportAllocs()
	for b.Loop() {
		for range 5 {
			if _, err := talker.queue(chunk, nil); err != nil {
				b.Fatal(err)
			}
		}
		talker.mu.Lock()
		talker.forgetFramesLocked()
		talker.mu.Unlock()
	}
}
