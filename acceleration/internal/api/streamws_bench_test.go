package api

import (
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// BenchmarkAudioMessage measures the socket's per-chunk framing, which every chunk of
// every reply on a session socket passes through.
func BenchmarkAudioMessage(b *testing.B) {
	samples := make([]int16, 24_000/5)
	for i := range samples {
		samples[i] = int16(i % 3000)
	}
	pcm := audio.PcmData{Samples: samples, SampleRate: 24_000, Channels: 1}
	b.ReportAllocs()
	for b.Loop() {
		audioMessage(pcm)
	}
}
