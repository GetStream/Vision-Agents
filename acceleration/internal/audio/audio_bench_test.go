package audio

import "testing"

// frame20ms is a frame of the size the live path moves: 20 ms of 16 kHz mono, which is
// what a transcriber is fed, and 20 ms of 24 kHz, which is what a voice emits.
func frame20ms(sampleRate int) PcmData {
	samples := make([]int16, sampleRate/50)
	for i := range samples {
		samples[i] = int16(i % 3000)
	}
	return PcmData{Samples: samples, SampleRate: sampleRate, Channels: 1}
}

func BenchmarkFromBytes(b *testing.B) {
	raw := frame20ms(16_000).Bytes()
	b.ReportAllocs()
	for b.Loop() {
		FromBytes(raw, 16_000, 1)
	}
}

func BenchmarkBytes(b *testing.B) {
	pcm := frame20ms(16_000)
	b.ReportAllocs()
	for b.Loop() {
		pcm.Bytes()
	}
}

func BenchmarkResampleDown(b *testing.B) {
	pcm := frame20ms(24_000)
	b.ReportAllocs()
	for b.Loop() {
		Resample(pcm, 16_000, 1)
	}
}

func BenchmarkLevel(b *testing.B) {
	pcm := frame20ms(24_000)
	leveller := NewLeveller()
	b.ReportAllocs()
	for b.Loop() {
		leveller.Level(pcm)
	}
}
