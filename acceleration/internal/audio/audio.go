// Package audio carries the PCM type every provider exchanges.
//
// The sample rate is deliberately not fixed here. Speech-to-text wants 16 kHz mono
// because that is what the providers accept, while text-to-speech emits whatever its
// model produces, so each provider validates the rate it needs at its own boundary.
package audio

import (
	"encoding/binary"
	"fmt"
)

// PcmData is a chunk of signed 16-bit PCM audio.
type PcmData struct {
	Samples []int16
	// SampleRate is in Hz.
	SampleRate int
	// Channels is 1 for mono. Multi-channel samples are interleaved.
	Channels int
}

// FromBytes reads little-endian PCM16, the wire format every provider uses. A trailing
// odd byte cannot form a sample and is dropped.
func FromBytes(raw []byte, sampleRate, channels int) PcmData {
	samples := make([]int16, len(raw)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return PcmData{Samples: samples, SampleRate: sampleRate, Channels: channels}
}

// DurationMs returns the wall-clock length of the chunk.
func (p PcmData) DurationMs() float64 {
	if p.SampleRate <= 0 || p.Channels <= 0 {
		return 0
	}
	frames := len(p.Samples) / p.Channels
	return float64(frames) / float64(p.SampleRate) * 1000
}

// Bytes returns the samples as little-endian PCM16.
func (p PcmData) Bytes() []byte {
	out := make([]byte, len(p.Samples)*2)
	for i, sample := range p.Samples {
		binary.LittleEndian.PutUint16(out[i*2:], uint16(sample))
	}
	return out
}

// Validate reports whether the chunk is single-channel PCM at the wanted sample rate.
func (p PcmData) Validate(wantSampleRate int) error {
	if p.SampleRate != wantSampleRate {
		return fmt.Errorf("sample rate must be %d, got %d", wantSampleRate, p.SampleRate)
	}
	if p.Channels != 1 {
		return fmt.Errorf("channels must be 1, got %d", p.Channels)
	}
	return nil
}
