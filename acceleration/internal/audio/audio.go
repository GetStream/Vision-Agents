// Package audio carries the PCM type every provider exchanges.
//
// The sample rate is deliberately not fixed here. Speech-to-text wants 16 kHz mono
// because that is what the providers accept, while text-to-speech emits whatever its
// model produces, so each provider validates the rate it needs at its own boundary.
package audio

import (
	"encoding/binary"
	"fmt"
	"math"
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

// Resample converts a chunk to another sample rate and to mono.
//
// It exists because the two ends of an in-process call disagree: a voice provider emits
// whatever its model produces, usually 24 kHz, and every transcriber wants 16 kHz mono.
// Nothing else in this process could do it without shelling out to ffmpeg or crossing into
// cgo, and neither belongs on the path of a conversation.
//
// The interpolation is linear, under a box filter when the rate is coming down. That is not
// a resampler anybody would ship in a codec, and it is comfortably good enough for speech
// on its way into a transcriber, which is the only thing this is for.
func Resample(pcm PcmData, sampleRate, channels int) PcmData {
	if channels != 1 {
		return pcm
	}
	mono := downmix(pcm)
	if mono.SampleRate == sampleRate || len(mono.Samples) == 0 {
		mono.SampleRate = sampleRate
		return mono
	}

	ratio := float64(mono.SampleRate) / float64(sampleRate)
	// Coming down, each output sample is the average of the inputs it stands for, which is
	// what keeps the frequencies above the new limit from folding back as a whistle.
	width := 1
	if ratio > 1 {
		width = int(ratio)
	}

	length := int(float64(len(mono.Samples))/ratio + 0.5)
	resampled := make([]int16, length)
	for i := range resampled {
		at := float64(i) * ratio
		resampled[i] = clamp(average(mono.Samples, at, width))
	}
	return PcmData{Samples: resampled, SampleRate: sampleRate, Channels: 1}
}

// downmix folds every channel into one by averaging them.
func downmix(pcm PcmData) PcmData {
	if pcm.Channels <= 1 {
		return PcmData{Samples: pcm.Samples, SampleRate: pcm.SampleRate, Channels: 1}
	}

	folded := make([]int16, len(pcm.Samples)/pcm.Channels)
	for i := range folded {
		var sum int
		for channel := 0; channel < pcm.Channels; channel++ {
			sum += int(pcm.Samples[i*pcm.Channels+channel])
		}
		folded[i] = int16(sum / pcm.Channels)
	}
	return PcmData{Samples: folded, SampleRate: pcm.SampleRate, Channels: 1}
}

// average is the interpolated sample at a fractional position, over the given width.
func average(samples []int16, at float64, width int) float64 {
	var sum float64
	for offset := 0; offset < width; offset++ {
		sum += interpolate(samples, at+float64(offset))
	}
	return sum / float64(width)
}

func interpolate(samples []int16, at float64) float64 {
	index := int(at)
	if index >= len(samples)-1 {
		return float64(samples[len(samples)-1])
	}
	fraction := at - float64(index)
	return float64(samples[index])*(1-fraction) + float64(samples[index+1])*fraction
}

func clamp(value float64) int16 {
	switch {
	case value > math.MaxInt16:
		return math.MaxInt16
	case value < math.MinInt16:
		return math.MinInt16
	default:
		return int16(value)
	}
}
