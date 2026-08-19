package audio

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	// TelnyxRate is the PSTN sample rate used on the media stream.
	TelnyxRate = 8000
	// FrameSamples is 20 ms of 8 kHz mono audio.
	FrameSamples = 160
)

// PCM is mono signed 16-bit audio at a fixed sample rate.
type PCM struct {
	Rate    int
	Samples []int16
}

// WriteWAV writes a mono 16-bit PCM WAV file.
func WriteWAV(path string, pcm PCM) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return EncodeWAV(file, pcm)
}

// EncodeWAV writes a mono 16-bit PCM WAV stream.
func EncodeWAV(w io.Writer, pcm PCM) error {
	if pcm.Rate <= 0 {
		return fmt.Errorf("audio: sample rate must be positive")
	}
	dataBytes := len(pcm.Samples) * 2
	if err := writeWAVHeader(w, pcm.Rate, 1, dataBytes); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, pcm.Samples)
}

// WriteStereoWAV writes a two-channel 16-bit WAV (left = caller, right = agent).
func WriteStereoWAV(path string, rate int, left, right []int16) error {
	n := len(left)
	if len(right) > n {
		n = len(right)
	}
	interleaved := make([]int16, n*2)
	for i := 0; i < n; i++ {
		if i < len(left) {
			interleaved[i*2] = left[i]
		}
		if i < len(right) {
			interleaved[i*2+1] = right[i]
		}
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	dataBytes := len(interleaved) * 2
	if err := writeWAVHeader(file, rate, 2, dataBytes); err != nil {
		return err
	}
	return binary.Write(file, binary.LittleEndian, interleaved)
}

func writeWAVHeader(w io.Writer, rate, channels, dataBytes int) error {
	blockAlign := channels * 2
	var header [44]byte
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+dataBytes))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(rate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(rate*blockAlign))
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:36], 16)
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(dataBytes))
	_, err := w.Write(header[:])
	return err
}

// ReadWAV loads a mono 16-bit PCM WAV. Stereo files keep the left channel.
func ReadWAV(path string) (PCM, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return PCM{}, err
	}
	return DecodeWAV(data)
}

// DecodeWAV parses a 16-bit PCM WAV from memory.
func DecodeWAV(data []byte) (PCM, error) {
	if len(data) < 44 {
		return PCM{}, fmt.Errorf("audio: wav too short")
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return PCM{}, fmt.Errorf("audio: not a WAVE file")
	}
	offset := 12
	var rate, channels, bits int
	var pcm []int16
	for offset+8 <= len(data) {
		chunk := string(data[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(data[offset+4 : offset+8]))
		body := offset + 8
		if body+size > len(data) {
			return PCM{}, fmt.Errorf("audio: truncated %s chunk", chunk)
		}
		switch chunk {
		case "fmt ":
			if size < 16 {
				return PCM{}, fmt.Errorf("audio: fmt chunk too small")
			}
			if binary.LittleEndian.Uint16(data[body:body+2]) != 1 {
				return PCM{}, fmt.Errorf("audio: only PCM wav is supported")
			}
			channels = int(binary.LittleEndian.Uint16(data[body+2 : body+4]))
			rate = int(binary.LittleEndian.Uint32(data[body+4 : body+8]))
			bits = int(binary.LittleEndian.Uint16(data[body+14 : body+16]))
			if bits != 16 {
				return PCM{}, fmt.Errorf("audio: only 16-bit wav is supported")
			}
		case "data":
			samples := size / 2
			raw := make([]int16, samples)
			for i := 0; i < samples; i++ {
				raw[i] = int16(binary.LittleEndian.Uint16(data[body+i*2 : body+i*2+2]))
			}
			if channels == 2 {
				mono := make([]int16, samples/2)
				for i := range mono {
					mono[i] = raw[i*2]
				}
				pcm = mono
			} else {
				pcm = raw
			}
		}
		offset = body + size
		if size%2 == 1 {
			offset++
		}
	}
	if rate == 0 || pcm == nil {
		return PCM{}, fmt.Errorf("audio: missing fmt or data chunk")
	}
	return PCM{Rate: rate, Samples: pcm}, nil
}

// FromPCM16LE reads raw little-endian 16-bit samples.
func FromPCM16LE(raw []byte, rate int) PCM {
	n := len(raw) / 2
	samples := make([]int16, n)
	for i := 0; i < n; i++ {
		samples[i] = int16(uint16(raw[i*2]) | uint16(raw[i*2+1])<<8)
	}
	return PCM{Rate: rate, Samples: samples}
}

// Resample linearly converts PCM to a new sample rate.
func Resample(pcm PCM, rate int) PCM {
	if pcm.Rate == rate || len(pcm.Samples) == 0 {
		return PCM{Rate: rate, Samples: append([]int16(nil), pcm.Samples...)}
	}
	outLen := int(float64(len(pcm.Samples)) * float64(rate) / float64(pcm.Rate))
	if outLen < 1 {
		outLen = 1
	}
	out := make([]int16, outLen)
	ratio := float64(pcm.Rate) / float64(rate)
	last := len(pcm.Samples) - 1
	for i := range out {
		src := float64(i) * ratio
		idx := int(src)
		frac := src - float64(idx)
		if idx >= last {
			out[i] = pcm.Samples[last]
			continue
		}
		a := float64(pcm.Samples[idx])
		b := float64(pcm.Samples[idx+1])
		out[i] = int16(a + (b-a)*frac)
	}
	return PCM{Rate: rate, Samples: out}
}

// Silence returns n samples of zeros.
func Silence(n int) []int16 {
	if n < 0 {
		n = 0
	}
	return make([]int16, n)
}

// Concat joins sample buffers.
func Concat(parts ...[]int16) []int16 {
	n := 0
	for _, p := range parts {
		n += len(p)
	}
	out := make([]int16, 0, n)
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// PadRight extends samples to n with zeros.
func PadRight(samples []int16, n int) []int16 {
	if len(samples) >= n {
		return samples
	}
	out := make([]int16, n)
	copy(out, samples)
	return out
}
