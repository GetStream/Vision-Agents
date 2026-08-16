// Package testaudio loads the audio fixtures shared with the Python test suite so the
// Go providers can be exercised against real speech.
package testaudio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// assetDir is the fixture directory, relative to the repository root.
const assetDir = "tests/test_assets"

// Asset resolves a fixture by name by walking up from the working directory until the
// shared asset directory appears.
func Asset(name string) (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		candidate := filepath.Join(dir, assetDir, name)
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("testaudio: %s not found above %s", filepath.Join(assetDir, name), dir)
		}
		dir = parent
	}
}

// Load16kMono decodes an audio file to the 16 kHz mono PCM16 the providers expect.
// It shells out to ffmpeg, so callers should skip when ffmpeg is unavailable.
func Load16kMono(name string) (stt.PcmData, error) {
	path, err := Asset(name)
	if err != nil {
		return stt.PcmData{}, err
	}

	cmd := exec.Command(
		"ffmpeg", "-loglevel", "error", "-i", path,
		"-f", "s16le", "-acodec", "pcm_s16le",
		"-ar", fmt.Sprint(stt.SampleRate), "-ac", "1", "-",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stt.PcmData{}, fmt.Errorf("testaudio: ffmpeg failed: %w: %s", err, stderr.String())
	}

	raw := stdout.Bytes()
	samples := make([]int16, len(raw)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}

	return stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}, nil
}

// HasFFmpeg reports whether ffmpeg is on PATH.
func HasFFmpeg() bool {
	_, err := exec.LookPath("ffmpeg")
	return err == nil
}

// Silence returns the requested duration of 16 kHz mono silence, which providers need
// in order to detect the end of a turn.
func Silence(durationMs int) stt.PcmData {
	samples := stt.SampleRate * durationMs / 1000
	return stt.PcmData{Samples: make([]int16, samples), SampleRate: stt.SampleRate, Channels: 1}
}

// Chunks splits audio into fixed-duration pieces, mirroring how a live call arrives.
func Chunks(pcm stt.PcmData, durationMs int) []stt.PcmData {
	size := pcm.SampleRate * durationMs / 1000
	if size <= 0 {
		return nil
	}

	var chunks []stt.PcmData
	for start := 0; start < len(pcm.Samples); start += size {
		end := min(start+size, len(pcm.Samples))
		chunks = append(chunks, stt.PcmData{
			Samples:    pcm.Samples[start:end],
			SampleRate: pcm.SampleRate,
			Channels:   pcm.Channels,
		})
	}
	return chunks
}
