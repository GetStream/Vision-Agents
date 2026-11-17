// Package testaudio loads the audio fixtures shared with the Python test suite so the
// Go providers can be exercised against real speech.
package testaudio

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// assetDir is the fixture directory, relative to the repository root.
const assetDir = "tests/test_assets"

// peakAmplitude is full scale for PCM16.
const peakAmplitude = 32767

// noiseSeed keeps generated noise the same from run to run.
const noiseSeed = 20260817

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

// Reference returns what is actually said in a fixture, read from the sidecar JSON the
// Python suite already keeps beside it. It is the yardstick a transcript is scored
// against, so a provider that drops half a sentence is caught rather than credited for
// the words it did get.
func Reference(name string) (string, error) {
	path, err := Asset(strings.TrimSuffix(name, filepath.Ext(name)) + ".json")
	if err != nil {
		return "", err
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("testaudio: reading %s: %w", path, err)
	}

	var metadata struct {
		Transcript string `json:"transcript"`
	}
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return "", fmt.Errorf("testaudio: parsing %s: %w", path, err)
	}
	if metadata.Transcript == "" {
		return "", fmt.Errorf("testaudio: %s has no transcript", path)
	}
	return metadata.Transcript, nil
}

// Load16kMono decodes an audio file to the 16 kHz mono PCM16 the providers expect.
// It shells out to ffmpeg, so callers should skip when ffmpeg is unavailable.
func Load16kMono(name string) (stt.PcmData, error) {
	path, err := Asset(name)
	if err != nil {
		return stt.PcmData{}, err
	}

	return convert(exec.Command(
		"ffmpeg", "-loglevel", "error", "-i", path,
		"-f", "s16le", "-acodec", "pcm_s16le",
		"-ar", fmt.Sprint(stt.SampleRate), "-ac", "1", "-",
	))
}

// Resample16kMono converts audio to the 16 kHz mono PCM16 a call delivers, whatever rate
// the voice that produced it used. It shells out to ffmpeg, like Load16kMono.
func Resample16kMono(pcm stt.PcmData) (stt.PcmData, error) {
	if pcm.SampleRate == stt.SampleRate && pcm.Channels == 1 {
		return pcm, nil
	}
	if pcm.SampleRate <= 0 || pcm.Channels <= 0 {
		return stt.PcmData{}, fmt.Errorf("testaudio: cannot resample %d Hz in %d channels",
			pcm.SampleRate, pcm.Channels)
	}

	cmd := exec.Command(
		"ffmpeg", "-loglevel", "error",
		"-f", "s16le", "-ar", fmt.Sprint(pcm.SampleRate), "-ac", fmt.Sprint(pcm.Channels), "-i", "-",
		"-f", "s16le", "-acodec", "pcm_s16le",
		"-ar", fmt.Sprint(stt.SampleRate), "-ac", "1", "-",
	)
	cmd.Stdin = bytes.NewReader(pcm.Bytes())
	return convert(cmd)
}

// Noise returns the requested duration of broadband room noise at the given amplitude,
// 0 to 1. The seed is fixed, so a conversation that fails in noise fails in the same
// noise next time.
func Noise(durationMs int, amplitude float64) stt.PcmData {
	random := rand.New(rand.NewPCG(noiseSeed, noiseSeed))
	samples := make([]int16, stt.SampleRate*durationMs/1000)
	// Room noise sits low in the spectrum, so the white source is smoothed rather than
	// used raw: broadband hiss is both easier to transcribe through and unlike a room.
	var smoothed float64
	for i := range samples {
		smoothed += 0.2 * ((random.Float64()*2 - 1) - smoothed)
		samples[i] = clamp(smoothed * amplitude * float64(peakAmplitude) / 0.2)
	}
	return stt.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
}

// Mix lays one chunk over another at the given gain, the way a room's noise sits under a
// voice. The result is as long as the base, and loud sums clip rather than wrap.
func Mix(base, overlay stt.PcmData, gain float64) stt.PcmData {
	mixed := stt.PcmData{
		Samples:    make([]int16, len(base.Samples)),
		SampleRate: base.SampleRate,
		Channels:   base.Channels,
	}
	for i, sample := range base.Samples {
		sum := float64(sample)
		if len(overlay.Samples) > 0 {
			sum += float64(overlay.Samples[i%len(overlay.Samples)]) * gain
		}
		mixed.Samples[i] = clamp(sum)
	}
	return mixed
}

func clamp(sample float64) int16 {
	if sample > peakAmplitude {
		return peakAmplitude
	}
	if sample < -peakAmplitude {
		return -peakAmplitude
	}
	return int16(sample)
}

// convert runs an ffmpeg command that writes 16 kHz mono PCM16 to stdout.
func convert(cmd *exec.Cmd) (stt.PcmData, error) {
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
