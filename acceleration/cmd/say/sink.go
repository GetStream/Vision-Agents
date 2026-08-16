package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
)

// sink consumes synthesised audio. Both implementations learn the sample rate from the
// first chunk rather than being told it, since that is the only place it is authoritative:
// the router may have failed over to a provider that generates at a different rate.
type sink interface {
	Write(pcm audio.PcmData) error
	Close() error
}

// newSink plays the audio, or writes it to a WAV file when a path is given.
func newSink(path string) (sink, error) {
	if path == "" {
		return &player{}, nil
	}
	return newWavWriter(path)
}

// player streams audio into ffplay, which starts playing the moment the first chunk
// arrives.
//
// The stream is a WAV rather than raw PCM even though the audio is already PCM: a header
// carries the sample rate and channel layout, so no command-line flag has to describe
// them. That matters because the flag for channel count has changed across ffmpeg
// releases, and this way any version can play what it is given.
type player struct {
	command *exec.Cmd
	stdin   io.WriteCloser
}

func (p *player) Write(pcm audio.PcmData) error {
	if p.command == nil {
		if err := p.start(pcm.SampleRate, pcm.Channels); err != nil {
			return err
		}
	}
	_, err := p.stdin.Write(pcm.Bytes())
	return err
}

func (p *player) start(sampleRate, channels int) error {
	path, err := exec.LookPath("ffplay")
	if err != nil {
		return errors.New("ffplay is needed to play audio (brew install ffmpeg), or use -out to write a file")
	}

	// -nodisp because there is nothing to look at, -autoexit so it stops when stdin ends.
	p.command = exec.Command(path, "-f", "wav", "-nodisp", "-autoexit", "-loglevel", "error", "-i", "-")
	p.command.Stderr = os.Stderr

	stdin, err := p.command.StdinPipe()
	if err != nil {
		return err
	}
	p.stdin = stdin

	if err := p.command.Start(); err != nil {
		return fmt.Errorf("start ffplay: %w", err)
	}

	// The length is not known yet, so the header claims the maximum. This is how a WAV is
	// streamed, and ffplay stops at the end of the audio it actually receives.
	_, err = p.stdin.Write(wavHeader(sampleRate, channels, math.MaxUint32))
	return err
}

// Close lets ffplay finish the audio it has been given rather than cutting it off.
func (p *player) Close() error {
	if p.command == nil {
		return nil
	}
	if err := p.stdin.Close(); err != nil {
		return err
	}
	return p.command.Wait()
}

// wavHeaderBytes is the size of the canonical 44-byte PCM header.
const wavHeaderBytes = 44

// wavWriter writes a PCM16 WAV file. The size in the header is only known once the last
// chunk has arrived, so a placeholder is written first and patched on close.
type wavWriter struct {
	file       *os.File
	sampleRate int
	channels   int
	dataBytes  uint32
}

func newWavWriter(path string) (*wavWriter, error) {
	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	if _, err := file.Write(make([]byte, wavHeaderBytes)); err != nil {
		file.Close()
		return nil, err
	}
	return &wavWriter{file: file}, nil
}

func (w *wavWriter) Write(pcm audio.PcmData) error {
	if w.sampleRate == 0 {
		w.sampleRate, w.channels = pcm.SampleRate, pcm.Channels
	}
	if pcm.SampleRate != w.sampleRate {
		return fmt.Errorf("a wav file holds one sample rate, got %d after %d", pcm.SampleRate, w.sampleRate)
	}

	raw := pcm.Bytes()
	if _, err := w.file.Write(raw); err != nil {
		return err
	}
	w.dataBytes += uint32(len(raw))
	return nil
}

func (w *wavWriter) Close() error {
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		w.file.Close()
		return err
	}
	if _, err := w.file.Write(wavHeader(w.sampleRate, w.channels, w.dataBytes)); err != nil {
		w.file.Close()
		return err
	}
	return w.file.Close()
}

// wavHeader builds the canonical 44-byte PCM16 header.
func wavHeader(sampleRate, channels int, dataBytes uint32) []byte {
	const bitsPerSample = 16
	channels = max(channels, 1)
	byteRate := uint32(sampleRate * channels * bitsPerSample / 8)
	blockAlign := uint16(channels * bitsPerSample / 8)

	// A streaming header claims the maximum length, and adding to it would wrap.
	riffSize := dataBytes
	if riffSize < math.MaxUint32-36 {
		riffSize += 36
	}

	header := make([]byte, 0, wavHeaderBytes)
	header = append(header, "RIFF"...)
	header = binary.LittleEndian.AppendUint32(header, riffSize)
	header = append(header, "WAVE"...)
	header = append(header, "fmt "...)
	header = binary.LittleEndian.AppendUint32(header, 16)
	header = binary.LittleEndian.AppendUint16(header, 1)
	header = binary.LittleEndian.AppendUint16(header, uint16(channels))
	header = binary.LittleEndian.AppendUint32(header, uint32(sampleRate))
	header = binary.LittleEndian.AppendUint32(header, byteRate)
	header = binary.LittleEndian.AppendUint16(header, blockAlign)
	header = binary.LittleEndian.AppendUint16(header, bitsPerSample)
	header = append(header, "data"...)
	return binary.LittleEndian.AppendUint32(header, dataBytes)
}
