package stream

import (
	"encoding/binary"
	"fmt"
	"iter"
	"os"
	"time"
)

// transcriptionRate is the rate Transcriber.Send takes audio at.
const transcriptionRate = 16_000

// The pace a call arrives at, and the quiet after it.
const (
	recordedChunk = 100 * time.Millisecond
	recordedQuiet = 2 * time.Second
)

// RecordedCall is a WAV file delivered the way a call delivers audio, for a program with
// nobody at a microphone.
//
// Sending a whole clip at once has the model see all of it before anybody has finished
// talking, which is not the problem a streaming model solves. The silence after the clip is
// not padding either: a streaming model decides a turn is over by hearing the caller stop,
// and a clip that ends the instant the speech does never gives it that.
//
// The file has to be 16 kHz mono PCM16 already, which is what Transcriber.Send takes.
//
//	call, err := stream.RecordedCall("saturday_seven_thirty.wav")
//	for chunk := range call {
//		err := transcriber.Send(chunk)
//	}
func RecordedCall(path string) (iter.Seq[[]byte], error) {
	samples, err := readWAV(path)
	if err != nil {
		return nil, err
	}

	// Two bytes to a sample, so a chunk is that many samples of that many bytes.
	size := 2 * transcriptionRate * int(recordedChunk.Milliseconds()) / 1000
	quiet := make([]byte, 2*transcriptionRate*int(recordedQuiet.Milliseconds())/1000)

	return func(yield func([]byte) bool) {
		for _, audio := range [][]byte{samples, quiet} {
			for start := 0; start < len(audio); start += size {
				if !yield(audio[start:min(start+size, len(audio))]) {
					return
				}
				time.Sleep(recordedChunk)
			}
		}
	}, nil
}

// readWAV is the samples of a 16 kHz mono PCM16 WAV.
//
// The chunks before the audio are walked rather than skipped by a fixed offset: a file
// written by ffmpeg carries a LIST chunk of its own metadata, so the samples do not start
// where a bare 44-byte header would put them.
func readWAV(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 12 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, fmt.Errorf("stream: %s is not a WAV file", path)
	}

	for at := 12; at+8 <= len(raw); {
		name := string(raw[at : at+4])
		size := int(binary.LittleEndian.Uint32(raw[at+4 : at+8]))
		body := at + 8
		size = min(size, len(raw)-body)

		switch name {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("stream: %s has a format chunk too short to read", path)
			}
			format := binary.LittleEndian.Uint16(raw[body : body+2])
			channels := binary.LittleEndian.Uint16(raw[body+2 : body+4])
			rate := binary.LittleEndian.Uint32(raw[body+4 : body+8])
			bits := binary.LittleEndian.Uint16(raw[body+14 : body+16])
			if format != 1 || channels != 1 || rate != transcriptionRate || bits != 16 {
				return nil, fmt.Errorf(
					"stream: %s is not 16 kHz mono PCM16: format %d, %d channels, %d Hz, %d bits",
					path, format, channels, rate, bits)
			}
		case "data":
			return raw[body : body+size], nil
		}

		// Chunks are padded to an even length, and the pad byte is not counted in the
		// size, so a chunk of odd length is followed by one byte that belongs to nobody.
		at = body + size + size%2
	}
	return nil, fmt.Errorf("stream: %s holds no audio", path)
}
