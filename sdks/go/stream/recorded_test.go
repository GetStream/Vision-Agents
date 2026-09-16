package stream

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// wav writes a WAV of the given samples, with a LIST chunk before the audio the way ffmpeg
// leaves one, so the reader is held to walking the chunks rather than skipping 44 bytes.
func wav(t *testing.T, rate, channels, samples int) string {
	t.Helper()

	list := []byte("LIST\x04\x00\x00\x00INFO")
	audio := make([]byte, samples*2)
	for i := range samples {
		binary.LittleEndian.PutUint16(audio[i*2:], uint16(i))
	}

	file := make([]byte, 0, 12+24+len(list)+8+len(audio))
	file = append(file, "RIFF\x00\x00\x00\x00WAVE"...)
	format := make([]byte, 24)
	copy(format, "fmt ")
	binary.LittleEndian.PutUint32(format[4:], 16)
	binary.LittleEndian.PutUint16(format[8:], 1)
	binary.LittleEndian.PutUint16(format[10:], uint16(channels))
	binary.LittleEndian.PutUint32(format[12:], uint32(rate))
	binary.LittleEndian.PutUint32(format[16:], uint32(rate*channels*2))
	binary.LittleEndian.PutUint16(format[20:], uint16(channels*2))
	binary.LittleEndian.PutUint16(format[22:], 16)
	file = append(file, format...)
	file = append(file, list...)
	header := make([]byte, 8)
	copy(header, "data")
	binary.LittleEndian.PutUint32(header[4:], uint32(len(audio)))
	file = append(file, header...)
	file = append(file, audio...)
	binary.LittleEndian.PutUint32(file[4:8], uint32(len(file)-8))

	path := filepath.Join(t.TempDir(), "clip.wav")
	if err := os.WriteFile(path, file, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestARecordedCallArrivesAChunkAtATimeAndThenGoesQuiet(t *testing.T) {
	// Two and a half chunks of speech, so the clip does not divide evenly into them.
	spoken := transcriptionRate / 4
	call, err := RecordedCall(wav(t, transcriptionRate, 1, spoken))
	if err != nil {
		t.Fatal(err)
	}

	started := time.Now()
	var sizes []int
	heard := 0
	for chunk := range call {
		sizes = append(sizes, len(chunk))
		heard += len(chunk)
	}
	elapsed := time.Since(started)

	// The clip and then the two seconds of quiet, in 100ms chunks, with the odd half
	// chunk of speech kept rather than rounded away.
	if want := 2 * (spoken + transcriptionRate*2); heard != want {
		t.Errorf("streamed %d bytes, want %d", heard, want)
	}
	if want := 3 + 20; len(sizes) != want {
		t.Errorf("streamed %d chunks, want %d", len(sizes), want)
	}
	if full := 2 * transcriptionRate / 10; sizes[0] != full || sizes[2] != full/2 {
		t.Errorf("chunks are %v bytes, want 100ms each and the remainder last", sizes[:3])
	}
	if elapsed < time.Second {
		t.Errorf("streamed %v of audio in %v, which is not the pace a call arrives at",
			recordedQuiet, elapsed)
	}
}

func TestBreakingOutOfARecordedCallStopsIt(t *testing.T) {
	call, err := RecordedCall(wav(t, transcriptionRate, 1, transcriptionRate))
	if err != nil {
		t.Fatal(err)
	}

	chunks := 0
	for range call {
		chunks++
		break
	}

	if chunks != 1 {
		t.Errorf("took %d chunks after breaking out, want 1", chunks)
	}
}

func TestACallTheTranscriberCouldNotTakeIsRefusedRatherThanSent(t *testing.T) {
	for _, refused := range []struct {
		why      string
		rate     int
		channels int
	}{
		{why: "the wrong sample rate", rate: 8_000, channels: 1},
		{why: "stereo", rate: transcriptionRate, channels: 2},
	} {
		if _, err := RecordedCall(wav(t, refused.rate, refused.channels, 100)); err == nil {
			t.Errorf("%s was accepted, and would have been transcribed as gibberish", refused.why)
		}
	}

	if _, err := RecordedCall(filepath.Join(t.TempDir(), "absent.wav")); err == nil {
		t.Error("a file that is not there was accepted")
	}
}
