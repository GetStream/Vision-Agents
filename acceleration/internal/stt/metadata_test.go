package stt

import (
	"context"
	"encoding/binary"
	"os"
	"os/exec"
	"testing"
)

func TestProbeDurationUsesOriginalMediaAndCleansTemporaryFile(t *testing.T) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe is unavailable")
	}
	tempDir := t.TempDir()
	t.Setenv("TMPDIR", tempDir)

	audio := pcmWAV(1, 16000)
	recording := Recording{Audio: append([]byte(nil), audio...), Filename: "call.wav"}
	duration, ok := ProbeDuration(context.Background(), recording)
	if !ok {
		t.Fatal("ProbeDuration did not recognize the valid WAV fixture")
	}
	if duration != 1000 {
		t.Fatalf("ProbeDuration() = %d ms, want 1000 ms", duration)
	}
	if string(recording.Audio) != string(audio) {
		t.Fatal("ProbeDuration changed the original audio bytes")
	}
	entries, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("ProbeDuration left %d temporary files behind", len(entries))
	}
}

func TestProbeDurationReturnsUnknownForCanceledContext(t *testing.T) {
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe is unavailable")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if duration, ok := ProbeDuration(ctx, Recording{Audio: pcmWAV(1, 16000)}); ok || duration != 0 {
		t.Fatalf("ProbeDuration(canceled context) = (%d, %v), want (0, false)", duration, ok)
	}
}

func pcmWAV(seconds, sampleRate int) []byte {
	const headerBytes = 44
	dataBytes := seconds * sampleRate * 2
	wav := make([]byte, headerBytes+dataBytes)
	copy(wav[0:4], "RIFF")
	binary.LittleEndian.PutUint32(wav[4:8], uint32(len(wav)-8))
	copy(wav[8:12], "WAVE")
	copy(wav[12:16], "fmt ")
	binary.LittleEndian.PutUint32(wav[16:20], 16)
	binary.LittleEndian.PutUint16(wav[20:22], 1)
	binary.LittleEndian.PutUint16(wav[22:24], 1)
	binary.LittleEndian.PutUint32(wav[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(wav[28:32], uint32(sampleRate*2))
	binary.LittleEndian.PutUint16(wav[32:34], 2)
	binary.LittleEndian.PutUint16(wav[34:36], 16)
	copy(wav[36:40], "data")
	binary.LittleEndian.PutUint32(wav[40:44], uint32(dataBytes))
	return wav
}
