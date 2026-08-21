package transport

import (
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

func TestChunk(t *testing.T) {
	t.Run("ragged writes rebuild the stream in order", func(t *testing.T) {
		const total = audio.FrameSamples * 50
		src := make([]int16, total)
		for i := range src {
			src[i] = int16(i % 1000)
		}

		out := make(chan Frame, 1024)
		var pending []int16
		for i := 0; i < total; {
			n := min(i%173+1, total-i)
			pending, _ = Chunk(pending, src[i:i+n], out)
			i += n
		}
		close(out)

		var got []int16
		for frame := range out {
			if len(frame.PCM) != audio.FrameSamples {
				t.Fatalf("frame carried %d samples, want %d", len(frame.PCM), audio.FrameSamples)
			}
			got = append(got, frame.PCM...)
		}

		whole := total / audio.FrameSamples * audio.FrameSamples
		if len(got) != whole {
			t.Fatalf("emitted %d samples, want %d", len(got), whole)
		}
		for i := range got {
			if got[i] != src[i] {
				t.Fatalf("sample %d is %d, want %d", i, got[i], src[i])
			}
		}
		if len(pending) != total-whole {
			t.Fatalf("kept %d leftover samples, want %d", len(pending), total-whole)
		}
	})

	t.Run("drops frames when the consumer is full and says how many", func(t *testing.T) {
		out := make(chan Frame, 1)
		pending, dropped := Chunk(nil, make([]int16, audio.FrameSamples*4), out)
		if len(pending) != 0 {
			t.Fatalf("kept %d leftover samples, want 0", len(pending))
		}
		if len(out) != 1 {
			t.Fatalf("delivered %d frames, want 1", len(out))
		}
		if dropped != 3 {
			t.Fatalf("reported %d dropped frames, want 3", dropped)
		}
	})
}
