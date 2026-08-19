package telephony

import (
	"testing"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
)

func TestUlawPayloadRoundTrip(t *testing.T) {
	pcm := audio.Tone(audio.FrameSamples, 220, 8000)
	encoded := encodePayload(audio.EncodeUlaw(pcm))
	got, err := decodePayload(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(pcm) {
		t.Fatalf("len %d", len(got))
	}
}
