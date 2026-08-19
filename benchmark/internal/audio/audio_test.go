package audio

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestUlawRoundTrip(t *testing.T) {
	original := make([]int16, 256)
	for i := range original {
		original[i] = int16((i - 128) * 200)
	}
	encoded := EncodeUlaw(original)
	decoded := DecodeUlaw(encoded)
	if len(decoded) != len(original) {
		t.Fatalf("len %d != %d", len(decoded), len(original))
	}
	var errSum float64
	for i := range original {
		d := float64(decoded[i] - original[i])
		errSum += d * d
	}
	rms := errSum / float64(len(original))
	if rms > 4e6 {
		t.Fatalf("mu-law error too high: %v", rms)
	}
}

func TestWAVRoundTrip(t *testing.T) {
	pcm := PCM{Rate: TelnyxRate, Samples: Tone(TelnyxRate, 440, 8000)}
	dir := t.TempDir()
	path := filepath.Join(dir, "tone.wav")
	if err := WriteWAV(path, pcm); err != nil {
		t.Fatal(err)
	}
	got, err := ReadWAV(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Rate != pcm.Rate || len(got.Samples) != len(pcm.Samples) {
		t.Fatalf("wav mismatch rate=%d n=%d", got.Rate, len(got.Samples))
	}
	if got.Samples[100] != pcm.Samples[100] {
		t.Fatalf("sample %d != %d", got.Samples[100], pcm.Samples[100])
	}
}

func TestEncodeWAVBuffer(t *testing.T) {
	pcm := PCM{Rate: 8000, Samples: []int16{0, 1, -1, 32767}}
	var buf bytes.Buffer
	if err := EncodeWAV(&buf, pcm); err != nil {
		t.Fatal(err)
	}
	got, err := DecodeWAV(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != 4 {
		t.Fatalf("got %d samples", len(got.Samples))
	}
}

func TestResample(t *testing.T) {
	pcm := PCM{Rate: 16000, Samples: make([]int16, 16000)}
	got := Resample(pcm, 8000)
	if got.Rate != 8000 || len(got.Samples) != 8000 {
		t.Fatalf("resampled %d @ %d", len(got.Samples), got.Rate)
	}
}

func TestDetectSpeech(t *testing.T) {
	samples := Concat(Silence(800), Tone(1600, 220, 12000), Silence(800))
	spans := DetectSpeech(samples, TelnyxRate, DefaultSpeechThreshold, DefaultHangoverMs)
	if len(spans) != 1 {
		t.Fatalf("got %d spans: %+v", len(spans), spans)
	}
	if spans[0].StartMs < 80 || spans[0].StartMs > 140 {
		t.Fatalf("start %d ms", spans[0].StartMs)
	}
}

func TestConversationNoiseAndTalker(t *testing.T) {
	babble := ConversationNoise(TelnyxRate, 7)
	if len(babble) != TelnyxRate {
		t.Fatalf("babble len %d", len(babble))
	}
	if meanSquare(babble) < 1e4 {
		t.Fatal("conversation noise too quiet")
	}
	talker := Talker(3)
	if len(talker) < TelnyxRate {
		t.Fatalf("talker too short: %d", len(talker))
	}
	bed := ScaleNoiseForSNR(babble, 10)
	frame := Tone(FrameSamples, 200, 8000)
	mixed := Add(frame, bed, 0)
	if mixed[10] == frame[10] && mixed[40] == frame[40] {
		t.Fatal("expected bed to change the frame")
	}
}

func TestMixSNR(t *testing.T) {
	speech := Tone(TelnyxRate, 200, 8000)
	noise := KitchenNoise(TelnyxRate, 1)
	mixed := MixSNR(speech, noise, 10)
	if len(mixed) != len(speech) {
		t.Fatal("mix length")
	}
	if mixed[100] == speech[100] && mixed[500] == speech[500] {
		t.Fatal("expected noise to change samples")
	}
}

func TestWriteStereoWAV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mix.wav")
	left := Tone(400, 220, 4000)
	right := Tone(200, 440, 4000)
	if err := WriteStereoWAV(path, TelnyxRate, left, right); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() < 44 {
		t.Fatal("wav empty")
	}
}
