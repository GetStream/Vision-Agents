package audio

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestWAVRoundTrip(t *testing.T) {
	pcm := PCM{Rate: Rate, Samples: Tone(Rate, 440, 8000)}
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
	samples := Concat(Silence(Rate/10), Tone(Rate/5, 220, 12000), Silence(Rate/10))
	spans := DetectSpeech(samples, Rate, DefaultSpeechThreshold, DefaultHangoverMs)
	if len(spans) != 1 {
		t.Fatalf("got %d spans: %+v", len(spans), spans)
	}
	if spans[0].StartMs < 80 || spans[0].StartMs > 140 {
		t.Fatalf("start %d ms", spans[0].StartMs)
	}
}

func TestConversationNoiseAndTalker(t *testing.T) {
	babble := ConversationNoise(Rate, 7)
	if len(babble) != Rate {
		t.Fatalf("babble len %d", len(babble))
	}
	if meanSquare(babble) < 1e4 {
		t.Fatal("conversation noise too quiet")
	}
	talker := Talker(3)
	if len(talker) < Rate {
		t.Fatalf("talker too short: %d", len(talker))
	}
	bed := ScaleNoiseForSNR(babble, 10)
	frame := Tone(FrameSamples, 200, 8000)
	orig10, orig40 := frame[10], frame[40]
	mixed := Add(frame, bed, 0)
	if mixed[10] == orig10 && mixed[40] == orig40 {
		t.Fatal("expected bed to change the frame")
	}
}

func TestWriteStereoWAV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mix.wav")
	left := Tone(400, 220, 4000)
	right := Tone(200, 440, 4000)
	if err := WriteStereoWAV(path, Rate, left, right); err != nil {
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
