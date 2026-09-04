package target

import (
	"context"
	"reflect"
	"testing"
)

func TestNames(t *testing.T) {
	want := []string{AcceleratedName, AccelerationName, LiveKitName, PythonName}
	if got := Names(); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestBuildAccelerated(t *testing.T) {
	built, err := Build(AcceleratedName, Config{URL: "http://127.0.0.1:8000"})
	if err != nil {
		t.Fatal(err)
	}
	got, ok := built.(*Accelerated)
	if !ok {
		t.Fatalf("got %T", built)
	}
	if got.Pipeline != "accelerated" {
		t.Fatalf("pipeline %q", got.Pipeline)
	}
}

func TestAcceleratedPipelineEnvDefaults(t *testing.T) {
	t.Setenv("VOICEBENCH_MODEL", "")
	t.Setenv("VOICEBENCH_STT", "")
	t.Setenv("VOICEBENCH_TTS", "")
	t.Setenv("VOICEBENCH_SUBAGENT", "")
	t.Setenv("STREAM_ACCELERATION_CUSTOMER_ID", "")
	got := acceleratedPipelineEnv()
	want := []string{
		"VOICEBENCH_MODEL=" + DefaultAcceleratedModel,
		"VOICEBENCH_STT=" + DefaultAcceleratedSTT,
		"VOICEBENCH_TTS=" + DefaultAcceleratedTTS,
		"VOICEBENCH_SUBAGENT=" + DefaultAcceleratedSubagent,
		"STREAM_ACCELERATION_CUSTOMER_ID=voicebench",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestLiveKitPipelineEnvDefaults(t *testing.T) {
	t.Setenv("VOICEBENCH_LIVEKIT_PIPELINE", "")
	t.Setenv("VOICEBENCH_LIVEKIT_MODEL", "")
	t.Setenv("VOICEBENCH_LIVEKIT_STT", "")
	t.Setenv("VOICEBENCH_LIVEKIT_TTS", "")
	t.Setenv("VOICEBENCH_LIVEKIT_VOICE", "")
	got := liveKitPipelineEnv()
	want := []string{
		"VOICEBENCH_LIVEKIT_PIPELINE=" + DefaultLiveKitPipeline,
		"VOICEBENCH_LIVEKIT_MODEL=" + DefaultLiveKitModel,
		"VOICEBENCH_LIVEKIT_STT=" + DefaultLiveKitSTT,
		"VOICEBENCH_LIVEKIT_TTS=" + DefaultLiveKitTTS,
		"VOICEBENCH_LIVEKIT_VOICE=" + DefaultLiveKitVoice,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestRemoteTargetsRequireURL(t *testing.T) {
	for _, built := range []Target{&Python{}, &Acceleration{Instructions: "test", Tools: []AccelTool{{Name: "tool"}}}} {
		if _, err := built.Prepare(context.Background()); err == nil {
			t.Fatalf("%T accepted an empty URL", built)
		}
	}
}
