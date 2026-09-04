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

func TestRemoteTargetsRequireURL(t *testing.T) {
	for _, built := range []Target{&Python{}, &Acceleration{Instructions: "test", Tools: []AccelTool{{Name: "tool"}}}} {
		if _, err := built.Prepare(context.Background()); err == nil {
			t.Fatalf("%T accepted an empty URL", built)
		}
	}
}
