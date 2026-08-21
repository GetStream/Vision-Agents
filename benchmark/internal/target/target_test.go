package target

import (
	"context"
	"reflect"
	"testing"
)

func TestNames(t *testing.T) {
	want := []string{AccelerationName, LiveKitName, PythonName}
	if got := Names(); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestBuildPython(t *testing.T) {
	built, err := Build(PythonName, Config{URL: "http://127.0.0.1:8000"})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := built.(*Python); !ok {
		t.Fatalf("got %T", built)
	}
}

func TestRemoteTargetsRequireURL(t *testing.T) {
	for _, built := range []Target{&Python{}, &Acceleration{Instructions: "test", Tools: []AccelTool{{Name: "tool"}}}} {
		if _, err := built.Prepare(context.Background()); err == nil {
			t.Fatalf("%T accepted an empty URL", built)
		}
	}
}
