package agents

import (
	"testing"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

func TestManagedSandbox(t *testing.T) {
	h := &Harness{VM: ManagedSandbox("support")}
	if err := h.Validate(); err != nil {
		t.Fatal(err)
	}
	var call stream.Call
	h.apply(&call)
	if call.SandboxProfile != "support" || call.Sandbox != "" {
		t.Fatal(call)
	}
	h.VM = Daytona()
	h.apply(&call)
	if call.Sandbox != "daytona" || call.SandboxProfile != "" {
		t.Fatal(call)
	}
}
