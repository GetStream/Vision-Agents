package research

import (
	"context"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestCursorProcessCanRecoverWithFreshSession(t *testing.T) {
	root, repos := fixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	start := func() *ACP {
		cmd := exec.Command(os.Args[0], "-test.run=^TestACPProcess$")
		cmd.Env = append(os.Environ(), "RESEARCH_PROTOCOL_PEER=1")
		a, err := startACP(ctx, cmd, root)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	a := start()
	a.Close()
	if a.Alive() {
		t.Fatal("closed process alive")
	}
	a = start()
	defer a.Close()
	got, err := a.Research(ctx, Request{Question: "new investigation"}, repos, "", func(string) {})
	if err != nil || got != "session-2" {
		t.Fatal(got, err)
	}
}
