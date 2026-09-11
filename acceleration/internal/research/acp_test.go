package research

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// A real child process speaks ACP over stdio to exercise transport, session boundaries,
// extension replies, cancellation and process death without Cursor credentials.
func TestACPProcess(t *testing.T) {
	if os.Getenv("RESEARCH_PROTOCOL_PEER") == "1" {
		protocolPeer()
		os.Exit(0)
	}
	root, repos := fixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := exec.Command(os.Args[0], "-test.run=^TestACPProcess$")
	cmd.Env = append(os.Environ(), "RESEARCH_PROTOCOL_PEER=1")
	a, err := startACP(ctx, cmd, root)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	for _, expected := range []string{"session-2", "session-3"} {
		answer, err := a.Research(ctx, Request{Question: "test"}, repos, "", func(string) {})
		if err != nil || answer != expected {
			t.Fatalf("%s %v", answer, err)
		}
	}
	short, end := context.WithTimeout(ctx, 30*time.Millisecond)
	defer end()
	_, err = a.Research(short, Request{Question: "hang"}, repos, "", func(string) {})
	if err == nil || a.Alive() {
		t.Fatal("cancellation did not kill unhealthy process", err)
	}
}
func protocolPeer() {
	scan := bufio.NewScanner(os.Stdin)
	n := 0
	for scan.Scan() {
		var m rpcMessage
		_ = json.Unmarshal(scan.Bytes(), &m)
		if m.Method == "" {
			continue
		}
		result := any(map[string]any{})
		switch m.Method {
		case "session/new":
			n++
			result = map[string]string{"sessionId": fmt.Sprintf("session-%d", n)}
		case "session/prompt":
			if strings.Contains(string(m.Params), "hang") {
				continue
			}
			var p struct {
				SessionID string `json:"sessionId"`
			}
			_ = json.Unmarshal(m.Params, &p)
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"jsonrpc": "2.0", "id": 900, "method": "cursor/ask_question", "params": map[string]any{}})
			if !scan.Scan() {
				return
			}
			var reply rpcMessage
			_ = json.Unmarshal(scan.Bytes(), &reply)
			if !strings.Contains(string(reply.Result), "skipped") {
				return
			}
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"jsonrpc": "2.0", "method": "session/update", "params": map[string]any{"sessionId": p.SessionID, "update": map[string]any{"sessionUpdate": "agent_message_chunk", "content": map[string]string{"type": "text", "text": p.SessionID}}}})
		}
		if len(m.ID) > 0 {
			_ = json.NewEncoder(os.Stdout).Encode(map[string]any{"jsonrpc": "2.0", "id": m.ID, "result": result})
		}
	}
}
func TestACPPermissions(t *testing.T) {
	root, repos := fixture(t)
	a := &ACP{root: root, repos: repos[:1]}
	for _, kind := range []string{"execute", "edit", "fetch", "other"} {
		raw := json.RawMessage(fmt.Sprintf(`{"toolCall":{"kind":%q,"locations":[{"path":%q}]},"options":[{"kind":"allow_once","optionId":"yes"}]}`, kind, root+"/react/client.ts"))
		encoded, _ := json.Marshal(a.permission(raw))
		if !strings.Contains(string(encoded), "cancelled") {
			t.Fatal(string(encoded))
		}
	}
	for _, path := range []string{"/etc/passwd", root + "/swiftui/client.ts", root + "/react/../swiftui/client.ts"} {
		if _, err := a.scopedRead(path); err == nil {
			t.Fatal(path)
		}
	}
	raw := json.RawMessage(fmt.Sprintf(`{"toolCall":{"kind":"read","locations":[{"path":%q}]},"options":[{"kind":"allow_once","optionId":"yes"}]}`, root+"/react/client.ts"))
	encoded, _ := json.Marshal(a.permission(raw))
	if !strings.Contains(string(encoded), "selected") {
		t.Fatal(string(encoded))
	}
}
