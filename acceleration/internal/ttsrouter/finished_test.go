package ttsrouter

import (
	"fmt"
	"log/slog"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

func TestASessionRemembersOnlyItsLastSettledUtterances(t *testing.T) {
	session := newSession(newStubTTS(), routing.ProviderConfig{Provider: "stub", Model: "stub-model"},
		routing.Owner{CustomerID: "acme"}, routing.NewRecorder(routing.TTS, nil, nil, nil, slog.New(slog.DiscardHandler)),
		slog.New(slog.DiscardHandler))
	defer session.Close()

	for i := range finishedKept + 10 {
		session.settle(fmt.Sprintf("u%d", i))
	}

	session.mu.Lock()
	defer session.mu.Unlock()
	if len(session.finished) != finishedKept || len(session.finishedOrder) != finishedKept {
		t.Fatalf("a session kept %d settled utterances, not the last %d", len(session.finished), finishedKept)
	}
	if _, kept := session.finished["u0"]; kept {
		t.Fatal("the oldest settled utterance was kept")
	}
}
