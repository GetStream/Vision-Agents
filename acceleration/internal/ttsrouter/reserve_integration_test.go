//go:build integration

package ttsrouter

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy/policytest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// reservingSession is a session over a stub voice whose every character costs $1, for an
// organization with credit micros, reserving through the real enforcer. perSentence makes it a
// voice that takes each sentence as its own request.
func reservingSession(t *testing.T, money *policytest.Money, credit int64, perSentence bool) (*Session, *stubTTS, routing.Owner) {
	t.Helper()
	owner := money.Owner(t, credit)
	session, provider := reservingSessionOf(t, money, owner, perSentence)
	return session, provider, owner
}

// reservingSessionOf is a reservingSession for an owner that already has its credit.
func reservingSessionOf(t *testing.T, money *policytest.Money, owner routing.Owner, perSentence bool) (*Session, *stubTTS) {
	t.Helper()
	config := routing.ProviderConfig{Provider: "stub", Model: "stub-model", Price: routing.Price{PerMillionChars: 1_000_000}}
	prices, err := routing.Prices(routing.Config{routing.TTS: {Providers: []routing.ProviderConfig{config}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	require.NoError(t, err)
	provider := newStubTTS()
	provider.perSentence = perSentence
	session := newSession(provider, config, owner, money.Recorder(routing.TTS, prices), slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Cleanup(func() { _ = session.Close() })
	return session, provider
}

// rowsOf waits for a customer's want rows and returns them.
func rowsOf(t *testing.T, money *policytest.Money, owner routing.Owner, want int) []store.Request {
	t.Helper()
	var rows []store.Request
	require.Eventually(t, func() bool {
		rows = nil
		require.NoError(t, money.Store.DB().NewSelect().Model(&rows).Where("customer_id = ?", owner.CustomerID).Order("id").Scan(context.Background()))
		return len(rows) == want
	}, 5*time.Second, 10*time.Millisecond, "%d rows", want)
	return rows
}

func TestClosingASessionReleasesWhatItsUtterancesHeld(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, _, owner := reservingSession(t, money, 1_000_000, false)

	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
	require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "its one character is held")

	require.NoError(t, session.Close())

	require.Eventually(t, func() bool { return money.Spend(t, owner) == nil }, 2*time.Second, 10*time.Millisecond,
		"an utterance that never completed holds nothing once the session is closed")
}

func TestClosingASessionNobodyReadsStillReleasesWhatItHeld(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, owner := reservingSession(t, money, 1_000_000, false)
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
	// The caller stopped reading: the provider's events fill every buffer between them.
	sent := make(chan struct{})
	go func() {
		defer close(sent)
		for i := range 500 {
			provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "other-" + strconv.Itoa(i)})
		}
	}()
	require.Eventually(t, func() bool { return len(session.Events()) == cap(session.Events()) }, 2*time.Second, 5*time.Millisecond)
	require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits)

	require.NoError(t, session.Close())

	require.Eventually(t, func() bool { return money.Spend(t, owner) == nil }, 2*time.Second, 10*time.Millisecond,
		"what the unfinished utterance held is let go though nobody drains the events")
	<-sent
}

func TestClosingASessionReleasesEverySentenceOfAnId(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, owner := reservingSession(t, money, 9_000_000, true)
	other, _ := reservingSessionOf(t, money, owner, true)
	require.NoError(t, other.Synthesize(tts.Request{ID: "o1", Text: "yz"}))
	// Two sentences of u1, neither completed: the one cut by the barge-in and the one after.
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	require.NoError(t, session.Interrupt())
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}))
	require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "$10 is held of $9")

	require.NoError(t, session.Close())

	require.Eventually(t, func() bool { return money.Spend(t, owner) == nil }, 2*time.Second, 10*time.Millisecond,
		"both sentences' $8 is let go, and only the other session's $2 is still held")
}

func TestTextTheVoiceRefusedIsNoLongerHeld(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, owner := reservingSession(t, money, 2_000_000, false)
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
	require.NoError(t, money.Spend(t, owner), "$1 of $2 is held")

	provider.refuse = errors.New("the voice dropped the connection")
	require.Error(t, session.Synthesize(tts.Request{ID: "u1", Text: "b"}))

	require.NoError(t, money.Spend(t, owner), "what is held is the one character sent, not the two")
	provider.refuse = errors.New("still down")
	require.Error(t, session.Synthesize(tts.Request{ID: "u1", Text: "cd"}))
	require.NoError(t, money.Spend(t, owner))
}

// spoken passes on the sentence's start and end as the voice reports them, and waits until
// the session has settled it.
func spoken(t *testing.T, session *Session, provider *stubTTS, id string, characters int64) {
	t.Helper()
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: id, At: time.Now()})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: id, Characters: characters})
	deadline := time.After(5 * time.Second)
	for {
		select {
		case event, open := <-session.Events():
			if !open {
				t.Fatal("the session closed before the sentence settled")
			}
			if complete, ok := event.(tts.SynthesisComplete); ok && complete.SynthesisID == id {
				return
			}
		case <-deadline:
			t.Fatal("the sentence did not settle within 5s")
		}
	}
}

// voiceKinds is a voice that streams text into one utterance and one that takes each
// sentence as its own request, by whether it is per-sentence. Text for an id whose utterance
// settled is a new sentence on both.
var voiceKinds = map[string]bool{"streaming": false, "per-sentence": true}

func TestAnIdReusedAfterItsSentenceFinishedHoldsItsText(t *testing.T) {
	for name, perSentence := range voiceKinds {
		t.Run(name, func(t *testing.T) {
			money := policytest.Start(t, "ttsrouter_reserve")
			session, provider, owner := reservingSession(t, money, 3_000_000, perSentence)
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
			spoken(t, session, provider, "u1", 1)
			require.NoError(t, money.Spend(t, owner), "$1 of $3 is spent")

			// The voice speaks it as a new sentence, so its 7 characters are held as a fresh
			// id's are: there is room for it, and none for anything after it.
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}))
			require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "the reused id's 7 characters are held")
			require.ErrorIs(t, session.Synthesize(tts.Request{ID: "u2", Text: "i", Final: true}), policy.ErrNoCredits)
			require.Len(t, provider.said, 2, "the sentence refused for credit is never sent")
		})
	}
}

func TestAnIdReusedAfterItsSentenceFinishedIsRefusedOnceNoCreditIsLeft(t *testing.T) {
	for name, perSentence := range voiceKinds {
		t.Run(name, func(t *testing.T) {
			money := policytest.Start(t, "ttsrouter_reserve")
			session, provider, owner := reservingSession(t, money, 1_000_000, perSentence)
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
			spoken(t, session, provider, "u1", 1)

			require.ErrorIs(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}), policy.ErrNoCredits)
			require.Len(t, provider.said, 1, "the refused sentence is not sent")
			session.mu.Lock()
			_, left := session.sentences["u1"]
			session.mu.Unlock()
			require.False(t, left, "nothing is left counted or held for the refused sentence")
			require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "$1 of $1 is spent")
		})
	}
}

// After a barge-in the voice takes the same id again before the interrupted sentence's
// completion has come back.
func TestAnIdReusedAfterItsFinalTextIsRefusedOnceNoCreditIsLeft(t *testing.T) {
	for name, perSentence := range voiceKinds {
		t.Run(name, func(t *testing.T) {
			money := policytest.Start(t, "ttsrouter_reserve")
			session, provider, _ := reservingSession(t, money, 1_000_000, perSentence)
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
			provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
			require.NoError(t, session.Interrupt())

			require.ErrorIs(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}), policy.ErrNoCredits,
				"the $1 of credit is held for the sentence sent")
			require.Len(t, provider.said, 1, "the refused sentence is not sent")
			session.mu.Lock()
			left := session.sentences["u1"]
			session.mu.Unlock()
			require.Len(t, left, 1, "only the sentence sent is waiting for its completion")
			require.EqualValues(t, 1, left[0].characters)
		})
	}
}

func TestAnIdReusedAfterABargeInCutItsUnfinishedSentenceIsRefusedOnceNoCreditIsLeft(t *testing.T) {
	for name, perSentence := range voiceKinds {
		t.Run(name, func(t *testing.T) {
			money := policytest.Start(t, "ttsrouter_reserve")
			session, provider, owner := reservingSession(t, money, 2_000_000, perSentence)
			other, _ := reservingSessionOf(t, money, owner, perSentence)
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
			require.NoError(t, other.Synthesize(tts.Request{ID: "o1", Text: "y"}))
			provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
			require.NoError(t, session.Interrupt())

			if perSentence {
				// The voice throws away a sentence whose final text never came, and never
				// completes it.
				require.NoError(t, money.Spend(t, owner), "nothing is held for the sentence thrown away")
				require.NoError(t, other.Synthesize(tts.Request{ID: "o2", Text: "zz"}))
			}
			require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits, "no credit is left")
			require.ErrorIs(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}), policy.ErrNoCredits,
				"the barge-in ended the sentence, so the text after it is a new one")
			require.Len(t, provider.said, 1, "the refused sentence is not sent")
		})
	}
}

// A streaming voice speaks text sent after an utterance's final text in that utterance, and
// completes it once.
func TestTextAfterAStreamingUtterancesFinalTextIsBilledInItsOneCompletion(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, owner := reservingSession(t, money, 10_000_000, false)
	go func() {
		for range session.Events() {
		}
	}()
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}))
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "u1", Characters: 8})
	rows := rowsOf(t, money, owner, 1)
	require.EqualValues(t, 8, rows[0].Characters, "the utterance is billed both texts")
	// Once it settled, the id is a new utterance with a completion of its own.
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "x", Final: true}))
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "u1", Characters: 1})

	rows = rowsOf(t, money, owner, 2)
	require.EqualValues(t, 1, rows[1].Characters, "the new utterance is billed its own text")
	session.mu.Lock()
	left := len(session.sentences)
	session.mu.Unlock()
	require.Zero(t, left, "nothing is left waiting for a completion")
	// $9 of $10 is spent and nothing is still held for it, so one more character is the last.
	require.NoError(t, money.Spend(t, owner))
	require.NoError(t, session.Synthesize(tts.Request{ID: "u3", Text: "j", Final: true}))
	require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits)
}

func TestASentenceSpokenAgainAfterABargeInIsBilledOnItsOwn(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, owner := reservingSession(t, money, 9_000_000, true)
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	require.NoError(t, session.Interrupt())
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}))
	go func() {
		for range session.Events() {
		}
	}()
	// The voice starts the new sentence before the interrupted one's completion comes back.
	provider.emitter.Send(tts.SynthesisStarted{SynthesisID: "u1", At: time.Now()})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "u1", Characters: 1, Interrupted: true})
	provider.emitter.Send(tts.SynthesisComplete{SynthesisID: "u1", Characters: 9})

	var rows []store.Request
	require.Eventually(t, func() bool {
		rows = nil
		require.NoError(t, money.Store.DB().NewSelect().Model(&rows).Where("customer_id = ?", owner.CustomerID).Order("id").Scan(context.Background()))
		return len(rows) == 2
	}, 5*time.Second, 10*time.Millisecond, "each sentence is a row")
	require.EqualValues(t, 1, rows[0].Characters)
	require.EqualValues(t, 7, rows[1].Characters, "the new sentence is billed what was sent")
	require.NotEqual(t, rows[0].UsageID, rows[1].UsageID)

	// $8 of $9 is spent and nothing is still held for it, so one more character is the last.
	require.NoError(t, money.Spend(t, owner))
	require.NoError(t, session.Synthesize(tts.Request{ID: "u3", Text: "j", Final: true}))
	require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits)
}

func TestMoreTextForAReusedIdsNewSentenceLetsItFinish(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, _ := reservingSession(t, money, 2_000_000, false)
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
	spoken(t, session, provider, "u1", 1)
	// $1 is spent and the new sentence's first character holds the other.

	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "b"}))
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "cd"}),
		"the new sentence is speaking and its final text was not sent, so it finishes past its credit")
	require.Len(t, provider.said, 3)
}

// On a voice that takes each sentence as its own request, a final text the voice took ends
// the sentence; one it refused does not.
func TestASentenceWhoseFinalTextWasRefusedTakesItAgainAsItsOwn(t *testing.T) {
	money := policytest.Start(t, "ttsrouter_reserve")
	session, provider, _ := reservingSession(t, money, 1_000_000, true)
	// Its first character holds the $1 of credit.
	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a"}))
	provider.refuse = errors.New("the voice dropped the connection")
	require.Error(t, session.Synthesize(tts.Request{ID: "u1", Text: "bc", Final: true}))
	provider.refuse = nil

	require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "bc", Final: true}),
		"the final text never went out, so the sentence is still speaking and finishes past its credit")
	require.Len(t, provider.said, 2)
}

func TestAnIdReusedAfterItsSentenceFinishedIsBilledOnceForWhatWasSent(t *testing.T) {
	for name, perSentence := range voiceKinds {
		t.Run(name, func(t *testing.T) {
			money := policytest.Start(t, "ttsrouter_reserve")
			session, provider, owner := reservingSession(t, money, 9_000_000, perSentence)
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "a", Final: true}))
			spoken(t, session, provider, "u1", 1)
			require.NoError(t, session.Synthesize(tts.Request{ID: "u1", Text: "bcdefgh", Final: true}))
			require.Len(t, provider.said, 2, "the reused id is spoken")
			// The adapter counts 9; the row bills the 7 handed to it.
			spoken(t, session, provider, "u1", 9)

			var rows []store.Request
			require.Eventually(t, func() bool {
				rows = nil
				require.NoError(t, money.Store.DB().NewSelect().Model(&rows).Where("customer_id = ?", owner.CustomerID).Order("id").Scan(context.Background()))
				return len(rows) == 2
			}, 5*time.Second, 10*time.Millisecond)
			require.EqualValues(t, 1, rows[0].Characters)
			require.EqualValues(t, 7, rows[1].Characters, "the reused id is billed what was sent")
			require.NotEqual(t, rows[0].UsageID, rows[1].UsageID, "each sentence is its own unit of work")

			// $8 of $9 is spent and nothing is still held for it, so one more character is the last.
			require.NoError(t, money.Spend(t, owner))
			require.NoError(t, session.Synthesize(tts.Request{ID: "u3", Text: "j", Final: true}))
			require.ErrorIs(t, money.Spend(t, owner), policy.ErrNoCredits)
		})
	}
}
