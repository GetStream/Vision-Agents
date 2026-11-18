package agent

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// The server's answers, every option's probability in the prompt's order: at a pause the
// caller has finished and wants a tool, and the agent's other answers are near zero; over the
// agent they want it to stop, and no tool can do what they ask.
const (
	pauseAnswer = `{"answer":"respond","answer_probability":0.91,"p_eot":0.999,"decide_ms":41.5,` +
		`"options":[{"name":"respond","p":0.91},{"name":"wait","p":0.001},{"name":"clarify","p":0.000001},{"name":"ignore","p":0.065},` +
		`{"name":"cannot","p":0.004},{"name":"get_weather","p":0.02}],` +
		`"ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"}}`
	overAgentAnswer = `{"answer":"stop","answer_probability":0.7,"decide_ms":40,` +
		`"options":[{"name":"stop","p":0.7},{"name":"shorten","p":0.2},{"name":"continue","p":0.1},{"name":"ignore","p":0}],` +
		`"ruling":{"disposition":"respond","floor":"stop","cannot":true}}`
)

// checkServer answers a check while the agent talks (the turn says it is speaking) and one
// with the floor free as the server does, or answer to both when it is set.
func checkServer(t *testing.T, answer string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/info":
			_, _ = w.Write([]byte(floorInfo))
		case "/v1/agents":
			_, _ = w.Write([]byte(`{"agent":"a"}`))
		default:
			body, _ := io.ReadAll(r.Body)
			switch {
			case answer != "":
				_, _ = w.Write([]byte(answer))
			case bytes.Contains(body, []byte(`"speaking":true`)):
				_, _ = w.Write([]byte(overAgentAnswer))
			default:
				_, _ = w.Write([]byte(pauseAnswer))
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// recorded is a check as the call's record was handed it.
type recorded struct {
	turn  string
	check live.FloorCheck
}

// watched is an acting audio floor whose checks are kept for the call by record.
type watched struct {
	*audioFloor
	mu   sync.Mutex
	kept []recorded
}

func watchedFloor(t *testing.T, answer string) *watched {
	client := audiofloor.New(checkServer(t, answer).URL, audiofloor.Opus, nil)
	w := &watched{audioFloor: newAudioFloor(client, audiofloor.Act, audiofloor.EndOfTurn{}, quietLogger())}
	w.watch(func(turn string, check live.FloorCheck) {
		w.mu.Lock()
		defer w.mu.Unlock()
		w.kept = append(w.kept, recorded{turn: turn, check: check})
	})
	return w
}

func (w *watched) record() []recorded {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]recorded(nil), w.kept...)
}

// A pause's check and one over the agent are kept for the call with the turn each is about,
// every probability of the four answers the model rules the floor by as the server gave it,
// the others of at least 1%, the ruling and the tool it intends; and the turn carries both.
func TestAPauseCheckAndACheckOverTheAgentAreKeptWithTheirTurn(t *testing.T) {
	w := watchedFloor(t, "")
	hearer := w.acting()
	ctx := context.Background()
	began := time.Now()

	ruled, err := hearer.Hear(ctx, harness.FlowTurn{ID: "turn-1", ParticipantID: "alice", Instructions: "i", Audio: someAudio,
		Paused: 600 * time.Millisecond})
	require.NoError(t, err)
	require.Equal(t, harness.Respond, ruled.Disposition, "the pause ended the turn")
	_, err = hearer.Hear(ctx, harness.FlowTurn{ID: "over-2", ReplyID: "turn-1", ParticipantID: "alice", Instructions: "i",
		Audio: someAudio, Speaking: true, Unfinished: true, Paused: time.Second})
	require.NoError(t, err)

	kept := w.record()
	require.Len(t, kept, 2)
	pause, over := kept[0], kept[1]
	require.Equal(t, "turn-1", pause.turn)
	require.Equal(t, live.FloorCheckPause, pause.check.Kind)
	require.Equal(t, 600.0, *pause.check.PausedMs)
	require.Equal(t, []live.FloorOption{{Option: "respond", Probability: 0.91}, {Option: "wait", Probability: 0.001},
		{Option: "clarify", Probability: 0.000001}, {Option: "ignore", Probability: 0.065}, {Option: "get_weather", Probability: 0.02}},
		pause.check.Options, "the four answers whatever their probability; cannot, under 1%, left out")
	require.Equal(t, live.FloorRuling{Disposition: "respond", Floor: "continue", Tool: "get_weather"}, pause.check.Ruling)
	require.True(t, pause.check.Acted)
	require.Equal(t, "preset:safe", pause.check.Point, "the server's default preset")
	require.Positive(t, pause.check.LatencyMs)

	require.Equal(t, "turn-1", over.turn, "the reply it was asked over, not the words' own ID")
	require.Equal(t, live.FloorCheckOverAgent, over.check.Kind)
	require.Nil(t, over.check.PausedMs)
	require.Equal(t, []live.FloorOption{{Option: "stop", Probability: 0.7}, {Option: "shorten", Probability: 0.2},
		{Option: "continue", Probability: 0.1}, {Option: "ignore", Probability: 0}}, over.check.Options)
	require.Equal(t, live.FloorRuling{Disposition: "respond", Floor: "stop", Cannot: true}, over.check.Ruling)
	require.True(t, over.check.Acted, "a stop is handed to the agent")
	require.Empty(t, over.check.Point)

	require.Equal(t, []live.FloorCheck{pause.check, over.check}, w.take("turn-1", began), "the turn carries both")
	require.Nil(t, w.take("turn-1", began), "a turn's checks are carried once")
}

// Settled words put to the model while the agent talks are asked the question over the agent,
// and kept as a check over it, at the pause they were put at, about the reply.
func TestSettledWordsAskedAboutWhileTheAgentTalksAreACheckOverTheAgent(t *testing.T) {
	w := watchedFloor(t, "")
	_, err := w.acting().Hear(context.Background(), harness.FlowTurn{ID: "turn-2", ReplyID: "turn-1", ParticipantID: "alice",
		Instructions: "i", Audio: someAudio, Speaking: true, Paused: 200 * time.Millisecond})
	require.NoError(t, err)

	kept := w.record()
	require.Len(t, kept, 1)
	require.Equal(t, "turn-1", kept[0].turn)
	require.Equal(t, live.FloorCheckOverAgent, kept[0].check.Kind)
	require.Equal(t, 200.0, *kept[0].check.PausedMs)
	require.Equal(t, "stop", kept[0].check.Options[0].Option)
}

// A check reported after the call's record has closed, as one out when the call ends, is let
// go of rather than taking the router down.
func TestAFloorCheckAfterTheRecorderClosedIsLetGo(t *testing.T) {
	r := buildDecisionRecorder(nil, routing.Owner{CustomerID: "c", AgentID: "a", CallID: "call-1"}, quietLogger())
	r.send = func(context.Context, []live.CallEvent) error { return nil }
	r.start().Close()
	require.NotPanics(t, func() {
		r.RecordFloorCheck("turn-1", live.FloorCheck{At: time.Now()})
		r.Record(Decided{Kind: "wait"})
	})
}

// A check over a reply after its turn was reported, the reply still playing out, is kept for
// the call with that turn all the same: the timeline joins it there.
func TestACheckAfterItsTurnWasReportedIsKeptForTheCall(t *testing.T) {
	w := watchedFloor(t, "")
	require.Nil(t, w.take("turn-1", time.Now()), "reported at the end of its synthesis, with no check yet")

	_, err := w.acting().Hear(context.Background(), harness.FlowTurn{ID: "over-2", ReplyID: "turn-1", ParticipantID: "alice",
		Instructions: "i", Audio: someAudio, Speaking: true, Unfinished: true})
	require.NoError(t, err)

	kept := w.record()
	require.Len(t, kept, 1)
	require.Equal(t, "turn-1", kept[0].turn)
}

// A pause's check with the floor free names no turn: a turn carries those from before it began,
// which led to it, and leaves the later ones to the next.
func TestATurnCarriesThePauseChecksThatLedToIt(t *testing.T) {
	w := watchedFloor(t, "")
	at := time.Now()
	w.floorChecked(audiofloor.Checked{Participant: "alice", At: at, Paused: 300 * time.Millisecond})
	w.floorChecked(audiofloor.Checked{Participant: "alice", At: at.Add(2 * time.Second)})
	w.floorChecked(audiofloor.Checked{Participant: "alice", Turn: "turn-9", At: at, OverAgent: true})

	first := w.take("turn-1", at.Add(time.Second))
	require.Len(t, first, 1)
	require.Equal(t, 300.0, *first[0].PausedMs)
	require.Len(t, w.take("turn-2", at.Add(3*time.Second)), 1)
	require.Len(t, w.take("turn-9", at), 1, "a check about another turn waits for it")
	for _, kept := range w.record()[:2] {
		require.Empty(t, kept.turn, "kept for the call naming none, which the timeline joins by time")
	}
}

// A turn carries its latest floorChecksKept checks, oldest first, and what waits for turns
// still to be reported is bounded.
func TestATurnCarriesItsLatestFloorChecks(t *testing.T) {
	w := watchedFloor(t, "")
	start := time.Now()
	for i := range floorChecksKept + 10 {
		w.floorChecked(audiofloor.Checked{Turn: "turn-1", At: start.Add(time.Duration(i) * time.Millisecond)})
	}
	checks := w.take("turn-1", start)
	require.Len(t, checks, floorChecksKept)
	require.Equal(t, start.Add(10*time.Millisecond).UTC(), checks[0].At)
	require.Equal(t, start.Add(time.Duration(floorChecksKept+9)*time.Millisecond).UTC(), checks[floorChecksKept-1].At)
	require.Len(t, w.record(), floorChecksKept+10, "the call's record keeps them all")

	for range recentChecks + 5 {
		w.floorChecked(audiofloor.Checked{Turn: "never-reported", At: start})
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	require.Len(t, w.recent, recentChecks)
}

// A server that reports no options is read as it is, with no probabilities; one whose answer
// has no ruling is no answer, and nothing is kept of it.
func TestAServerAnswerWithNoOptionsOrNoRulingIsReadAsItIs(t *testing.T) {
	plain := watchedFloor(t, `{"answer":"respond","answer_probability":0.9,"p_eot":0.999,"ruling":{"disposition":"respond","floor":"continue"}}`)
	_, err := plain.acting().Hear(context.Background(), harness.FlowTurn{ID: "turn-1", Instructions: "i", Audio: someAudio})
	require.NoError(t, err)
	kept := plain.record()
	require.Len(t, kept, 1)
	require.Nil(t, kept[0].check.Options)
	require.Equal(t, "respond", kept[0].check.Ruling.Disposition)

	unruled := watchedFloor(t, `{"answer":"respond","answer_probability":0.9,"p_eot":0.999,"options":[{"name":"respond","p":0.9}]}`)
	_, err = unruled.acting().Hear(context.Background(), harness.FlowTurn{ID: "turn-1", Instructions: "i", Audio: someAudio})
	require.Error(t, err)
	require.Empty(t, unruled.record())
}

// Checks are reported from the flow controller's goroutine and the transcriber's at once, while
// turns are reported from the agent's: run with -race.
func TestFloorChecksAreKeptFromManyGoroutinesAtOnce(t *testing.T) {
	w := watchedFloor(t, "")
	hearer := w.acting()
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Go(func() {
			for range 10 {
				_, err := hearer.Hear(context.Background(), harness.FlowTurn{ID: "turn-1", ReplyID: "turn-0", Speaking: i%2 == 0,
					Unfinished: i%2 == 0, Instructions: "i", Audio: someAudio})
				if err != nil {
					t.Error(err)
				}
				w.take("turn-1", time.Now())
			}
		})
	}
	wg.Wait()
	require.Len(t, w.record(), 80)
}

// The recorder keeps a check apart from the judgements, with the turn it is about, and never
// waits on a writer that has fallen behind.
func TestAFloorCheckIsRecordedForTheCallWithoutWaiting(t *testing.T) {
	r := buildDecisionRecorder(nil, routing.Owner{CustomerID: "c", AgentID: "a", CallID: "call-1"}, quietLogger())
	check := live.FloorCheck{At: time.Now().UTC(), Kind: live.FloorCheckOverAgent, Acted: true}
	for range decisionQueueSize + 10 {
		r.RecordFloorCheck("turn-1", check)
	}
	require.Equal(t, int64(10), r.dropped.Load(), "dropped, not waited for")
	row := <-r.queue
	require.Equal(t, "floor_check", row.Kind)
	require.Equal(t, "turn-1", row.TurnID)
	require.Equal(t, "call-1", row.CallID)
	require.Equal(t, &check, row.FloorCheck)
}

// An agent without chorus-asr floor control makes no checks, and its turns carry none.
func TestAnAgentWithoutChorusASRFloorControlHasNoFloorChecks(t *testing.T) {
	var none *audioFloor
	none.watch(func(string, live.FloorCheck) { t.Fatal("nothing to keep") })
	require.Nil(t, none.take("turn-1", time.Now()))
	require.Nil(t, newAudioFloor(audiofloor.New(checkServer(t, "").URL, audiofloor.Opus, nil), audiofloor.Off,
		audiofloor.EndOfTurn{}, quietLogger()))
}
