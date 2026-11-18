package agent

import (
	"bytes"
	"context"
	"fmt"
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
)

// The server's answers, with every option's probability in the prompt's order: at a pause
// the caller has finished and wants a tool, and over the agent they want it to stop, and no
// tool can do what they ask.
const (
	pauseAnswer = `{"answer":"respond","answer_probability":0.91,"p_eot":0.999,"decide_ms":41.5,` +
		`"options":[{"name":"respond","p":0.91},{"name":"wait","p":0.001},{"name":"clarify","p":0},{"name":"ignore","p":0.089}],` +
		`"ruling":{"disposition":"respond","floor":"continue","tool":"get_weather"}}`
	overAgentAnswer = `{"answer":"stop","answer_probability":0.7,"decide_ms":40,` +
		`"options":[{"name":"stop","p":0.7},{"name":"shorten","p":0.2},{"name":"continue","p":0.1},{"name":"ignore","p":0}],` +
		`"ruling":{"disposition":"respond","floor":"stop","cannot":true}}`
)

// checkServer answers a check over the agent (a turn unfinished) and a pause's alike as the
// server does.
func checkServer(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/info":
			_, _ = w.Write([]byte(floorInfo))
		case "/v1/agents":
			_, _ = w.Write([]byte(`{"agent":"a"}`))
		default:
			body, _ := io.ReadAll(r.Body)
			if bytes.Contains(body, []byte(`"unfinished":true`)) {
				_, _ = w.Write([]byte(overAgentAnswer))
				return
			}
			_, _ = w.Write([]byte(pauseAnswer))
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// watched is an acting audio floor whose checks are reported to it, the agent talking over
// speaking's turn.
type watched struct {
	*audioFloor
	mu       sync.Mutex
	speaking string
	sent     []FloorChecked
}

func watchedFloor(t *testing.T) *watched {
	client := audiofloor.New(checkServer(t).URL, audiofloor.Opus, nil)
	w := &watched{audioFloor: newAudioFloor(client, audiofloor.Act, audiofloor.EndOfTurn{}, quietLogger())}
	w.watch(func() string {
		w.mu.Lock()
		defer w.mu.Unlock()
		return w.speaking
	}, func(event Event) {
		w.mu.Lock()
		defer w.mu.Unlock()
		w.sent = append(w.sent, event.(FloorChecked))
	})
	return w
}

// A pause's check and one over the agent join the turn they are about with every option's
// probability as the server gave it, the ruling and the tool it intends, and each is reported
// live as it joins.
func TestAPauseCheckAndACheckOverTheAgentJoinTheirTurn(t *testing.T) {
	w := watchedFloor(t)
	hearer := w.acting()
	ctx := context.Background()

	ruled, err := hearer.Hear(ctx, harness.FlowTurn{ID: "turn-1", ParticipantID: "alice", Instructions: "i", Audio: someAudio,
		Paused: 600 * time.Millisecond})
	require.NoError(t, err)
	require.Equal(t, harness.Respond, ruled.Disposition, "the pause ended the turn")
	w.mu.Lock()
	w.speaking = "turn-1"
	w.mu.Unlock()
	_, err = hearer.Hear(ctx, harness.FlowTurn{ID: "over-turn-2", ParticipantID: "alice", Instructions: "i", Audio: someAudio,
		Speaking: true, Unfinished: true, Paused: time.Second})
	require.NoError(t, err)

	checks := w.take("turn-1")
	require.Len(t, checks, 2)
	pause, over := checks[0], checks[1]
	require.Equal(t, live.FloorCheckPause, pause.Kind)
	require.Equal(t, 600.0, *pause.PausedMs)
	require.Equal(t, "respond", pause.Answer)
	require.Equal(t, 0.91, pause.AnswerProbability)
	require.Equal(t, 0.999, *pause.PEOT)
	require.Equal(t, 41.5, *pause.DecideMs)
	require.Equal(t, []live.FloorOption{{Option: "respond", Probability: 0.91}, {Option: "wait", Probability: 0.001},
		{Option: "clarify", Probability: 0}, {Option: "ignore", Probability: 0.089}}, pause.Options)
	require.Equal(t, live.FloorRuling{Disposition: "respond", Floor: "continue", Tool: "get_weather"}, pause.Ruling)
	require.True(t, pause.Acted)
	require.Equal(t, "preset:safe", pause.Point, "the server's default preset")
	require.Positive(t, pause.LatencyMs)

	require.Equal(t, live.FloorCheckOverAgent, over.Kind, "joined to the reply it talked over, not its own ID")
	require.Nil(t, over.PausedMs)
	require.Nil(t, over.PEOT)
	require.Equal(t, []live.FloorOption{{Option: "stop", Probability: 0.7}, {Option: "shorten", Probability: 0.2},
		{Option: "continue", Probability: 0.1}, {Option: "ignore", Probability: 0}}, over.Options)
	require.Equal(t, live.FloorRuling{Disposition: "respond", Floor: "stop", Cannot: true}, over.Ruling)
	require.True(t, over.Acted, "a stop is handed to the agent")
	require.Empty(t, over.Point)

	w.mu.Lock()
	defer w.mu.Unlock()
	require.Len(t, w.sent, 2)
	for i, sent := range w.sent {
		require.Equal(t, "turn-1", sent.TurnID)
		require.Equal(t, "alice", sent.Participant)
		require.Equal(t, checks[i], sent.Check)
	}
	require.Nil(t, w.take("turn-1"), "a turn's checks are taken once")
}

// The transcriber's own checks of a pause wait for the turn the participant's words are put
// to the flow controller as, and are reported then; words still coming over the agent are no
// such turn.
func TestATranscribersPauseChecksJoinTheParticipantsNextTurn(t *testing.T) {
	w := watchedFloor(t)
	at := time.Now()
	w.floorChecked(audiofloor.Checked{Participant: "alice", At: at, Paused: 300 * time.Millisecond,
		Answer: audiofloor.Answer{Answer: "wait", AnswerProbability: 0.8, Ruling: &audiofloor.Ruling{Disposition: "respond", Floor: "continue"}}})
	w.floorChecked(audiofloor.Checked{Participant: "bob", At: at})
	require.Empty(t, w.sent, "no turn yet")

	provisional := harness.FlowTurn{ID: "over-1", Unfinished: true, Speaking: true}
	w.listen(&provisional, "alice", "", nil)
	require.Empty(t, w.sent)
	turn := harness.FlowTurn{ID: "turn-1"}
	w.listen(&turn, "alice", "", nil)

	checks := w.take("turn-1")
	require.Len(t, checks, 1, "alice's alone")
	require.Equal(t, 300.0, *checks[0].PausedMs)
	require.Equal(t, "wait", checks[0].Answer)
	require.False(t, checks[0].Acted)
	require.Len(t, w.sent, 1)
	require.Equal(t, "turn-1", w.sent[0].TurnID)

	w.forget("bob")
	other := harness.FlowTurn{ID: "turn-2"}
	w.listen(&other, "bob", "", nil)
	require.Nil(t, w.take("turn-2"), "a participant who left takes their checks with them")
}

// A turn keeps its latest floorChecksKept checks, oldest first, and only floorCheckTurns turns
// wait for their rows.
func TestATurnKeepsItsLatestFloorChecks(t *testing.T) {
	w := watchedFloor(t)
	start := time.Now()
	for i := range floorChecksKept + 10 {
		w.floorChecked(audiofloor.Checked{Participant: "alice", Turn: "turn-1", At: start.Add(time.Duration(i) * time.Millisecond)})
	}
	checks := w.take("turn-1")
	require.Len(t, checks, floorChecksKept)
	require.Equal(t, start.Add(10*time.Millisecond).UTC(), checks[0].At)
	require.Equal(t, start.Add(time.Duration(floorChecksKept+9)*time.Millisecond).UTC(), checks[floorChecksKept-1].At)

	for i := range floorCheckTurns + 1 {
		w.floorChecked(audiofloor.Checked{Participant: "alice", Turn: fmt.Sprint("t", i), At: start})
	}
	require.Nil(t, w.take("t0"), "the oldest turn waiting is let go")
	require.Len(t, w.take(fmt.Sprint("t", floorCheckTurns)), 1)
}

// An agent without chorus-asr floor control makes no checks, and its turns carry none.
func TestAnAgentWithoutChorusASRFloorControlHasNoFloorChecks(t *testing.T) {
	var none *audioFloor
	none.watch(func() string { return "turn-1" }, func(Event) { t.Fatal("nothing to report") })
	require.Nil(t, none.take("turn-1"))
	require.Nil(t, newAudioFloor(audiofloor.New(checkServer(t).URL, audiofloor.Opus, nil), audiofloor.Off, audiofloor.EndOfTurn{}, quietLogger()))
}
