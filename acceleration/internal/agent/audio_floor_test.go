package agent

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func quietLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// someAudio is a moment of a participant's audio.
var someAudio = harness.Audio{Type: "audio/ogg", Data: []byte("OggS"), Duration: 20 * time.Millisecond}

// floorInfo is a chorus-asr server's policy: two presets naming two measured points, the
// faster of which fails the dictation guard.
const floorInfo = `{"template":"t","end_of_turn":{"first_check_ms":200,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500,` +
	`"default":"safe","presets":{"safe":1100,"fast":500},"frontier":[` +
	`{"max_wait_ms":500,"bars":[{"from_ms":0,"bar":0.8},{"from_ms":500,"bar":0.3}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.1451,"answer_ms_median":500,"answer_ms_p90":800,"dictation_pauses_cut":0.16,"slow_counts_cut":9,"slow_counts":9,"dictation_guard":false}},` +
	`{"max_wait_ms":1100,"bars":[{"from_ms":0,"bar":0.995},{"from_ms":1100,"bar":0.2}],"checks_in_a_row":1,` +
	`"measured":{"cut_ins":0.0454,"answer_ms_median":1100,"answer_ms_p90":1100,"dictation_pauses_cut":0.1257,"slow_counts_cut":0,"slow_counts":9,"dictation_guard":true}}]}}`

// floorServer answers every turn with the ruling given, after the policy it needs.
func floorServer(t *testing.T, decision string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/info":
			_, _ = w.Write([]byte(floorInfo))
		case "/v1/agents":
			_, _ = w.Write([]byte(`{"agent":"a"}`))
		default:
			_, _ = w.Write([]byte(decision))
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// Off, or with no server, it asks nothing and keeps nothing, and every call on it is safe.
func TestAudioFloorIsOffUnlessAModeAsksIt(t *testing.T) {
	client := audiofloor.New("http://127.0.0.1:1", audiofloor.Opus, nil)
	require.Nil(t, newAudioFloor(nil, audiofloor.Act, audiofloor.EndOfTurn{}, quietLogger()))
	v := newAudioFloor(client, "", audiofloor.EndOfTurn{}, quietLogger())
	require.Nil(t, v)
	v.heard(InboundAudio{Audio: audio.PcmData{Samples: []int16{1}, SampleRate: 16000, Channels: 1}})
	turn := harness.FlowTurn{ID: "t"}
	v.listen(&turn, "p", nil)
	require.Empty(t, turn.Audio.Data)
	v.ask(turn)
	v.ruled(harness.Decided{CandidateID: "t", Disposition: harness.Respond, Floor: harness.Continue})
	require.Nil(t, v.acting())
	v.close()
}

// Acting, the model is the flow controller's Hearer; in shadow mode the flow controller has
// none.
func TestOnlyActingIsTheFlowControllersHearer(t *testing.T) {
	client := audiofloor.New("http://127.0.0.1:1", audiofloor.Opus, nil)
	acting := newAudioFloor(client, audiofloor.Act, audiofloor.EndOfTurn{}, quietLogger())
	defer acting.close()
	require.NotNil(t, acting.acting())
	shadow := newAudioFloor(client, audiofloor.Shadow, audiofloor.EndOfTurn{}, quietLogger())
	defer shadow.close()
	require.Nil(t, shadow.acting())
}

// The router's own tools don't reach the model's prompt: listing them flips its answers.
func TestOnlyACustomersToolsReachTheModel(t *testing.T) {
	got := customerTools([]harness.Tool{
		{Name: "book_crossing"}, {Name: toolEndCall}, {Name: toolTransfer}, {Name: toolPress}, {Name: toolLookup},
		{Name: toolSearch}, {Name: confirmTool}, {Name: VideoFramesTool}, {Name: "cancel_crossing"},
	})
	names := []string{}
	for _, tool := range got {
		names = append(names, tool.Name)
	}
	require.Equal(t, []string{"book_crossing", "cancel_crossing"}, names)
}

// A turn carries the participant's last 16 s, their own and nobody else's, encoded as they
// arrived.
func TestATurnHearsItsParticipant(t *testing.T) {
	v := newAudioFloor(audiofloor.New("http://127.0.0.1:1", audiofloor.PCM, nil), audiofloor.Act, audiofloor.EndOfTurn{}, quietLogger())
	defer v.close()
	frame := func(sample int16) InboundAudio {
		samples := make([]int16, 320)
		samples[0] = sample
		return InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: samples, SampleRate: 16000, Channels: 1}}
	}
	v.heard(frame(1))
	v.heard(frame(2))
	other := frame(3)
	other.Participant.ID = "q"
	v.heard(other)
	turn := harness.FlowTurn{ID: "t"}
	v.listen(&turn, "p", []harness.Tool{{Name: "book_crossing"}, {Name: toolEndCall}})
	require.Equal(t, 40*time.Millisecond, turn.Audio.Duration)
	require.Equal(t, append(frame(1).Audio.Bytes(), frame(2).Audio.Bytes()...), turn.Audio.Data)
	require.Len(t, turn.Tools, 1)
}

// A server that never answers costs the caller nothing in shadow mode: asking returns at
// once, and a turn asked about while a question is out is dropped rather than queued.
func TestAHungServerCostsACallNothing(t *testing.T) {
	var arrived atomic.Int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		arrived.Add(1)
		<-release
	}))
	defer srv.Close()
	defer close(release)
	v := newAudioFloor(audiofloor.New(srv.URL, audiofloor.Opus, srv.Client()), audiofloor.Shadow, audiofloor.EndOfTurn{}, quietLogger())

	start := time.Now()
	for range 50 {
		v.ask(harness.FlowTurn{ID: "t", Instructions: "x", Audio: someAudio})
		v.heard(InboundAudio{Audio: audio.PcmData{Samples: make([]int16, 320), SampleRate: 16000, Channels: 1}})
		v.ruled(harness.Decided{CandidateID: "t", Disposition: harness.Respond, Floor: harness.Continue})
	}
	require.Less(t, time.Since(start), 500*time.Millisecond)
	require.Eventually(t, func() bool { return arrived.Load() == 1 }, 2*time.Second, 5*time.Millisecond)
	time.Sleep(50 * time.Millisecond)
	require.EqualValues(t, 1, arrived.Load())

	// Closing a call doesn't wait out the question's deadline.
	start = time.Now()
	v.close()
	require.Less(t, time.Since(start), harness.HearDeadline(harness.FlowTurn{}))
}

// The two rulings meet whichever comes first, and nothing is left behind.
func TestTheTwoRulingsAreComparedInEitherOrder(t *testing.T) {
	srv := floorServer(t, `{"answer":"wait","answer_probability":0.8,"p_eot":0.2,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`)
	v := newAudioFloor(audiofloor.New(srv.URL, audiofloor.Opus, srv.Client()), audiofloor.Shadow, audiofloor.EndOfTurn{}, quietLogger())
	defer v.close()

	pending := func() int {
		v.mu.Lock()
		defer v.mu.Unlock()
		return len(v.turns)
	}
	v.ruled(harness.Decided{CandidateID: "first", Disposition: harness.Respond, Floor: harness.Continue})
	v.ask(harness.FlowTurn{ID: "first", Audio: someAudio})
	require.Eventually(t, func() bool { return pending() == 0 }, 2*time.Second, 5*time.Millisecond)

	v.running.Wait()
	v.ask(harness.FlowTurn{ID: "second", Audio: someAudio})
	require.Eventually(t, func() bool { return pending() == 1 }, 2*time.Second, 5*time.Millisecond)
	v.ruled(harness.Decided{CandidateID: "second", Disposition: harness.Wait, Floor: harness.Continue})
	require.Zero(t, pending())
}

// Audio of another rate is not what the model hears, and isn't kept.
func TestOnly16kMonoIsKept(t *testing.T) {
	v := newAudioFloor(audiofloor.New("http://127.0.0.1:1", audiofloor.Opus, nil), audiofloor.Shadow, audiofloor.EndOfTurn{}, quietLogger())
	defer v.close()
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: make([]int16, 960), SampleRate: 48000, Channels: 1}})
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: make([]int16, 640), SampleRate: 16000, Channels: 2}})
	require.Zero(t, v.stream("p").Frames())
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: make([]int16, 320), SampleRate: 16000, Channels: 1}})
	require.EqualValues(t, 1, v.stream("p").Frames())
}

// The shadow comparison says how the model's ruling was read: the agent's own choice, and
// whether that point passes the dictation guard, in the log line and in the metric.
func TestTheComparisonSaysWhichEndOfTurnItWasReadBy(t *testing.T) {
	srv := floorServer(t, `{"answer":"wait","answer_probability":0.4,"p_eot":0.6,"options":[],"decide_ms":1,"ruling":{"disposition":"respond","floor":"continue"}}`)
	var log bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&log, &slog.HandlerOptions{Level: slog.LevelDebug}))
	v := newAudioFloor(audiofloor.New(srv.URL, audiofloor.Opus, srv.Client()), audiofloor.Shadow, audiofloor.EndOfTurn{Preset: "fast"}, logger)
	defer v.close()
	counted := audioFloorCompared.WithLabelValues("respond", "respond", "continue", "continue", "preset:fast", "fail")
	before := testutil.ToFloat64(counted)

	v.ask(harness.FlowTurn{ID: "t", Audio: someAudio, Waited: true, Paused: 800 * time.Millisecond})
	v.running.Wait()
	v.ruled(harness.Decided{CandidateID: "t", Disposition: harness.Respond, Floor: harness.Continue})

	require.InDelta(t, before+1, testutil.ToFloat64(counted), 1e-9)
	require.Contains(t, log.String(), "end_of_turn=preset:fast dictation_guard=fail bar=0.3 in_a_row=1 checks_in_a_row=1 ended=true")
}
