package agent

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func quietLogger() *slog.Logger { return slog.New(slog.NewTextHandler(io.Discard, nil)) }

// floorServer answers every turn with the ruling given, after the policy it needs.
func floorServer(t *testing.T, decision string) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/info":
			_, _ = w.Write([]byte(`{"template":"t","end_of_turn":{"first_check_ms":200,"first_bar":0.7,"later_bar":0.2,"first_pause_logit":0.5,"interval_ms":300,"cap_ms":2500}}`))
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
	client := audiofloor.New("http://127.0.0.1:1", nil)
	require.Nil(t, newAudioFloor(nil, audiofloor.Act, quietLogger()))
	v := newAudioFloor(client, "", quietLogger())
	require.Nil(t, v)
	v.heard(InboundAudio{Audio: audio.PcmData{Samples: []int16{1}, SampleRate: 16000, Channels: 1}})
	turn := harness.FlowTurn{ID: "t"}
	v.listen(&turn, "p", nil)
	require.Nil(t, turn.Audio)
	v.ask(turn)
	v.ruled(harness.Decided{CandidateID: "t", Disposition: harness.Respond, Floor: harness.Continue})
	require.Nil(t, v.acting())
	v.close()
}

// Acting, the model is the flow controller's Hearer; in shadow mode the flow controller has
// none.
func TestOnlyActingIsTheFlowControllersHearer(t *testing.T) {
	client := audiofloor.New("http://127.0.0.1:1", nil)
	acting := newAudioFloor(client, audiofloor.Act, quietLogger())
	defer acting.close()
	require.NotNil(t, acting.acting())
	shadow := newAudioFloor(client, audiofloor.Shadow, quietLogger())
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

// A turn carries the participant's last 16 s, their own and nobody else's.
func TestATurnHearsItsParticipant(t *testing.T) {
	v := newAudioFloor(audiofloor.New("http://127.0.0.1:1", nil), audiofloor.Act, quietLogger())
	defer v.close()
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: []int16{1, 2}, SampleRate: 16000, Channels: 1}})
	v.heard(InboundAudio{Participant: stt.Participant{ID: "q"}, Audio: audio.PcmData{Samples: []int16{3}, SampleRate: 16000, Channels: 1}})
	turn := harness.FlowTurn{ID: "t"}
	v.listen(&turn, "p", []harness.Tool{{Name: "book_crossing"}, {Name: toolEndCall}})
	require.Equal(t, []int16{1, 2}, turn.Audio)
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
	v := newAudioFloor(audiofloor.New(srv.URL, srv.Client()), audiofloor.Shadow, quietLogger())

	start := time.Now()
	for range 50 {
		v.ask(harness.FlowTurn{ID: "t", Instructions: "x"})
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
	v := newAudioFloor(audiofloor.New(srv.URL, srv.Client()), audiofloor.Shadow, quietLogger())
	defer v.close()

	pending := func() int {
		v.mu.Lock()
		defer v.mu.Unlock()
		return len(v.turns)
	}
	v.ruled(harness.Decided{CandidateID: "first", Disposition: harness.Respond, Floor: harness.Continue})
	v.ask(harness.FlowTurn{ID: "first"})
	require.Eventually(t, func() bool { return pending() == 0 }, 2*time.Second, 5*time.Millisecond)

	v.running.Wait()
	v.ask(harness.FlowTurn{ID: "second"})
	require.Eventually(t, func() bool { return pending() == 1 }, 2*time.Second, 5*time.Millisecond)
	v.ruled(harness.Decided{CandidateID: "second", Disposition: harness.Wait, Floor: harness.Continue})
	require.Zero(t, pending())
}

// Audio of another rate is not what the model hears, and isn't kept.
func TestOnly16kMonoIsKept(t *testing.T) {
	v := newAudioFloor(audiofloor.New("http://127.0.0.1:1", nil), audiofloor.Shadow, quietLogger())
	defer v.close()
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: make([]int16, 10), SampleRate: 48000, Channels: 1}})
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: make([]int16, 10), SampleRate: 16000, Channels: 2}})
	require.Empty(t, v.ring("p").Last())
	v.heard(InboundAudio{Participant: stt.Participant{ID: "p"}, Audio: audio.PcmData{Samples: make([]int16, 10), SampleRate: 16000, Channels: 1}})
	require.Len(t, v.ring("p").Last(), 10)
}
