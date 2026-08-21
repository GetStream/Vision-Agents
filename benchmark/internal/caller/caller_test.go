package caller

import (
	"context"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

func TestPlayLoopback(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate, 220, 10000))
		time.Sleep(200 * time.Millisecond)
		for range loop.AgentRecv {
		}
	}()

	text := "hello table for four"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/2, 180, 10000),
		},
		ClosingGraceMS: 300,
	}
	sc := scenario.Scenario{
		ID:   "t",
		Pack: "restaurant",
		Turns: []scenario.Turn{
			{
				ID:   "intro",
				Text: text,
				Trigger: scenario.Trigger{
					Kind:    scenario.TriggerAfterAgent,
					DelayMS: 50,
				},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Caller) == 0 {
		t.Fatal("expected caller audio")
	}
	if len(got.Events) != 1 {
		t.Fatalf("events %d", len(got.Events))
	}
}

func TestDeadlineReturnsPartialRecording(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	first := "first turn"
	second := "second turn"
	eng := Engine{Audio: map[string][]int16{
		first:  audio.Tone(audio.Rate/10, 180, 10000),
		second: audio.Tone(audio.Rate/10, 190, 10000),
	}}
	sc := scenario.Scenario{Turns: []scenario.Turn{
		{ID: "first", Text: first, Trigger: scenario.Trigger{Kind: scenario.TriggerImmediate}},
		{ID: "second", Text: second, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
	}}
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err == nil {
		t.Fatal("expected deadline")
	}
	if got.Rate != audio.Rate || len(got.Caller) == 0 || len(got.Events) != 1 {
		t.Fatalf("partial recording was not preserved: rate=%d samples=%d events=%d", got.Rate, len(got.Caller), len(got.Events))
	}
}

func TestMidReplyPauseDoesNotAdvanceTurn(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate/5, 220, 10000))
		heard := false
		for frame := range loop.AgentRecv {
			if heard || audio.FrameEnergy(frame.PCM) < audio.DefaultSpeechThreshold {
				continue
			}
			heard = true
			_ = loop.SendAgent(audio.Tone(audio.Rate/5, 210, 10000))
			time.Sleep(400 * time.Millisecond)
			_ = loop.SendAgent(audio.Tone(audio.Rate, 210, 10000))
		}
	}()

	first := "first turn"
	second := "second turn"
	eng := Engine{
		Audio: map[string][]int16{
			first:  audio.Tone(audio.Rate/4, 180, 10000),
			second: audio.Tone(audio.Rate/4, 190, 10000),
		},
		TurnHangoverMS: 1200,
		ClosingGraceMS: 300,
	}
	sc := scenario.Scenario{
		ID:   "pause",
		Pack: "restaurant",
		Turns: []scenario.Turn{
			{ID: "a", Text: first, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
			{ID: "b", Text: second, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 2 {
		t.Fatalf("events %d", len(got.Events))
	}
	gap := got.Events[1].RecStartMs - got.Events[0].RecEndMs
	if gap < 1000 {
		t.Fatalf("second turn started %dms after first in the recording; pause should have been ignored", gap)
	}
}

func TestDuringAgentOverlapPlaysWhileAgentSpeaks(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(2*audio.Rate, 220, 10000))
		for range loop.AgentRecv {
		}
	}()

	eng := Engine{ClosingGraceMS: 300}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{
				ID:           "cough",
				OverlapSound: "cough",
				Trigger: scenario.Trigger{
					Kind:    scenario.TriggerDuringAgent,
					AfterMS: 200,
				},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 1 {
		t.Fatalf("events %d", len(got.Events))
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) == 0 {
		t.Fatal("expected agent speech")
	}
	start := got.Events[0].RecStartMs
	inside := false
	for _, s := range spans {
		if start >= s.StartMs && start <= s.EndMs {
			inside = true
			break
		}
	}
	if !inside {
		t.Fatalf("cough RecStartMs=%d not inside agent spans %+v", start, spans)
	}
}

func TestBargeInWaitsForAgentSpeech(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		time.Sleep(1500 * time.Millisecond)
		_ = loop.SendAgent(audio.Tone(2*audio.Rate, 220, 10000))
		for range loop.AgentRecv {
		}
	}()

	text := "wait make it six"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/4, 180, 10000),
		},
		ClosingGraceMS: 300,
	}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{
				ID:   "change",
				Text: text,
				Trigger: scenario.Trigger{
					Kind:    scenario.TriggerBargeIn,
					AfterMS: 200,
				},
			},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 12*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 1 {
		t.Fatalf("events %d", len(got.Events))
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) == 0 {
		t.Fatal("expected agent speech")
	}
	start := got.Events[0].RecStartMs
	inside := false
	for _, s := range spans {
		if start >= s.StartMs && start <= s.EndMs {
			inside = true
			break
		}
	}
	if !inside {
		t.Fatalf("barge RecStartMs=%d not inside agent spans %+v", start, spans)
	}
}

func TestFinalWaitCapturesLateReply(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate/5, 220, 10000))
		heard := false
		for frame := range loop.AgentRecv {
			if heard || audio.FrameEnergy(frame.PCM) < audio.DefaultSpeechThreshold {
				continue
			}
			heard = true
			time.Sleep(2 * time.Second)
			_ = loop.SendAgent(audio.Tone(audio.Rate, 260, 12000))
		}
	}()

	text := "book me"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/4, 180, 10000),
		},
		ClosingGraceMS: 3000,
	}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{ID: "intro", Text: text, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) < 2 {
		t.Fatalf("expected greeting and late reply, spans=%d", len(spans))
	}
}

// The contract prompts make the agent speak filler while tools run, so it is still talking
// when the last scripted turn ends. Hanging up on that first silence drops the substantive
// reply that the entity gates score.
func TestFinalWaitCapturesReplyAfterFiller(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate/5, 220, 10000))
		heard := false
		for frame := range loop.AgentRecv {
			if heard || audio.FrameEnergy(frame.PCM) < audio.DefaultSpeechThreshold {
				continue
			}
			heard = true
			_ = loop.SendAgent(audio.Tone(audio.Rate/2, 240, 12000))
			time.Sleep(1500 * time.Millisecond)
			_ = loop.SendAgent(audio.Tone(audio.Rate, 260, 12000))
		}
	}()

	text := "book me"
	eng := Engine{
		Audio: map[string][]int16{
			text: audio.Tone(audio.Rate/4, 180, 10000),
		},
		// The hangover is short so the old "hang up on first silence" behaviour would end
		// the call ~900ms after the filler, well before the reply lands at 1500ms.
		TurnHangoverMS: 500,
		ClosingGraceMS: 2000,
	}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{ID: "intro", Text: text, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 20}},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	spans := audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs)
	if len(spans) < 3 {
		t.Fatalf("expected greeting, filler, and reply after the filler, spans=%d", len(spans))
	}
}

// The filler-then-reply pattern that TestFinalWaitCapturesReplyAfterFiller covers at the end of
// the call happens at every mid-call turn boundary too. The agent goes quiet long enough for the
// hangover to expire, resumes inside the next turn's trigger delay, and the caller plays over it
// — which marks the turn as overlapping and throws its V2V sample away. All three dropped
// samples in out/hc-stream-fix1-k3 were this.
func TestMidCallSettleWaitsOutTheReplyAfterFiller(t *testing.T) {
	loop := transport.NewLoopback()
	defer loop.Close()

	go func() {
		_ = loop.SendAgent(audio.Tone(audio.Rate/5, 220, 10000))
		heard := false
		for frame := range loop.AgentRecv {
			if heard || audio.FrameEnergy(frame.PCM) < audio.DefaultSpeechThreshold {
				continue
			}
			heard = true
			_ = loop.SendAgent(audio.Tone(audio.Rate/4, 240, 12000))
			// Longer than the hangover, shorter than the hangover plus the trigger delay.
			time.Sleep(600 * time.Millisecond)
			_ = loop.SendAgent(audio.Tone(audio.Rate, 260, 12000))
		}
	}()

	first := "first turn"
	second := "second turn"
	eng := Engine{
		Audio: map[string][]int16{
			first:  audio.Tone(audio.Rate/4, 180, 10000),
			second: audio.Tone(audio.Rate/4, 190, 10000),
		},
		TurnHangoverMS: 400,
		ClosingGraceMS: 800,
	}
	sc := scenario.Scenario{
		Turns: []scenario.Turn{
			{ID: "a", Text: first, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 500}},
			{ID: "b", Text: second, Trigger: scenario.Trigger{Kind: scenario.TriggerAfterAgent, DelayMS: 500}},
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Second)
	defer cancel()
	got, err := eng.Play(ctx, sc, loop)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Events) != 2 {
		t.Fatalf("events %d", len(got.Events))
	}
	if got.Events[1].Overlap {
		t.Fatal("second turn played over the reply that followed the filler")
	}
	start := got.Events[1].RecStartMs
	for _, s := range audio.DetectSpeech(got.Agent, got.Rate, audio.DefaultSpeechThreshold, audio.DefaultHangoverMs) {
		if s.StartMs < start && s.EndMs > start {
			t.Fatalf("second turn at %dms landed inside agent speech %+v", start, s)
		}
	}
}
